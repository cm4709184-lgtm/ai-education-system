#!/usr/bin/env python3
"""
MCU Mel频谱实时接收 + PC端模型推理迭代工具
从MCU串口接收Mel频谱数据，实时进行模型推理并显示结果

说明: 此处 DSCNN 必须与 model_tab.py 中的新结构（DS-CNN + SE + 拒识）保持一致，
      否则加载新训练的权重会因 missing/unexpected key 报错。
"""

import sys, time, struct, threading, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import serial

FRAME_START = 0x7E  # 帧起始标记
TYPE_MEL = 0x01  # Mel数据帧类型

MEL_FILTERS = 20  # Mel滤波器数量
FRAME_WINDOW = 32  # 帧窗口大小
HOP_SIZE = 256  # 帧移
SAMPLE_RATE = 16000  # 采样率


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力块（与训练端一致）"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        # x: (B, C, T) → 全局平均池化 → (B, C)
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class DSCNN(nn.Module):
    """深度可分离卷积神经网络（DS-CNN）+ SE 通道注意力（与训练端 / MCU 端对齐）。"""
    def __init__(self, num_classes, mel_filters=20, conv1_out=32, conv2_out=48, conv3_out=64,
                 conf_threshold=0.7, margin_threshold=0.2, noise_index=None):
        super().__init__()
        self.conv1 = nn.Conv1d(mel_filters, conv1_out, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.depth1 = nn.Conv1d(conv1_out, conv1_out, 3, padding=1, groups=conv1_out)
        self.bn_d1 = nn.BatchNorm1d(conv1_out)
        self.point1 = nn.Conv1d(conv1_out, conv2_out, 1)
        self.bn_p1 = nn.BatchNorm1d(conv2_out)
        self.se1 = SEBlock(conv2_out)
        self.depth2 = nn.Conv1d(conv2_out, conv2_out, 3, padding=1, groups=conv2_out)
        self.bn_d2 = nn.BatchNorm1d(conv2_out)
        self.point2 = nn.Conv1d(conv2_out, conv3_out, 1)
        self.bn_p2 = nn.BatchNorm1d(conv3_out)
        self.se2 = SEBlock(conv3_out)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(conv3_out, num_classes)
        self.pool = nn.MaxPool1d(2)
        # 拒识门限（与 MCU softmax_with_reject 行为一致）
        self.conf_threshold = conf_threshold
        self.margin_threshold = margin_threshold
        self.noise_index = noise_index if noise_index is not None else num_classes - 1

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn_d1(self.depth1(x)))
        x = self.pool(self.se1(F.relu(self.bn_p1(self.point1(x)))))
        x = F.relu(self.bn_d2(self.depth2(x)))
        x = self.se2(F.relu(self.bn_p2(self.point2(x))))
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

    def predict(self, x):
        """带拒识的推理：与 MCU 端 softmax_with_reject 行为一致。"""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(probs, 2, dim=-1)
        p1, p2 = top2.values[:, 0], top2.values[:, 1]
        idx = top2.indices[:, 0]
        uncertain = (p1 < self.conf_threshold) | ((p1 - p2) < self.margin_threshold)
        noise_t = torch.tensor(self.noise_index, device=idx.device, dtype=idx.dtype)
        idx = torch.where(uncertain, noise_t, idx)
        return idx, probs


class MelReceiver:
    """串口Mel频谱数据接收器，后台线程解析协议帧"""
    def __init__(self, port, baud=921600):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.mel_buffer = []
        self.frame_count = 0
        self.running = True
        self.lock = threading.Lock()

    def start(self):
        """启动接收线程"""
        self.thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.thread.start()

    def _rx_loop(self):
        """接收循环：解析串口数据，提取Mel帧"""
        buf = b''
        while self.running:
            try:
                data = self.ser.read(512)
                if not data:
                    continue
                buf += data
                while len(buf) >= 4:
                    idx = buf.find(FRAME_START)
                    if idx < 0:
                        buf = buf[-3:]
                        break
                    buf = buf[idx:]
                    if len(buf) < 3:
                        break
                    ftype = buf[1]
                    flen = buf[2]
                    if len(buf) < 3 + flen:
                        break
                    payload = buf[3:3+flen]
                    buf = buf[3+flen:]
                    if ftype == TYPE_MEL and flen == MEL_FILTERS:
                        mel = np.array([b if b < 128 else b - 256 for b in payload],
                                       dtype=np.float32)
                        with self.lock:
                            self.mel_buffer.append(mel)
                            self.frame_count += 1
            except Exception:
                pass

    def get_frames(self):
        """获取并清空缓冲区中的Mel帧"""
        with self.lock:
            frames = self.mel_buffer[:]
            self.mel_buffer.clear()
        return frames

    def stop(self):
        """停止接收并关闭串口"""
        self.running = False
        self.ser.close()


def load_model(pth_path):
    """加载PyTorch模型，自动检测类别数 + 拒识门限。"""
    state = torch.load(pth_path, map_location='cpu')
    fc_w = [k for k in state if 'fc.weight' in k]
    if fc_w:
        num_classes = state[fc_w[0]].shape[0]
    else:
        num_classes = 7
    model = DSCNN(num_classes)
    # 优先 strict=False 以兼容老权重（仅缺 SE 时回退到无 SE 模型）
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        # 如果权重里没有 SE（旧模型），用 strict=False 兜底
        if 'se1' in str(e) or 'se2' in str(e):
            model = DSCNN(num_classes)
            model.load_state_dict(state, strict=False)
            print(f"[警告] 权重缺少 SE 键，按无 SE 模型加载（精度会下降）")
        else:
            raise
    model.eval()
    return model, num_classes


def load_config(pth_path, num_classes):
    """尝试读取 model_config.json 里的 conf/margin 门限。"""
    import json
    import os
    candidates = [
        pth_path.replace('.pth', '.json').replace('weights', 'reports').replace('model_weights', 'model_config'),
        os.path.join(os.path.dirname(pth_path), '..', 'evaluation', 'model_config.json'),
        os.path.join(os.path.dirname(os.path.dirname(pth_path)), 'evaluation', 'model_config.json'),
    ]
    cfg = {
        "conf_threshold": 0.7,
        "margin_threshold": 0.2,
        "noise_index": num_classes - 1,
        "class_names": [f'C{i}' for i in range(num_classes)],
    }
    for c in candidates:
        if os.path.isfile(c):
            try:
                with open(c, 'r', encoding='utf-8') as f:
                    user_cfg = json.load(f)
                cfg.update(user_cfg)
                break
            except Exception:
                pass
    return cfg


def print_mel_heatmap(mel_window):
    """在终端打印Mel频谱热力图（ASCII艺术）"""
    rows, cols = mel_window.shape
    min_v = mel_window.min()
    max_v = mel_window.max()
    rng = max_v - min_v if max_v > min_v else 1.0
    blocks = [' ', '░', '▒', '▓', '█']
    for f in range(rows - 1, -1, -1):
        line = f"M{f:02d}|"
        for t in range(cols):
            norm = (mel_window[f, t] - min_v) / rng
            idx = min(int(norm * len(blocks)), len(blocks) - 1)
            line += blocks[idx]
        line += "|"
        print(line)


def main():
    """主函数：接收Mel数据并实时推理显示"""
    parser = argparse.ArgumentParser(description='MCU Mel + PC推理迭代')
    parser.add_argument('-p', '--port', default='COM10')
    parser.add_argument('-m', '--model', default='runs/latest/weights/model_weights.pth')
    parser.add_argument('-t', '--threshold', type=float, default=None,
                        help='手动覆盖 conf_threshold（默认读 cfg）')
    args = parser.parse_args()

    print("加载模型...")
    model, num_classes = load_model(args.model)
    print(f"  类别数: {num_classes}")

    cfg = load_config(args.model, num_classes)
    conf_threshold = args.threshold if args.threshold is not None else cfg.get('conf_threshold', 0.7)
    margin_threshold = cfg.get('margin_threshold', 0.2)
    noise_index = cfg.get('noise_index', num_classes - 1)
    class_names = cfg.get('class_names', [f'C{i}' for i in range(num_classes)])

    # 应用到模型（覆盖 __init__ 默认值）
    model.conf_threshold = conf_threshold
    model.margin_threshold = margin_threshold
    model.noise_index = noise_index

    print(f"  类别: {class_names}")
    print(f"  门限: conf={conf_threshold}, margin={margin_threshold}, noise_idx={noise_index}")

    print(f"\n连接 {args.port}...")
    receiver = MelReceiver(args.port)
    receiver.start()
    print("等待Mel数据...\n")

    mel_window = np.zeros((FRAME_WINDOW, MEL_FILTERS), dtype=np.float32)
    window_count = 0
    infer_count = 0
    last_print = 0

    try:
        while True:
            frames = receiver.get_frames()
            if not frames:
                time.sleep(0.01)
                continue

            for mel in frames:
                mel_window = np.roll(mel_window, -1, axis=0)
                mel_window[-1] = mel
                window_count += 1

            now = time.time()
            if now - last_print < 0.5:
                continue
            last_print = now

            x = torch.tensor(mel_window).unsqueeze(0)
            with torch.no_grad():
                # 用 predict()，与 MCU 端 softmax_with_reject 行为一致
                best_idx_t, probs = model.predict(x)
                probs = probs.numpy()[0]
                best_idx = int(best_idx_t.item())
                best_prob = probs[best_idx]

            infer_count += 1

            sys.stdout.write('\033[2J\033[H')
            print(f"=== 芯声 Mel迭代工具 ===  帧:{receiver.frame_count}  推理:{infer_count}")
            print(f"时间: {time.strftime('%H:%M:%S')}  窗口: {window_count}/{FRAME_WINDOW}")
            print()

            print_mel_heatmap(mel_window.T)
            print()

            print("概率分布:")
            for i, (name, p) in enumerate(zip(class_names, probs)):
                bar = '█' * int(p * 40)
                marker = ' ←' if i == best_idx else ''
                print(f"  {name:16s} {p*100:5.1f}% {bar}{marker}")

            print()
            # 与 MCU 端行为完全一致：拒识后 best_idx == noise_index
            if best_idx == noise_index:
                print(f">>> noise/不确定 (置信度: {best_prob*100:.1f}%, top1-top2: {(probs.max() - sorted(probs)[-2])*100:.1f}%)")
            else:
                print(f">>> 识别: {class_names[best_idx]} ({best_prob*100:.1f}%)")

            rms = np.sqrt(np.mean(mel_window[-1] ** 2))
            print(f"\nRMS: {rms:.1f}  AGC: {receiver.frame_count}")

    except KeyboardInterrupt:
        print("\n退出")
    finally:
        receiver.stop()

if __name__ == '__main__':
    main()
