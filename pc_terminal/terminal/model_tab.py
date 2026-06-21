"""
模型训练与管理界面模块
提供模型训练、Flash烧录、C头文件导出等功能
"""
import os
import sys
import re
import json
import time
import struct
import shutil
import subprocess
from glob import glob
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QLineEdit, QDialog, QDialogButtonBox, QFormLayout,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QProgressBar, QTextEdit, QSplitter, QAbstractItemView,
    QMessageBox, QListWidget, QListWidgetItem, QScrollArea,
    QApplication, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QMutex, QWaitCondition
from PySide6.QtGui import QFont, QColor

import pyqtgraph as pg

from terminal.serial_worker import CMD_SWITCH_MODEL, build_cmd_frame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
ALLDATA_DIR = os.path.join(BASE_DIR, "alldata")  # 数据集目录
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")  # 项目目录
CORE_INC_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "Core", "Inc"))  # STM32 Core头文件目录

def _find_stm32_cli():
    """查找STM32_Programmer_CLI可执行文件路径"""
    import shutil
    cli = shutil.which("STM32_Programmer_CLI")
    if cli:
        return cli
    candidates = [
        r"E:\QRS\stprogrammer\bin\STM32_Programmer_CLI.exe",
        r"C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe",
        r"C:\Program Files (x86)\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

STM32_CLI = _find_stm32_cli()

COMPILED_WEIGHTS_H = os.path.join(CORE_INC_DIR, "model_weights.h")  # 编译后的权重头文件路径

MEL_FILTERS = 20  # Mel滤波器数量
FRAME_WINDOW = 32  # 帧窗口大小
BASE_COMMANDS = ["forward", "backward", "turn", "start", "stop"]  # 基础指令集
BASE_CLASSES = ['backward', 'forward', 'start', 'stop', 'turn', 'noise']  # 基础类别


def get_latest_model_dir(project_dir):
    """获取项目中最新的模型目录（如 model1, model2...）"""
    max_num = 0
    for name in os.listdir(project_dir) if os.path.isdir(project_dir) else []:
        if name.startswith("model"):
            try:
                num = int(name[5:])
                if num > max_num:
                    max_num = num
            except ValueError:
                pass
    return f"model{max_num}" if max_num > 0 else "model1"


def parse_folder_name(name):
    """解析文件夹名称，提取显示名称和命令（如 '前进(forward)' -> ('前进', 'forward')）"""
    m = re.match(r'^(.+)\((.+)\)$', name)
    if m:
        display_name = m.group(1)
        command = m.group(2).strip()
        if command in BASE_COMMANDS:
            return display_name, command
    return name, name


def detect_all_classes(dataset_dir=ALLDATA_DIR):
    """检测数据集目录中的所有类别及其样本数量"""
    classes = []
    if not os.path.isdir(dataset_dir):
        return classes
    for name in sorted(os.listdir(dataset_dir)):
        cls_dir = os.path.join(dataset_dir, name)
        if not os.path.isdir(cls_dir):
            continue
        npy_files = glob(os.path.join(cls_dir, "*.npy"))
        if len(npy_files) == 0:
            continue
        display, command = parse_folder_name(name)
        classes.append({
            "name": name,
            "display": display,
            "command": command,
            "count": len(npy_files),
        })
    non_noise = [c for c in classes if c["name"] != "noise" and c["command"] != "noise"]
    noise = [c for c in classes if c["name"] == "noise" or c["command"] == "noise"]
    return non_noise + noise


def read_compiled_model_info():
    """从编译后的C头文件中读取模型信息（类别数、名称等）"""
    info = {"class_count": 0, "class_names": [], "class_display": [], "class_to_cmd": []}
    if not os.path.isfile(COMPILED_WEIGHTS_H):
        return info
    try:
        with open(COMPILED_WEIGHTS_H, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        m = re.search(r'#define\s+NUM_CLASSES\s+(\d+)', content)
        if m:
            info["class_count"] = int(m.group(1))
        m = re.search(r'static\s+const\s+char\s*\*\s*class_labels\[NUM_CLASSES\]\s*=\s*\{([^}]+)\}', content)
        if m:
            names = re.findall(r'"([^"]*)"', m.group(1))
            info["class_names"] = names
        m = re.search(r'static\s+const\s+char\s*\*\s*class_display\[NUM_CLASSES\]\s*=\s*\{([^}]+)\}', content)
        if m:
            names = re.findall(r'"([^"]*)"', m.group(1))
            info["class_display"] = names
        m = re.search(r'static\s+const\s+int\s+class_to_cmd\[NUM_CLASSES\]\s*=\s*\{([^}]+)\}', content)
        if m:
            nums = re.findall(r'\d+', m.group(1))
            info["class_to_cmd"] = [int(n) for n in nums]
    except Exception:
        pass
    return info


def read_flash_model_info():
    """从Flash中读取模型信息（当前为空实现）"""
    return {"class_count": 0, "class_names": [], "class_display": [], "class_commands": [], "class_to_cmd": [], "version": 0}


def load_project_meta(project_dir):
    """加载项目元数据（JSON配置文件）"""
    meta_path = os.path.join(project_dir, "project_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_project_meta(project_dir, meta):
    """保存项目元数据到JSON文件"""
    meta_path = os.path.join(project_dir, "project_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def spec_augment(mel):
    """SpecAugment数据增强：随机遮蔽时间和频率维度"""
    mel = mel.copy()
    T = mel.shape[0]
    t0 = np.random.randint(0, T - 1)
    tw = np.random.randint(1, min(4, T - t0))
    mel[t0:t0+tw, :] = 0.0
    if np.random.random() > 0.7:
        f_bins = mel.shape[1]
        f0 = np.random.randint(0, f_bins - 2)
        fw = np.random.randint(2, min(5, f_bins - f0))
        mel[:, f0:f0+fw] = np.random.normal(0, 0.1, (T, fw)).astype(np.float32)
    return mel


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力块（参数开销 <5%，判别力提升明显）"""
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
    """深度可分离卷积神经网络（DS-CNN）+ SE 通道注意力，用于音频分类任务。

    新增推理门限 `predict()`：当 top1 置信度低于 conf_threshold，或 top1-top2
    差小于 margin_threshold 时，判定为 noise（拒识），用于降低误识别率。
    """
    def __init__(self, num_classes, mel_filters=20, conv1_out=32, conv2_out=48, conv3_out=64,
                 conf_threshold=0.7, margin_threshold=0.2, noise_index=None):
        super().__init__()
        self.conv1 = nn.Conv1d(mel_filters, conv1_out, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        self.depth1 = nn.Conv1d(conv1_out, conv1_out, 3, padding=1, groups=conv1_out)
        self.bn_d1 = nn.BatchNorm1d(conv1_out)
        self.point1 = nn.Conv1d(conv1_out, conv2_out, 1)
        self.bn_p1 = nn.BatchNorm1d(conv2_out)
        self.se1 = SEBlock(conv2_out)                # SE 注意力 1（接 point1）
        self.depth2 = nn.Conv1d(conv2_out, conv2_out, 3, padding=1, groups=conv2_out)
        self.bn_d2 = nn.BatchNorm1d(conv2_out)
        self.point2 = nn.Conv1d(conv2_out, conv3_out, 1)
        self.bn_p2 = nn.BatchNorm1d(conv3_out)
        self.se2 = SEBlock(conv3_out)                # SE 注意力 2（接 point2）
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(conv3_out, num_classes)
        self.pool = nn.MaxPool1d(2)
        # 推理门限（也会写入 cfg 同步到 MCU 端）
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
        """带拒识的推理：低置信度或类别边界模糊都判定为 noise。

        返回:
            idx:   (B,) 预测类别索引（不确定的样本归为 noise_index）
            probs: (B, C) softmax 概率
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(probs, 2, dim=-1)
        p1, p2 = top2.values[:, 0], top2.values[:, 1]
        idx = top2.indices[:, 0]
        uncertain = (p1 < self.conf_threshold) | ((p1 - p2) < self.margin_threshold)
        noise_t = torch.tensor(self.noise_index, device=idx.device, dtype=idx.dtype)
        idx = torch.where(uncertain, noise_t, idx)
        return idx, probs


class TrainingWorker(QThread):
    """训练工作线程，支持暂停/取消，发射进度和准确率信号"""
    progress = Signal(int, int, str)
    accuracy = Signal(float, float)
    loss_signal = Signal(float)
    log = Signal(str)
    finished_signal = Signal(bool, str)
    curve_data = Signal(float, float, float)

    def __init__(self, project_dir, class_names, params, model_name="model1", dataset_dir=ALLDATA_DIR):
        super().__init__()
        self.project_dir = project_dir
        self.class_names = class_names
        self.params = params
        self.model_name = model_name
        self.dataset_dir = dataset_dir
        self._paused = False
        self._cancelled = False
        self._mutex = QMutex()
        self._pause_cond = QWaitCondition()

    def pause(self):
        """暂停训练"""
        self._mutex.lock()
        self._paused = True
        self._mutex.unlock()

    def resume(self):
        """恢复训练"""
        self._mutex.lock()
        self._paused = False
        self._pause_cond.wakeAll()
        self._mutex.unlock()

    def cancel(self):
        """取消训练"""
        self._mutex.lock()
        self._cancelled = True
        self._paused = False
        self._pause_cond.wakeAll()
        self._mutex.unlock()

    def _check_pause(self):
        """检查是否需要暂停，返回是否应取消"""
        self._mutex.lock()
        while self._paused and not self._cancelled:
            self._pause_cond.wait(self._mutex, 100)
        should_cancel = self._cancelled
        self._mutex.unlock()
        return should_cancel

    def run(self):
        """线程主入口"""
        try:
            self._do_train()
        except Exception as e:
            self.log.emit(f"[错误] {str(e)}")
            self.finished_signal.emit(False, str(e))

    def _do_train(self):
        """执行训练流程：加载数据、训练模型、保存权重和评估结果"""
        epochs = self.params.get("epochs", 250)
        lr = self.params.get("lr", 0.002)
        batch_size = self.params.get("batch_size", 64)
        optimizer_name = self.params.get("optimizer", "adamw")
        mixup_alpha = float(self.params.get("mixup_alpha", 0.3))
        class_names = list(self.class_names)

        self.log.emit(f"开始训练: {len(class_names)} 类, {epochs} epochs, lr={lr}, bs={batch_size}, opt={optimizer_name}, mixup={mixup_alpha}")
        self.log.emit(f"类别: {', '.join(class_names)}")

        self.log.emit("加载数据集...")
        self.progress.emit(0, epochs, "加载数据集")
        X, y, sample_count = self._load_data(class_names)
        if len(X) == 0:
            self.log.emit("[错误] 无有效训练数据")
            self.finished_signal.emit(False, "无有效训练数据")
            return
        self.log.emit(f"总样本: {len(X)}")

        idx = np.arange(len(X))
        np.random.shuffle(idx)
        split = int(len(idx) * 0.8)
        train_idx, val_idx = idx[:split], idx[split:]

        tX = torch.tensor(X[train_idx])
        ty = torch.tensor(y[train_idx])
        vX = torch.tensor(X[val_idx])
        vy = torch.tensor(y[val_idx])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log.emit(f"设备: {device}")

        warmup_epochs = 5

        seed = int(time.time() * 1000) % 100000
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        g = torch.Generator()
        g.manual_seed(seed + 999)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(tX, ty),
            batch_size=batch_size, shuffle=True, generator=g)
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(vX, vy),
            batch_size=batch_size, shuffle=True, generator=g)

        # 找 noise 类索引（用于推理拒识）
        noise_idx = None
        for i, cn in enumerate(class_names):
            if cn == "noise":
                noise_idx = i
                break
        if noise_idx is None and len(class_names) > 0:
            noise_idx = len(class_names) - 1

        # 类别权重：用 1/sqrt(count) 平衡样本不均衡
        from collections import Counter
        cnt = Counter(y.tolist())
        cw = np.array([1.0 / np.sqrt(max(cnt.get(i, 1), 1))
                       for i in range(len(class_names))], dtype=np.float32)
        cw = cw / cw.sum() * len(class_names)        # 归一化后均值≈1
        class_weights = torch.tensor(cw, dtype=torch.float32)
        self.log.emit(f"类别权重: {[f'{c:.2f}' for c in cw]}")

        # 推理门限（写到 cfg / C 头文件）
        conf_threshold = float(self.params.get("conf_threshold", 0.5))
        margin_threshold = float(self.params.get("margin_threshold", 0.1))

        model = DSCNN(
            num_classes=len(class_names),
            conf_threshold=conf_threshold,
            margin_threshold=margin_threshold,
            noise_index=noise_idx,
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        self.log.emit(f"参数量: {total_params:,} ({total_params * 4 / 1024:.1f} KB)")
        self.log.emit(f"SE注意力: 启用 (r=4), 推理门限: conf={conf_threshold}, margin={margin_threshold}")

        # 优化器选择：AdamW 准确率比 SGD 高 2-3%
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=1e-5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_epochs])
        # Label Smoothing + 类别权重：防过自信、平衡不均衡
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)

        # EMA（指数移动平均）：泛化更好，用于最终保存与评估
        ema_decay = 0.999
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        def _ema_update():
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    sv = ema_state[k]
                    if sv.dtype in (torch.float32, torch.float64):
                        sv.mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
                    else:
                        sv.copy_(v.detach())

        best_acc = 0.0
        best_state = None
        best_ema_acc = 0.0
        best_ema_state = None

        for epoch in range(1, epochs + 1):
            if self._check_pause():
                self.log.emit("训练已取消")
                self.finished_signal.emit(False, "已取消")
                return

            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                if self._cancelled:
                    self.log.emit("训练已取消")
                    self.finished_signal.emit(False, "已取消")
                    return
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()

                # Mixup：50% 概率触发，提升泛化
                if mixup_alpha > 0 and np.random.random() < 0.5:
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    perm = torch.randperm(xb.size(0), device=device)
                    xb_mixed = lam * xb + (1.0 - lam) * xb[perm]
                    logits = model(xb_mixed)
                    loss = lam * loss_fn(logits, yb) + (1.0 - lam) * loss_fn(logits, yb[perm])
                else:
                    loss = loss_fn(model(xb), yb)

                loss.backward()
                optimizer.step()
                _ema_update()
                train_loss += loss.item()
            scheduler.step()

            avg_loss = train_loss / len(train_loader)
            self.loss_signal.emit(avg_loss)

            # 评估使用 EMA 权重（泛化更好）
            model.load_state_dict(ema_state)
            model.eval()
            correct, total_v = 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    _, pred = torch.max(model(xb), 1)
                    total_v += yb.size(0)
                    correct += (pred == yb).sum().item()
            val_acc = correct / total_v if total_v > 0 else 0.0

            self.progress.emit(epoch, epochs, f"Epoch {epoch}/{epochs}")
            self.accuracy.emit(val_acc, best_acc)
            self.curve_data.emit(float(epoch), val_acc, avg_loss)

            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                self.log.emit(
                    f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
                    f"val_acc(EMA)={val_acc*100:.1f}%  best={best_acc*100:.1f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in ema_state.items()}

        if best_state is None:
            best_state = model.state_dict()

        weights_dir = os.path.join(self.project_dir, self.model_name, "weights")
        reports_dir = os.path.join(self.project_dir, self.model_name, "evaluation")
        os.makedirs(weights_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        pth_path = os.path.join(weights_dir, "model_weights.pth")
        torch.save(best_state, pth_path)
        self.log.emit(f"权重已保存: {pth_path}")

        model.load_state_dict(best_state)
        model.eval()
        all_preds, all_labels, all_top1 = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                idx, probs = model.predict(xb)        # 带拒识的预测
                _, top1 = torch.max(probs, dim=-1)
                all_preds.extend(idx.cpu().numpy())
                all_top1.extend(top1.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
        nc = len(class_names)
        cm = [[0]*nc for _ in range(nc)]
        for t, p in zip(all_labels, all_preds):
            cm[t][p] += 1
        per_class_acc = []
        for i in range(nc):
            row_sum = sum(cm[i])
            per_class_acc.append(cm[i][i] / row_sum if row_sum > 0 else 0.0)

        # 拒识相关指标（针对降低误识别）
        rejected_mask = [p == noise_idx for p in all_preds]
        rejected_count = sum(rejected_mask)
        n_total = len(all_labels)
        if noise_idx is not None:
            # 真实是 noise 且被拒识的占比
            true_noise = sum(1 for l in all_labels if l == noise_idx)
            true_non_noise = n_total - true_noise
            false_reject = sum(1 for l, p, r in zip(all_labels, all_preds, rejected_mask)
                               if r and l != noise_idx)        # 把真类判成 noise
            false_accept = sum(1 for l, p, r in zip(all_labels, all_preds, rejected_mask)
                               if (not r) and l == noise_idx)  # 噪声被判成指令
            accept_count = n_total - rejected_count
            true_accept = sum(1 for l, p, r in zip(all_labels, all_preds, rejected_mask)
                              if (not r) and l != noise_idx)
            precision_non_noise = true_accept / accept_count if accept_count > 0 else 0.0
        else:
            false_reject = false_accept = precision_non_noise = 0
            true_noise = 0
        self.log.emit(f"拒识率: {rejected_count}/{n_total} ({rejected_count*100/n_total:.1f}%)")
        self.log.emit(f"非噪声精度(剔除拒识后): {precision_non_noise*100:.1f}%")

        cfg = {
            "dataset_dir": self.dataset_dir,
            "num_classes": len(class_names),
            "class_names": list(class_names),
            "class_display": [""] * len(class_names),
            "class_commands": [""] * len(class_names),
            "class_to_cmd": [0] * len(class_names),
            "mel_filters": MEL_FILTERS,
            "frame_window": FRAME_WINDOW,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "conv1_out": 32,
            "conv2_out": 48,
            "conv3_out": 64,
            # 新增：SE + 门限
            "has_se": True,
            "se_reduction": 4,
            "conf_threshold": conf_threshold,
            "margin_threshold": margin_threshold,
            "noise_index": noise_idx if noise_idx is not None else -1,
            "label_smoothing": 0.1,
            "mixup_alpha": mixup_alpha,
            "optimizer": optimizer_name,
            "ema_decay": ema_decay,
        }
        all_data = np.concatenate([X[i] if isinstance(X[i], np.ndarray) else X[i].numpy() for i in range(len(X))], axis=0)
        cfg["norm_mean"] = float(all_data.mean())
        cfg["norm_std"] = float(all_data.std())
        for i, cn in enumerate(class_names):
            display, cmd = parse_folder_name(cn)
            cmd_map = {"forward": 1, "backward": 2, "turn": 3, "start": 4, "stop": 5}
            cfg["class_to_cmd"][i] = cmd_map.get(cmd, 0)
            cfg["class_commands"][i] = cmd
            cfg["class_display"][i] = display

        cfg_path = os.path.join(reports_dir, "model_config.json")
        cfg["confusion_matrix"] = cm
        cfg["per_class_acc"] = per_class_acc
        cfg["best_accuracy"] = float(best_acc)
        cfg["precision_non_noise"] = float(precision_non_noise)
        cfg["rejection_rate"] = float(rejected_count / n_total if n_total > 0 else 0.0)
        cfg["false_reject_count"] = int(false_reject)
        cfg["false_accept_count"] = int(false_accept)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        self.log.emit(f"准确率: {best_acc*100:.1f}%")
        self.log.emit("训练完成!")
        self.finished_signal.emit(True, f"{best_acc*100:.1f}")

    def _load_data(self, class_names):
        """加载训练数据，返回特征X、标签y和各类别样本数"""
        X, y = [], []
        counts = {}
        for label, cls_name in enumerate(class_names):
            pattern = os.path.join(self.dataset_dir, cls_name, "*.npy")
            files = sorted(glob(pattern))
            count = 0
            for fpath in files:
                try:
                    data = np.load(fpath)
                except Exception:
                    continue
                if data.ndim != 2 or data.shape[0] < FRAME_WINDOW:
                    continue
                for i in range(0, data.shape[0] - FRAME_WINDOW + 1, FRAME_WINDOW // 2):
                    seg = data[i:i+FRAME_WINDOW].astype(np.float32)
                    if seg.shape[0] != FRAME_WINDOW:
                        continue
                    mean = seg.mean(axis=1, keepdims=True)
                    std = seg.std(axis=1, keepdims=True) + 1e-6
                    seg = (seg - mean) / std
                    if cls_name != "noise" and np.random.random() < 0.3:
                        seg = spec_augment(seg)
                    X.append(seg)
                    y.append(label)
                    count += 1
                    if cls_name != "noise":
                        aug = seg.copy()
                        aug *= np.random.uniform(0.7, 1.3)
                        aug += np.random.normal(0, 0.03, aug.shape).astype(np.float32)
                        if np.random.random() < 0.5:
                            t_shift = np.random.randint(-2, 3)
                            if t_shift != 0:
                                aug = np.roll(aug, t_shift, axis=0)
                        if np.random.random() < 0.3:
                            aug = spec_augment(aug)
                        X.append(aug)
                        y.append(label)
                        count += 1
                if self._cancelled:
                    return np.array([]), np.array([]), {}
            counts[cls_name] = count
            self.log.emit(f"  {cls_name}: {len(files)} 文件 → {count} 样本")
        X = np.array(X, dtype=np.float32) if X else np.array([])
        y = np.array(y, dtype=np.int64) if y else np.array([])
        return X, y, counts


class SerialFlashWorker(QThread):
    """串口Flash烧录工作线程，通过串口协议将模型写入MCU Flash"""
    log = Signal(str)
    progress = Signal(int)
    finished_signal = Signal(bool, str)

    def __init__(self, serial_worker, bin_path, slot=1):
        super().__init__()
        self.sw = serial_worker
        self.bin_path = bin_path
        self.slot = slot

    def _crc32(self, d):
        """计算CRC32校验和"""
        c = 0xFFFFFFFF
        for b in d:
            c ^= b
            for _ in range(8):
                c = (c >> 1) ^ (0xEDB88320 & (-(c & 1)))
        return (~c) & 0xFFFFFFFF

    def run(self):
        """线程主入口"""
        try:
            self._do_flash()
        except Exception as e:
            self.log.emit(f"[错误] 串口烧录异常: {e}")
            self.finished_signal.emit(False, str(e))

    def _do_flash(self):
        """执行串口烧录流程：分包传输、CRC校验、状态查询"""
        from terminal.serial_worker import build_cmd_frame as _build
        import struct as _st

        with open(self.bin_path, 'rb') as f:
            data = f.read()

        slot = self.slot

        self.log.emit(f"=== 串口烧录开始 ===")
        self.log.emit(f"文件: {self.bin_path}")
        self.log.emit(f"大小: {len(data)} 字节 ({len(data)/1024:.2f} KB)")
        self.log.emit(f"目标: Flash slot {slot}")

        CMD_UPLOAD_START = 0x01
        CMD_UPLOAD_DATA = 0x02
        CMD_UPLOAD_FINISH = 0x03

        self.log.emit("[1/4] 查询MCU状态...")
        self.sw.query_status()
        self.msleep(500)

        self.log.emit("[2/4] 发送上传启动命令...")
        payload_start = _st.pack('<BI', slot, len(data))
        self.sw.send_command(CMD_UPLOAD_START, payload_start)
        self.log.emit(f"  slot={slot}, size={len(data)}")
        self.log.emit("  等待MCU擦除Flash (最多10秒)...")

        ack_ok = False
        ack_payload = None
        def _on_ack(ftype, val, payload):
            nonlocal ack_ok, ack_payload
            if ftype == 0x81:
                ack_payload = payload
                if val == 1:
                    ack_ok = True
                self.log.emit(f"  收到ACK: type=0x{ftype:02X} val={val} payload_len={len(payload)}")
        self.sw.ack_received.connect(_on_ack)
        deadline = time.time() + 10
        while time.time() < deadline and not ack_ok:
            self.msleep(100)
        self.sw.ack_received.disconnect(_on_ack)
        if not ack_ok:
            self.log.emit("[错误] MCU未确认上传启动!")
            if ack_payload:
                self.log.emit(f"  ACK payload: {list(ack_payload)}")
            else:
                self.log.emit("  未收到任何ACK响应")
            self.finished_signal.emit(False, "MCU未确认上传启动")
            return
        self.log.emit("  MCU确认OK，开始传输数据...")

        self.log.emit("[3/4] 传输数据...")
        chunk_size = 240
        offset = 0
        chunk_count = 0
        while offset < len(data):
            end = min(offset + chunk_size, len(data))
            chunk = data[offset:end]
            self.sw.send_command(CMD_UPLOAD_DATA, chunk)
            offset = end
            chunk_count += 1
            pct = offset * 100 // len(data)
            self.progress.emit(pct)
            if chunk_count % 10 == 0:
                self.log.emit(f"  已传输: {offset}/{len(data)} ({pct}%)")
            self.msleep(50)
        self.log.emit(f"  数据传输完成: {chunk_count} 个分包")

        self.log.emit("[4/4] 发送完成命令...")
        model_crc = self._crc32(data)
        finish_payload = _st.pack('<I', model_crc)
        self.sw.send_command(CMD_UPLOAD_FINISH, finish_payload)
        self.log.emit(f"  CRC32: 0x{model_crc:08X}")

        self.log.emit("  等待MCU验证 (最多10秒)...")
        self.msleep(2000)

        flash_state = -1
        def _on_status(source, nc, sys_state, cmd, flash_st, dist=0):
            nonlocal flash_state
            flash_state = flash_st
            self.log.emit(f"  状态: source={source} classes={nc} flash_state={flash_st}")
        self.sw.status_received.connect(_on_status)

        for attempt in range(8):
            self.sw.query_status()
            self.msleep(1000)
            if flash_state in (0, 2):
                break
        self.sw.status_received.disconnect(_on_status)

        STATE_NAMES = {0: "IDLE", 1: "UPDATING", 2: "VALID"}
        state_name = STATE_NAMES.get(flash_state, f"未知({flash_state})")
        self.progress.emit(100)
        self.log.emit("=== 串口烧录结束 ===")
        if flash_state == 2:
            self.log.emit(f"[成功] Flash烧录完成! 状态={state_name}")
            self.finished_signal.emit(True, "串口烧录完成")
        else:
            self.log.emit(f"[失败] Flash状态={state_name}，烧录未成功，请重试或改用ST-Link")
            self.finished_signal.emit(False, f"Flash状态={state_name}")


class StlinkFlashWorker(QThread):
    """ST-Link烧录工作线程，通过STM32_Programmer_CLI刷写Flash"""
    log = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, bin_path):
        super().__init__()
        self.bin_path = bin_path

    def run(self):
        """线程主入口"""
        try:
            self._do_flash()
        except Exception as e:
            self.log.emit(f"[错误] ST-Link烧录异常: {e}")
            self.finished_signal.emit(False, str(e))

    def _do_flash(self):
        """执行ST-Link烧录：调用STM32_Programmer_CLI刷写到指定地址"""
        MODEL_A_ADDR = "0x080E0000"
        if STM32_CLI is None:
            self.log.emit("[错误] 未找到 STM32_Programmer_CLI")
            self.log.emit("  请安装 STM32CubeProgrammer 或设置环境变量")
            self.finished_signal.emit(False, "未找到 STM32_Programmer_CLI")
            return
        self.log.emit(f"ST-Link 烧录: {self.bin_path}")
        self.log.emit(f"  文件大小: {os.path.getsize(self.bin_path)} bytes ({os.path.getsize(self.bin_path)/1024:.2f} KB)")

        import tempfile
        tmp_dir = tempfile.mkdtemp()
        tmp_bin = os.path.join(tmp_dir, "model_flash.bin")
        shutil.copy2(self.bin_path, tmp_bin)
        self.log.emit(f"  临时文件: {tmp_bin}")

        try:
            self.log.emit(f"调用 STM32_Programmer_CLI ({STM32_CLI})")
            self.log.emit(f"刷写到 {MODEL_A_ADDR}...")
            cmd = [
                STM32_CLI, "-c", "port=SWD",
                "-w", tmp_bin, MODEL_A_ADDR,
                "-v", "-rst"
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
                creationflags=subprocess.CREATE_NO_WINDOW,
                encoding='gbk', errors='replace')
            if result.returncode == 0:
                self.log.emit("ST-Link 刷写成功!")
                self.log.emit(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            else:
                self.log.emit(f"[错误] 刷写失败: {result.stderr[-300:]}")
                self.finished_signal.emit(False, f"刷写失败: {result.stderr[-500:]}")
                return
        except FileNotFoundError:
            self.log.emit("[错误] 未找到 STM32_Programmer_CLI，请确保已安装并添加到PATH")
            self.finished_signal.emit(False, "未找到 STM32_Programmer_CLI")
            return
        except subprocess.TimeoutExpired:
            self.log.emit("[错误] 刷写超时")
            self.finished_signal.emit(False, "刷写超时")
            return
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        self.finished_signal.emit(True, "ST-Link烧录完成")


class NewProjectDialog(QDialog):
    """新建训练项目对话框，用于配置项目名称、类别和训练参数"""
    def __init__(self, all_classes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新建训练项目")
        self.setMinimumWidth(550)
        self.setMinimumHeight(500)
        self.all_classes = all_classes
        self._build_ui()

    def _build_ui(self):
        """构建对话框界面"""
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("输入项目名称")
        form.addRow("项目名称:", self.name_edit)
        layout.addLayout(form)

        base_group = QGroupBox("基础指令类")
        base_lay = QVBoxLayout(base_group)
        base_label = QLabel("基础指令类: " + ", ".join(BASE_CLASSES) + " (自动包含)")
        base_label.setStyleSheet("color: #8b949e; padding: 4px;")
        base_lay.addWidget(base_label)
        layout.addWidget(base_group)

        custom_group = QGroupBox("自定义类 (按名称/命令模式匹配)")
        custom_lay = QVBoxLayout(custom_group)
        self.custom_checks = {}
        for cls in self.all_classes:
            display, cmd = parse_folder_name(cls["name"])
            if display not in BASE_CLASSES:
                cb = QCheckBox(
                    f"{cls['name']}  →  命令: {cls['command']}  ({cls['count']} 文件)")
                self.custom_checks[cls["name"]] = cb
                custom_lay.addWidget(cb)
        if not self.custom_checks:
            custom_lay.addWidget(QLabel("（无自定义类可用）"))
        layout.addWidget(custom_group)

        param_group = QGroupBox("训练参数")
        param_lay = QFormLayout(param_group)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 2000)
        self.epochs_spin.setValue(250)
        self.epochs_spin.wheelEvent = lambda e: None
        lbl_epochs = QLabel("Epochs:")
        lbl_epochs.setToolTip("训练总轮数。250 是经验甜区（峰值约在150步），不会过拟合。")
        param_lay.addRow(lbl_epochs, self.epochs_spin)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.002)
        self.lr_spin.wheelEvent = lambda e: None
        lbl_lr = QLabel("学习率:")
        lbl_lr.setToolTip("AdamW 推荐 0.002；SGD 推荐 0.001-0.01。")
        param_lay.addRow(lbl_lr, self.lr_spin)
        self.bs_spin = QSpinBox()
        self.bs_spin.setRange(1, 512)
        self.bs_spin.setValue(64)
        self.bs_spin.wheelEvent = lambda e: None
        lbl_bs = QLabel("Batch Size:")
        lbl_bs.setToolTip("每次训练送入模型的样本数。越大训练越稳定但越耗显存。推荐32~128。")
        param_lay.addRow(lbl_bs, self.bs_spin)
        # 优化器选择
        from PySide6.QtWidgets import QComboBox
        self.opt_combo = QComboBox()
        self.opt_combo.addItems(["AdamW (推荐)", "SGD"])
        self.opt_combo.wheelEvent = lambda e: None
        lbl_opt = QLabel("优化器:")
        lbl_opt.setToolTip("AdamW 在本数据集上比 SGD 准确率高约2-3%。")
        param_lay.addRow(lbl_opt, self.opt_combo)
        # Mixup 强度
        self.mixup_spin = QDoubleSpinBox()
        self.mixup_spin.setRange(0.0, 1.0)
        self.mixup_spin.setDecimals(2)
        self.mixup_spin.setSingleStep(0.05)
        self.mixup_spin.setValue(0.3)
        self.mixup_spin.wheelEvent = lambda e: None
        lbl_mixup = QLabel("Mixup Alpha:")
        lbl_mixup.setToolTip("Mixup 强度，0=关闭。0.3 是当前最佳值。")
        param_lay.addRow(lbl_mixup, self.mixup_spin)
        layout.addWidget(param_group)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_result(self):
        """获取用户配置的项目参数"""
        selected = []
        for name, cb in self.custom_checks.items():
            if cb.isChecked():
                selected.append(name)
        class_names = list(BASE_CLASSES)
        for s in selected:
            if s not in class_names:
                class_names.append(s)
        opt_text = self.opt_combo.currentText()
        optimizer = "adamw" if "AdamW" in opt_text else "sgd"
        return {
            "name": self.name_edit.text().strip(),
            "class_names": class_names,
            "epochs": self.epochs_spin.value(),
            "lr": self.lr_spin.value(),
            "batch_size": self.bs_spin.value(),
            "optimizer": optimizer,
            "mixup_alpha": self.mixup_spin.value(),
        }


class ModelTab(QWidget):
    """模型训练与管理主界面，包含项目管理、训练控制、模型烧录等功能"""
    project_selected = Signal(str)

    def __init__(self, serial_worker):
        super().__init__()
        self.sw = serial_worker
        self._worker = None
        self._current_project = None
        self._current_record = None
        self._epoch_data = []
        self._acc_data = []
        self._loss_data = []
        self._compiled_info = read_compiled_model_info()
        self._flash_info = read_flash_model_info()
        self._active_source = 0
        self._flash_method = "stlink"
        self._build_ui()
        self._refresh_project_list()
        self._refresh_model_info()

        self.sw.status_received.connect(self._on_status)
        self.sw.ack_received.connect(self._on_ack)
        self.sw.model_info_received.connect(self._on_model_info)

    def _build_ui(self):
        """构建主界面布局"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        left_layout.addWidget(self._build_model_overview())
        left_layout.addWidget(self._build_project_panel())
        left_layout.addWidget(self._build_record_panel())
        left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        scroll = QScrollArea()
        scroll.setWidget(left_panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMinimumWidth(200)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { background: #161b22; width: 8px; }"
            "QScrollBar::handle:vertical { background: #30363d; border-radius: 4px; min-height: 20px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }")

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        right_layout.addWidget(self._build_control_panel(), stretch=0)
        right_layout.addWidget(self._build_curve_panel(), stretch=3)
        right_layout.addWidget(self._build_log_panel(), stretch=2)

        right_scroll = QScrollArea()
        right_scroll.setWidget(right_panel)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setWidgetResizable(True)
        right_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { background: #161b22; width: 8px; }"
            "QScrollBar::handle:vertical { background: #30363d; border-radius: 4px; min-height: 20px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([450, 600])
        main_layout.addWidget(splitter)

    def _build_model_overview(self):
        """构建模型概览面板"""
        group = QGroupBox("模型概览")
        lay = QVBoxLayout(group)

        compiled_box = QGroupBox("基座模型 (编译固件)")
        compiled_lay = QFormLayout(compiled_box)
        self.lbl_compiled_classes = QLabel("--")
        compiled_lay.addRow("类别数:", self.lbl_compiled_classes)
        lay.addWidget(compiled_box)

        flash_box = QGroupBox("学生模型 (Flash)")
        flash_lay = QFormLayout(flash_box)
        self.lbl_flash_classes = QLabel("--")
        flash_lay.addRow("类别数:", self.lbl_flash_classes)
        lay.addWidget(flash_box)

        self.lbl_active = QLabel("当前活跃: 基座模型")
        self.lbl_active.setStyleSheet(
            "color: #58a6ff; font-weight: bold; font-size: 13px; padding: 4px;")
        lay.addWidget(self.lbl_active)

        btn_row = QHBoxLayout()
        self.btn_switch_base = QPushButton("切换到编译基座")
        self.btn_switch_flash = QPushButton("切换到Flash模型")
        self.btn_switch_base.clicked.connect(lambda: self._switch_model(0))
        self.btn_switch_flash.clicked.connect(lambda: self._switch_model(1))
        btn_row.addWidget(self.btn_switch_base)
        btn_row.addWidget(self.btn_switch_flash)
        lay.addLayout(btn_row)

        return group

    def _build_project_panel(self):
        """构建训练项目管理面板"""
        group = QGroupBox("训练项目")
        lay = QVBoxLayout(group)

        self.project_table = QTableWidget()
        self.project_table.setMinimumHeight(120)
        self.project_table.setColumnCount(4)
        self.project_table.setHorizontalHeaderLabels(["项目名称", "数据量", "训练次数", "部署状态"])
        self.project_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.project_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.project_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.project_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.project_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.project_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.project_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.project_table.currentCellChanged.connect(self._on_project_selected)
        lay.addWidget(self.project_table)

        btn_row = QHBoxLayout()
        self.btn_new_project = QPushButton("新建项目")
        self.btn_del_project = QPushButton("删除项目")
        self.btn_new_project.setMinimumWidth(80)
        self.btn_del_project.setMinimumWidth(80)
        self.btn_new_project.clicked.connect(self._new_project)
        self.btn_del_project.clicked.connect(self._delete_project)
        btn_row.addWidget(self.btn_new_project)
        btn_row.addWidget(self.btn_del_project)
        lay.addLayout(btn_row)

        param_group = QGroupBox("训练参数")
        param_lay = QFormLayout(param_group)
        self.param_epochs = QSpinBox()
        self.param_epochs.setRange(10, 2000)
        self.param_epochs.setValue(200)
        self.param_epochs.wheelEvent = lambda e: None
        lbl_epochs = QLabel("Epochs:")
        lbl_epochs.setToolTip("训练总轮数。越大模型越充分学习，但过大会过拟合。推荐100~300。")
        param_lay.addRow(lbl_epochs, self.param_epochs)
        self.param_lr = QDoubleSpinBox()
        self.param_lr.setRange(0.00001, 1.0)
        self.param_lr.setDecimals(5)
        self.param_lr.setSingleStep(0.0001)
        self.param_lr.setValue(0.01)
        self.param_lr.wheelEvent = lambda e: None
        lbl_lr = QLabel("学习率:")
        lbl_lr.setToolTip("每次参数更新的步长。太大会震荡不收敛，太小会训练很慢。推荐0.0005~0.002。")
        param_lay.addRow(lbl_lr, self.param_lr)
        self.param_bs = QSpinBox()
        self.param_bs.setRange(1, 512)
        self.param_bs.setValue(64)
        self.param_bs.wheelEvent = lambda e: None
        lbl_bs = QLabel("Batch Size:")
        lbl_bs.setToolTip("每次训练送入模型的样本数。越大训练越稳定但越耗显存。推荐32~128。")
        param_lay.addRow(lbl_bs, self.param_bs)
        self.btn_save_params = QPushButton("保存参数")
        self.btn_save_params.clicked.connect(self._save_params)
        param_lay.addRow(self.btn_save_params)
        self.param_group = param_group
        param_group.setVisible(False)
        lay.addWidget(param_group)

        return group

    def _build_record_panel(self):
        """构建训练记录面板"""
        group = QGroupBox("训练记录")
        lay = QVBoxLayout(group)

        self.record_table = QTableWidget()
        self.record_table.setMinimumHeight(120)
        self.record_table.setColumnCount(4)
        self.record_table.setHorizontalHeaderLabels(["时间", "准确率", "Epochs", "状态"])
        self.record_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.record_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.record_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.record_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.record_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.record_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.record_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.record_table.currentCellChanged.connect(self._on_record_selected)
        lay.addWidget(self.record_table)

        btn_row = QHBoxLayout()
        self.btn_flash_record = QPushButton("刷写到Flash")
        self.btn_del_record = QPushButton("删除记录")
        self.btn_export_h = QPushButton("导出C头文件")
        self.cmb_flash_method = QComboBox()
        self.cmb_flash_method.addItems(["ST-Link", "串口"])
        self.cmb_flash_method.setCurrentText("ST-Link")
        self.cmb_flash_method.currentTextChanged.connect(lambda t: setattr(self, '_flash_method', 'stlink' if t == 'ST-Link' else 'serial'))
        self.cmb_flash_method.setFixedWidth(80)
        self.btn_flash_record.setMinimumWidth(90)
        self.btn_del_record.setMinimumWidth(90)
        self.btn_export_h.setMinimumWidth(90)
        self.btn_flash_record.clicked.connect(self._flash_record)
        self.btn_del_record.clicked.connect(self._delete_record)
        self.btn_export_h.clicked.connect(self._export_header)
        btn_row.addWidget(self.btn_flash_record)
        btn_row.addWidget(self.cmb_flash_method)
        btn_row.addWidget(self.btn_del_record)
        btn_row.addWidget(self.btn_export_h)
        lay.addLayout(btn_row)

        return group

    def _build_control_panel(self):
        """构建训练控制面板（开始/暂停/取消按钮、进度条）"""
        group = QGroupBox("训练控制")
        lay = QVBoxLayout(group)

        ctrl_row = QHBoxLayout()
        self.btn_start = QPushButton("开始训练")
        self.btn_pause = QPushButton("暂停")
        self.btn_cancel = QPushButton("取消")
        self.btn_start.setMinimumWidth(80)
        self.btn_pause.setMinimumWidth(80)
        self.btn_cancel.setMinimumWidth(80)
        self.btn_start.clicked.connect(self._start_training)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_cancel.clicked.connect(self._cancel_training)
        self.btn_pause.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.btn_start.setStyleSheet(
            "QPushButton { background: #238636; color: white; font-weight: bold; }"
            "QPushButton:hover { background: #2ea043; }"
            "QPushButton:disabled { background: #21262d; color: #484f58; }")
        self.btn_cancel.setStyleSheet(
            "QPushButton { background: #da3633; color: white; }"
            "QPushButton:hover { background: #f85149; }"
            "QPushButton:disabled { background: #21262d; color: #484f58; }")
        ctrl_row.addWidget(self.btn_start)
        ctrl_row.addWidget(self.btn_pause)
        ctrl_row.addWidget(self.btn_cancel)
        lay.addLayout(ctrl_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v%")
        self.progress_bar.setStyleSheet(
            "QProgressBar { background: #161b22; border: 1px solid #30363d; border-radius: 4px; "
            "text-align: center; color: #c9d1d9; height: 22px; }"
            "QProgressBar::chunk { background: #58a6ff; border-radius: 3px; }")
        lay.addWidget(self.progress_bar)

        info_row = QHBoxLayout()
        self.lbl_cur_acc = QLabel("当前准确率: --")
        self.lbl_best_acc = QLabel("最佳准确率: --")
        self.lbl_cur_loss = QLabel("当前Loss: --")
        for lbl in [self.lbl_cur_acc, self.lbl_best_acc, self.lbl_cur_loss]:
            lbl.setStyleSheet("color: #c9d1d9; font-size: 12px;")
            info_row.addWidget(lbl)
        lay.addLayout(info_row)

        return group

    def _build_curve_panel(self):
        """构建实时训练曲线面板（准确率和Loss曲线）"""
        group = QGroupBox("实时训练曲线")
        lay = QVBoxLayout(group)

        self.acc_plot = pg.PlotWidget(title="准确率曲线")
        self.acc_plot.setLabel('bottom', 'Epoch')
        self.acc_plot.setLabel('left', '准确率')
        self.acc_plot.setYRange(0, 1)
        self.acc_plot.setBackground('#0d1117')
        self.acc_plot.showGrid(x=True, y=True, alpha=0.3)
        self.acc_plot.getViewBox().setMouseEnabled(x=False, y=False)
        self.acc_curve = self.acc_plot.plot(pen=pg.mkPen('#58a6ff', width=2), name='当前准确率')
        self.best_curve = self.acc_plot.plot(pen=pg.mkPen('#f0883e', width=1, style=Qt.DashLine), name='最佳准确率')
        lay.addWidget(self.acc_plot)

        self.loss_plot = pg.PlotWidget(title="损失曲线")
        self.loss_plot.setLabel('bottom', 'Epoch')
        self.loss_plot.setLabel('left', 'Loss')
        self.loss_plot.setBackground('#0d1117')
        self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
        self.loss_plot.getViewBox().setMouseEnabled(x=False, y=False)
        self.loss_curve = self.loss_plot.plot(pen=pg.mkPen('#f85149', width=2), name='Loss')
        lay.addWidget(self.loss_plot)

        return group

    def _build_log_panel(self):
        """构建训练日志面板"""
        group = QGroupBox("训练日志")
        lay = QVBoxLayout(group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet(
            "QTextEdit { background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; "
            "border-radius: 4px; padding: 4px; }")
        lay.addWidget(self.log_text)
        return group

    def _refresh_model_info(self):
        """刷新模型信息显示（编译基座和Flash模型）"""
        self._compiled_info = read_compiled_model_info()
        self._flash_info = read_flash_model_info()

        cc = self._compiled_info["class_count"]
        self.lbl_compiled_classes.setText(str(cc) if cc > 0 else "无")
        if cc > 0:
            names = ", ".join(self._compiled_info.get("class_names", []))
            self.lbl_compiled_classes.setToolTip(names)

        base_meta = os.path.join(PROJECTS_DIR, "base", "model1", "weights", "model_config.json")
        if os.path.isfile(base_meta):
            try:
                with open(base_meta, "r", encoding="utf-8") as f:
                    c = json.load(f)
                self.lbl_compiled_classes.setText(str(c.get("num_classes", "--")))
            except Exception:
                pass

        fc = self._flash_info["class_count"]
        self.lbl_flash_classes.setText(str(fc) if fc > 0 else "无")

    def _update_active_label(self):
        """更新活跃模型标签显示"""
        names = {0: "基座模型", 1: "Flash"}
        src = names.get(self._active_source, "未知")
        color = "#58a6ff" if self._active_source == 0 else "#3fb950"
        self.lbl_active.setText(f"当前活跃: {src}")
        self.lbl_active.setStyleSheet(
                f"color: {color}; font-weight: bold; font-size: 13px; padding: 4px;")

    def _on_status(self, src, nc, state, cmd, progress, dist=0):
        """处理MCU状态更新"""
        self._active_source = src
        self._update_active_label()

    def _on_model_info(self, src, nc, names, maps):
        """处理模型信息更新"""
        self._active_source = src
        self._update_active_label()
        if src == 0:
            self.lbl_compiled_classes.setText(str(nc))
            if names:
                self.lbl_compiled_classes.setToolTip(", ".join(names))
        elif src == 1:
            self.lbl_flash_classes.setText(str(nc))
            if names:
                self.lbl_flash_classes.setToolTip(", ".join(names))

    def _on_ack(self, ftype, code, payload):
        """处理ACK响应（当前为空实现）"""
        pass

    def _switch_model(self, slot):
        """切换活跃模型（0=编译基座, 1=Flash）"""
        if not self.sw.connected:
            QMessageBox.warning(self, "提示", "请先连接串口")
            return
        names = {0: "编译基座", 1: "Flash"}
        self.sw.send_command(CMD_SWITCH_MODEL, bytes([slot]))
        self._log(f"已发送模型切换命令: {names.get(slot, '?')}")
        QTimer.singleShot(500, self.sw.query_model_info)
        QTimer.singleShot(500, self.sw.query_status)

    def _refresh_project_list(self):
        """刷新项目列表表格"""
        os.makedirs(PROJECTS_DIR, exist_ok=True)
        self.project_table.setRowCount(0)
        self._projects = []
        for name in sorted(os.listdir(PROJECTS_DIR)):
            pdir = os.path.join(PROJECTS_DIR, name)
            if not os.path.isdir(pdir):
                continue
            meta = load_project_meta(pdir)
            if meta is None:
                meta = {
                    "name": name,
                    "class_names": [],
                    "params": {},
                    "train_count": 0,
                    "best_acc": "0%",
                    "deployed": False,
                    "records": [],
                }
                save_project_meta(pdir, meta)
            self._projects.append((pdir, meta))
            row = self.project_table.rowCount()
            self.project_table.insertRow(row)
            self.project_table.setItem(row, 0, QTableWidgetItem(meta.get("name", name)))
            data_count = 0
            proj_data = os.path.join(pdir, "data")
            for cn in meta.get("class_names", []):
                cls_dir = os.path.join(proj_data, cn)
                if os.path.isdir(cls_dir):
                    data_count += len(glob(os.path.join(cls_dir, "*.npy")))
            self.project_table.setItem(row, 1, QTableWidgetItem(str(data_count)))
            self.project_table.setItem(row, 2, QTableWidgetItem(str(meta.get("train_count", 0))))
            status = "已部署" if meta.get("deployed") else "未部署"
            acc = meta.get("best_acc", "")
            if acc:
                status += f" ({acc})"
            self.project_table.setItem(row, 3, QTableWidgetItem(status))

    def _on_project_selected(self, row, col, prev_row, prev_col):
        """处理项目选中事件，加载项目参数和训练记录"""
        if row < 0 or row >= len(self._projects):
            self.param_group.setVisible(False)
            return
        self._current_project = self._projects[row]
        self._refresh_record_list()
        pdir, meta = self._current_project
        self.project_selected.emit(meta.get("name", ""))
        params = meta.get("params", {})
        self.param_epochs.setValue(params.get("epochs", 200))
        self.param_lr.setValue(params.get("lr", 0.001))
        self.param_bs.setValue(params.get("batch_size", 64))
        self.param_group.setVisible(True)

    def _new_project(self):
        """新建训练项目"""
        all_classes = detect_all_classes()
        if not all_classes:
            QMessageBox.warning(self, "提示", "未检测到数据集，请先采集数据")
            return
        dlg = NewProjectDialog(all_classes, self)
        if dlg.exec() != QDialog.Accepted:
            return
        result = dlg.get_result()
        name = result["name"]
        if not name:
            QMessageBox.warning(self, "提示", "请输入项目名称")
            return
        if not result["class_names"]:
            QMessageBox.warning(self, "提示", "请至少选择一个类别")
            return
        pdir = os.path.join(PROJECTS_DIR, name)
        if os.path.isdir(pdir):
            QMessageBox.warning(self, "提示", f"项目 '{name}' 已存在")
            return
        os.makedirs(pdir, exist_ok=True)

        cn = result["class_names"]

        meta = {
            "name": name,
            "class_names": cn,
            "params": {
                "epochs": result["epochs"],
                "lr": result["lr"],
                "batch_size": result["batch_size"],
            },
            "train_count": 0,
            "best_acc": "0%",
            "deployed": False,
            "records": [],
        }
        save_project_meta(pdir, meta)

        data_dir = os.path.join(pdir, "data")
        os.makedirs(data_dir, exist_ok=True)

        total_files = 0
        for cls_name in cn:
            src = os.path.join(ALLDATA_DIR, cls_name)
            if os.path.isdir(src):
                total_files += len(glob(os.path.join(src, "*.npy")))

        from PySide6.QtWidgets import QProgressDialog
        from PySide6.QtCore import Qt as QtCore
        progress = QProgressDialog("正在复制数据集...", None, 0, total_files, self)
        progress.setWindowTitle("创建项目")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        copied = 0
        for cls_name in cn:
            src = os.path.join(ALLDATA_DIR, cls_name)
            dst = os.path.join(data_dir, cls_name)
            if not os.path.isdir(src):
                continue
            os.makedirs(dst, exist_ok=True)
            for fpath in glob(os.path.join(src, "*.npy")):
                shutil.copy2(fpath, os.path.join(dst, os.path.basename(fpath)))
                copied += 1
                progress.setValue(copied)
                if copied % 50 == 0:
                    QApplication.processEvents()
                if progress.wasCanceled():
                    break
            if progress.wasCanceled():
                break
        progress.close()
        self._log(f"已复制 {copied} 个数据文件到项目")
        self._log(f"已创建项目: {name}, 类别: {cn}")
        self._refresh_project_list()

    def _save_params(self):
        """保存当前项目的训练参数"""
        if self._current_project is None:
            return
        pdir, meta = self._current_project
        meta["params"] = {
            "epochs": self.param_epochs.value(),
            "lr": self.param_lr.value(),
            "batch_size": self.param_bs.value(),
        }
        save_project_meta(pdir, meta)
        self._log(f"参数已保存: epochs={self.param_epochs.value()}, lr={self.param_lr.value()}, bs={self.param_bs.value()}")

    def _delete_project(self):
        """删除选中的训练项目"""
        row = self.project_table.currentRow()
        if row < 0 or row >= len(self._projects):
            QMessageBox.warning(self, "提示", "请先选择项目")
            return
        pdir, meta = self._projects[row]
        ret = QMessageBox.question(
            self, "确认删除",
            f"确定删除项目 '{meta['name']}' 及其所有训练记录？",
            QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            shutil.rmtree(pdir, ignore_errors=True)
            self._current_project = None
            self._refresh_project_list()
            self.record_table.setRowCount(0)
            self._log(f"已删除项目: {meta['name']}")

    def _refresh_record_list(self):
        """刷新训练记录列表"""
        self.record_table.setRowCount(0)
        if self._current_project is None:
            return
        pdir, meta = self._current_project
        records = meta.get("records", [])
        for rec in reversed(records):
            row = self.record_table.rowCount()
            self.record_table.insertRow(row)
            self.record_table.setItem(row, 0, QTableWidgetItem(rec.get("time", "--")))
            self.record_table.setItem(row, 1, QTableWidgetItem(rec.get("accuracy", "--")))
            self.record_table.setItem(row, 2, QTableWidgetItem(str(rec.get("epochs", "--"))))
            status = rec.get("status", "已完成")
            self.record_table.setItem(row, 3, QTableWidgetItem(status))

    def _on_record_selected(self, row, col, prev_row, prev_col):
        """处理训练记录选中事件"""
        if self._current_project is None:
            return
        pdir, meta = self._current_project
        records = meta.get("records", [])
        idx = len(records) - 1 - row
        if 0 <= idx < len(records):
            self._current_record = records[idx]
        else:
            self._current_record = None

    def _start_training(self):
        """开始训练任务"""
        if self._current_project is None:
            QMessageBox.warning(self, "提示", "请先选择或创建训练项目")
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "提示", "已有训练任务在运行")
            return

        pdir, meta = self._current_project
        class_names = meta["class_names"]
        params = meta.get("params", {})

        model_num = 1
        while os.path.isdir(os.path.join(pdir, f"model{model_num}")):
            model_num += 1
        model_name = f"model{model_num}"

        self._epoch_data = []
        self._acc_data = []
        self._loss_data = []
        self.acc_curve.setData([], [])
        self.best_curve.setData([], [])
        self.loss_curve.setData([], [])
        self.progress_bar.setValue(0)

        self._worker = TrainingWorker(pdir, class_names, params, model_name=model_name,
                                      dataset_dir=os.path.join(pdir, "data"))
        self._current_model_name = model_name
        self._worker.progress.connect(self._on_progress)
        self._worker.accuracy.connect(self._on_accuracy)
        self._worker.loss_signal.connect(self._on_loss)
        self._worker.log.connect(self._log)
        self._worker.curve_data.connect(self._on_curve_data)
        self._worker.finished_signal.connect(self._on_training_finished)
        self._worker.start()

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.btn_pause.setText("暂停")

    def _toggle_pause(self):
        """切换训练暂停/继续状态"""
        if self._worker is None:
            return
        if self._worker._paused:
            self._worker.resume()
            self.btn_pause.setText("暂停")
            self._log("训练已恢复")
        else:
            self._worker.pause()
            self.btn_pause.setText("继续")
            self._log("训练已暂停")

    def _cancel_training(self):
        """取消当前训练任务"""
        if self._worker is None:
            return
        ret = QMessageBox.question(
            self, "确认取消",
            "确定要取消当前训练吗？",
            QMessageBox.Yes | QMessageBox.No)
        if ret != QMessageBox.Yes:
            return
        self._worker.cancel()
        self._log("正在取消训练...")

    def _on_progress(self, current, total, msg):
        """更新训练进度条"""
        pct = int(current * 100 / total) if total > 0 else 0
        self.progress_bar.setValue(pct)

    def _on_accuracy(self, val_acc, best_acc):
        """更新准确率显示"""
        self.lbl_cur_acc.setText(f"当前准确率: {val_acc*100:.1f}%")
        self.lbl_best_acc.setText(f"最佳准确率: {best_acc*100:.1f}%")

    def _on_loss(self, loss_val):
        """更新Loss显示"""
        self.lbl_cur_loss.setText(f"当前Loss: {loss_val:.4f}")

    def _on_curve_data(self, epoch, acc, loss_val):
        """更新训练曲线数据"""
        self._epoch_data.append(epoch)
        self._acc_data.append(acc)
        self._loss_data.append(loss_val)
        self.acc_curve.setData(self._epoch_data, self._acc_data)
        if self._acc_data:
            best_so_far = []
            mx = 0.0
            for a in self._acc_data:
                mx = max(mx, a)
                best_so_far.append(mx)
            self.best_curve.setData(self._epoch_data, best_so_far)
        self.loss_curve.setData(self._epoch_data, self._loss_data)

    def _on_training_finished(self, success, result_str):
        """处理训练完成事件，更新项目状态和记录"""
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.btn_pause.setText("暂停")

        if not self._current_project:
            return

        pdir, meta = self._current_project
        meta["train_count"] = meta.get("train_count", 0) + 1

        record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy": f"{result_str}%" if success else "失败",
            "epochs": meta.get("params", {}).get("epochs", 0),
            "status": "已完成" if success else "失败",
            "project_dir": pdir,
            "record_dir": os.path.join(pdir, self._current_model_name),
        }

        if success:
            try:
                acc_val = float(result_str)
                old_best = meta.get("best_acc", "0%")
                old_val = float(old_best.replace("%", "")) if old_best else 0.0
                if acc_val > old_val:
                    meta["best_acc"] = f"{acc_val}%"
            except ValueError:
                pass

        records = meta.get("records", [])
        records.append(record)
        meta["records"] = records
        save_project_meta(pdir, meta)

        self._refresh_project_list()
        self._refresh_record_list()
        self._log(f"训练{'完成' if success else '失败'}: {result_str}")

    def _delete_record(self):
        """删除选中的训练记录"""
        if self._current_project is None or self._current_record is None:
            QMessageBox.warning(self, "提示", "请先选择训练记录")
            return
        pdir, meta = self._current_project
        rec = self._current_record
        ret = QMessageBox.question(
            self, "确认删除",
            f"确定删除训练记录 '{rec.get('time', '')}' (准确率: {rec.get('accuracy', '')})?",
            QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            rec_dir = rec.get("record_dir", "")
            if rec_dir and os.path.isdir(rec_dir):
                shutil.rmtree(rec_dir, ignore_errors=True)
            records = meta.get("records", [])
            records = [r for r in records if r.get("time") != rec.get("time")]
            meta["records"] = records
            save_project_meta(pdir, meta)
            self._current_record = None
            self._refresh_record_list()
            self._refresh_project_list()
            self._log(f"已删除训练记录")

    def _flash_record(self):
        """将选中的训练记录刷写到Flash"""
        if self._current_record is None:
            QMessageBox.warning(self, "提示", "请先选择训练记录")
            return
        rec = self._current_record
        record_dir = rec.get("record_dir", "")
        pth_path = os.path.join(record_dir, "weights", "model_weights.pth")
        cfg_path = os.path.join(record_dir, "evaluation", "model_config.json")
        if not os.path.isfile(pth_path):
            pdir = rec.get("project_dir", "")
            model_dir = get_latest_model_dir(pdir)
            pth_path = os.path.join(pdir, model_dir, "weights", "model_weights.pth")
            cfg_path = os.path.join(pdir, model_dir, "evaluation", "model_config.json")
        if not os.path.isfile(pth_path):
            msg_box = QMessageBox(QMessageBox.Warning, "提示", "找不到模型权重文件", QMessageBox.Ok, self)
            msg_box.setToolTip(pth_path)
            msg_box.exec()
            return
        if not os.path.isfile(cfg_path):
            msg_box = QMessageBox(QMessageBox.Warning, "提示", "找不到模型配置文件", QMessageBox.Ok, self)
            msg_box.setToolTip(cfg_path)
            msg_box.exec()
            return

        self._log("正在生成 Flash 二进制...")
        self._log(f"  权重文件: {pth_path} ({os.path.getsize(pth_path)} bytes)")
        self._log(f"  配置文件: {cfg_path}")
        QApplication.processEvents()

        try:
            flash_bin = os.path.join(os.path.dirname(pth_path), "model_flash.bin")
            self._export_flash_bin(pth_path, cfg_path, flash_bin)
            self._log(f"Flash 二进制已生成: {flash_bin}")
        except Exception as e:
            self._log(f"[错误] 生成Flash二进制失败: {e}")
            msg_box = QMessageBox(QMessageBox.Critical, "错误", f"生成Flash二进制失败: {e}", QMessageBox.Ok, self)
            msg_box.setToolTip(str(e))
            msg_box.exec()
            return

        compiled_info = read_compiled_model_info()
        self._log(f"编译基座: {compiled_info.get('class_count', 0)} 类")
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            new_nc = cfg.get("num_classes", 0)
            compiled_nc = compiled_info.get("class_count", 0)
            if new_nc != compiled_nc:
                self._log(f"[警告] 类别数不匹配: 编译={compiled_nc}, 训练={new_nc}")
        except Exception:
            pass

        self._log(f"烧录方式: {'ST-Link' if self._flash_method == 'stlink' else '串口'}")
        self._log(f"串口连接状态: {'已连接' if self.sw.connected else '未连接'}")
        self._flash_bin_path = flash_bin
        if self._flash_method == 'stlink':
            self._stlink_flash_worker = StlinkFlashWorker(flash_bin)
            self._stlink_flash_worker.log.connect(self._log)
            self._stlink_flash_worker.finished_signal.connect(self._on_stlink_flash_finished)
            self._stlink_flash_worker.start()
        else:
            if self.sw.connected:
                self._serial_flash_worker = SerialFlashWorker(self.sw, flash_bin, slot=1)
                self._serial_flash_worker.log.connect(self._log)
                self._serial_flash_worker.progress.connect(self.progress_bar.setValue)
                self._serial_flash_worker.finished_signal.connect(self._on_serial_flash_finished)
                self._serial_flash_worker.start()
            else:
                QMessageBox.warning(self, "提示", "串口未连接，无法使用串口烧录，请切换到ST-Link模式或连接串口")

    def _export_flash_bin(self, pth_path, cfg_path, output_path):
        """生成Flash二进制文件（包含权重、类别信息和CRC校验）"""
        state = torch.load(pth_path, map_location='cpu')
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        num_classes = cfg['num_classes']
        class_names = cfg['class_names']
        class_to_cmd = cfg.get('class_to_cmd', [0] * num_classes)

        weight_order = [
            'conv1.weight', 'conv1.bias',
            'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var',
            'depth1.weight', 'depth1.bias',
            'bn_d1.weight', 'bn_d1.bias', 'bn_d1.running_mean', 'bn_d1.running_var',
            'point1.weight', 'point1.bias',
            'bn_p1.weight', 'bn_p1.bias', 'bn_p1.running_mean', 'bn_p1.running_var',
            'se1.fc1.weight', 'se1.fc1.bias', 'se1.fc2.weight', 'se1.fc2.bias',
            'depth2.weight', 'depth2.bias',
            'bn_d2.weight', 'bn_d2.bias', 'bn_d2.running_mean', 'bn_d2.running_var',
            'point2.weight', 'point2.bias',
            'bn_p2.weight', 'bn_p2.bias', 'bn_p2.running_mean', 'bn_p2.running_var',
            'se2.fc1.weight', 'se2.fc1.bias', 'se2.fc2.weight', 'se2.fc2.bias',
        ]

        base_bytes = b''
        for key in weight_order:
            base_bytes += state[key].numpy().astype(np.float32).tobytes()

        fc_w = state['fc.weight'].numpy().astype(np.float32)
        fc_b = state['fc.bias'].numpy().astype(np.float32)

        base_size = len(base_bytes)
        fc_w_offset = 40 + base_size
        fc_w_size = fc_w.nbytes
        fc_b_offset = fc_w_offset + fc_w_size
        fc_b_size = fc_b.nbytes
        meta_offset = fc_b_offset + fc_b_size

        CLASS_NAME_LEN = 24
        MAX_CLASSES = 10
        # meta 布局（与 C 端 ModelMeta_t 对齐）：
        #   class_names  240  class_display 240  class_map 10  thresholds 40
        #   conf_thr 4  margin_thr 4  noise_index 1  has_se 1  se_red 1  se1_h 1  se2_h 1
        #   reserved 3  meta_crc 4
        meta = bytearray(550)
        pos = 0
        for i in range(MAX_CLASSES):
            if i < len(class_names):
                name = class_names[i]
            else:
                name = ''
            name_bytes = name[:CLASS_NAME_LEN-1].encode('utf-8').ljust(CLASS_NAME_LEN, b'\x00')
            meta[pos:pos+CLASS_NAME_LEN] = name_bytes
            pos += CLASS_NAME_LEN
        for i in range(MAX_CLASSES):
            if i < len(class_names):
                display, cmd = parse_folder_name(class_names[i])
            else:
                display = ''
            display_bytes = display[:CLASS_NAME_LEN-1].encode('utf-8').ljust(CLASS_NAME_LEN, b'\x00')
            meta[pos:pos+CLASS_NAME_LEN] = display_bytes
            pos += CLASS_NAME_LEN
        for i in range(MAX_CLASSES):
            if i < len(class_to_cmd):
                meta[pos] = class_to_cmd[i] & 0xFF
            pos += 1
        # ===== 每类独立阈值（per-class 拒识门限）=====
        # 同一类样本的 softmax 概率分布不一样：
        #   - turn / noise 学得稳，概率高 → 阈值 0.60
        #   - start  是关键命令，易混淆 → 阈值 0.50
        #   - backward/forward/stop 概率分布偏低 → 阈值 0.40~0.45
        # key 用 parse_folder_name 解析出的 display 名（小写）
        # 找不到的类默认 0.45
        PER_CLASS_THRESHOLDS = {
            "backward": 0.45,
            "forward":  0.45,
            "start":    0.50,   # 关键命令：START 触发要快但要准
            "stop":     0.40,
            "turn":     0.60,   # 特征明显，概率高
            "noise":    0.50,
            "pai1":     0.50,   # 别名
            "left":     0.55,
            "right":    0.55,
            "horn":     0.55,
        }
        DEFAULT_THRESH = 0.45
        for i in range(MAX_CLASSES):
            if i < len(class_names):
                disp, _cmd = parse_folder_name(class_names[i])
                disp_key = disp.strip().lower()
                thr = PER_CLASS_THRESHOLDS.get(disp_key, DEFAULT_THRESH)
            else:
                thr = DEFAULT_THRESH
            struct.pack_into('<f', meta, pos, thr)
            pos += 4
        # ===== 新增：SE + 推理门限配置（与 C 端 ModelMeta_t 对齐） =====
        conf_thr = float(cfg.get('conf_threshold', 0.5))
        margin_thr = float(cfg.get('margin_threshold', 0.1))
        noise_idx = int(cfg.get('noise_index', -1))
        has_se = 1 if bool(cfg.get('has_se', True)) else 0
        se_reduction = int(cfg.get('se_reduction', 4))
        se1_h = int(cfg.get('se1_hidden', 48 // se_reduction if has_se else 0))
        se2_h = int(cfg.get('se2_hidden', 64 // se_reduction if has_se else 0))
        struct.pack_into('<f', meta, pos, conf_thr);        pos += 4
        struct.pack_into('<f', meta, pos, margin_thr);      pos += 4
        meta[pos] = noise_idx & 0xFF;                       pos += 1
        meta[pos] = has_se & 0xFF;                          pos += 1
        meta[pos] = se_reduction & 0xFF;                    pos += 1
        meta[pos] = se1_h & 0xFF;                           pos += 1
        meta[pos] = se2_h & 0xFF;                           pos += 1
        pos += 3                                            # reserved

        def _crc32(data):
            c = 0xFFFFFFFF
            for b in data:
                c ^= b
                for _ in range(8):
                    c = (c >> 1) ^ (0xEDB88320 & (-(c & 1)))
            return (~c) & 0xFFFFFFFF

        # CRC 覆盖前 546 字节（前 550 字节除去末尾 4 字节 meta_crc）
        meta_crc = _crc32(bytes(meta[:546]))
        struct.pack_into('<I', meta, 546, meta_crc)

        header = bytearray(40)
        struct.pack_into('<I', header, 0, 0xDEADBEEF)
        struct.pack_into('<B', header, 4, 1)
        struct.pack_into('<B', header, 5, num_classes & 0xFF)
        struct.pack_into('<B', header, 6, 64)
        struct.pack_into('<I', header, 8, 0x0028)
        struct.pack_into('<I', header, 12, base_size)
        struct.pack_into('<I', header, 16, fc_w_offset)
        struct.pack_into('<I', header, 20, fc_w_size)
        struct.pack_into('<I', header, 24, fc_b_offset)
        struct.pack_into('<I', header, 28, fc_b_size)
        struct.pack_into('<I', header, 32, meta_offset)
        hdr_crc = _crc32(bytes(header[:36]))
        struct.pack_into('<I', header, 36, hdr_crc)

        blob = bytes(header) + base_bytes + fc_w.tobytes() + fc_b.tobytes() + bytes(meta)
        with open(output_path, 'wb') as f:
            f.write(blob)
        return len(blob)

    def _on_serial_flash_finished(self, success, msg):
        """处理串口烧录完成事件"""
        self.progress_bar.setValue(100 if success else 0)
        if success and self._current_project:
            pdir, meta = self._current_project
            meta["deployed"] = True
            save_project_meta(pdir, meta)
            self._refresh_project_list()
        elif not success and self._flash_bin_path:
            self._log("串口烧录未成功，询问用户是否重试或改用ST-Link...")
            retry_box = QMessageBox(self)
            retry_box.setWindowTitle("烧录未成功")
            retry_box.setText(
                f"串口烧录未成功（{msg}）。\n\n"
                "请选择操作："
            )
            btn_retry = retry_box.addButton("重试串口烧录", QMessageBox.ActionRole)
            btn_stlink = retry_box.addButton("使用ST-Link烧录", QMessageBox.ActionRole)
            btn_cancel = retry_box.addButton("取消", QMessageBox.RejectRole)
            retry_box.setDefaultButton(btn_retry)
            retry_box.exec()

            if retry_box.clickedButton() == btn_retry:
                self._log("用户选择重试串口烧录...")
                self._serial_flash_worker = SerialFlashWorker(self.sw, self._flash_bin_path, slot=1)
                self._serial_flash_worker.log.connect(self._log)
                self._serial_flash_worker.progress.connect(self.progress_bar.setValue)
                self._serial_flash_worker.finished_signal.connect(self._on_serial_flash_finished)
                self._serial_flash_worker.start()
            elif retry_box.clickedButton() == btn_stlink:
                self._log("用户选择使用ST-Link烧录...")
                self._stlink_flash_worker = StlinkFlashWorker(self._flash_bin_path)
                self._stlink_flash_worker.log.connect(self._log)
                self._stlink_flash_worker.finished_signal.connect(self._on_stlink_flash_finished)
                self._stlink_flash_worker.start()

    def _on_stlink_flash_finished(self, success, msg):
        """处理ST-Link烧录完成事件"""
        if not success:
            QMessageBox.critical(self, "刷写失败", msg)
        if success and self._current_project:
            pdir, meta = self._current_project
            meta["deployed"] = True
            save_project_meta(pdir, meta)
            self._refresh_project_list()

    def _export_header(self):
        """导出C头文件（包含浮点和INT8量化权重）"""
        if self._current_record is None:
            if self._current_project is None:
                QMessageBox.warning(self, "提示", "请先选择训练记录或项目")
                return
            pdir, meta = self._current_project
            model_dir = get_latest_model_dir(pdir)
            pth_path = os.path.join(pdir, model_dir, "weights", "model_weights.pth")
            cfg_path = os.path.join(pdir, model_dir, "evaluation", "model_config.json")
        else:
            rec = self._current_record
            record_dir = rec.get("record_dir", "")
            pth_path = os.path.join(record_dir, "weights", "model_weights.pth")
            cfg_path = os.path.join(record_dir, "evaluation", "model_config.json")
            if not os.path.isfile(pth_path):
                pdir = rec.get("project_dir", "")
                model_dir = get_latest_model_dir(pdir)
                pth_path = os.path.join(pdir, model_dir, "weights", "model_weights.pth")
                cfg_path = os.path.join(pdir, model_dir, "evaluation", "model_config.json")

        if not os.path.isfile(pth_path) or not os.path.isfile(cfg_path):
            QMessageBox.warning(self, "提示", "找不到模型权重或配置文件")
            return

        self._log("正在导出C头文件...")
        QApplication.processEvents()

        try:
            state = torch.load(pth_path, map_location='cpu')
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)

            nc = cfg['num_classes']
            if nc < 2 or nc > 10:
                QMessageBox.warning(self, "导出失败",
                    f"导出的模型类数必须在 2~10 之间（当前 {nc} 类）。\n\n"
                    f"请确认训练参数。")
                self._log(f"[错误] 导出失败: 当前模型为 {nc} 类，不在 2~10 范围内")
                return
            if nc > 6:
                ret_warn = QMessageBox.question(self, "提示",
                    f"当前模型为 {nc} 类。\n"
                    f"编译基座支持最多 10 类（MODEL_MAX_CLASSES=10），\n"
                    f"但 class_to_cmd 的 cmd 值必须 ≤ CMD_MAX（{5 + (nc-6)}）。\n"
                    f"如新增类映射到现有命令（如 pai→start），确认 cmd_map 中已包含。\n\n"
                    f"确定继续导出？",
                    QMessageBox.Yes | QMessageBox.No)
                if ret_warn != QMessageBox.Yes:
                    self._log("导出已取消")
                    return

            ret = QMessageBox.question(self, "确认导出",
                f"导出将覆盖 MCU 项目中的 model_weights.h 和 model_weights_q.h，\n"
                f"需要重新编译固件才能生效。\n\n"
                f"确定继续？",
                QMessageBox.Yes | QMessageBox.No)
            if ret != QMessageBox.Yes:
                self._log("导出已取消")
                return

            cn = cfg['class_names']
            cc = cfg.get('class_to_cmd', [0] * nc)
            has_se = bool(cfg.get('has_se', True))
            se_reduction = int(cfg.get('se_reduction', 4))
            se1_hidden = 48 // se_reduction if has_se else 0
            se2_hidden = 64 // se_reduction if has_se else 0
            conf_threshold = float(cfg.get('conf_threshold', 0.5))
            margin_threshold = float(cfg.get('margin_threshold', 0.1))
            noise_index = int(cfg.get('noise_index', -1))
            out_dir = CORE_INC_DIR
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, "model_weights.h")
            all_params = {}
            for name, tensor in state.items():
                all_params[name] = tensor.numpy()

            with open(out_path, "w", encoding="utf-8") as f:
                f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
                f.write("/* Auto-generated DS-CNN + SE Float32 Weights */\n\n")
                f.write(f"#define NUM_CLASSES     {nc}\n")
                f.write(f"#define FRAME_WINDOW    {FRAME_WINDOW}\n")
                f.write(f"#define MEL_NUM_FILTERS  {MEL_FILTERS}\n")
                f.write(f"#define CONV1_OUT       32\n")
                f.write(f"#define CONV2_OUT       48\n")
                f.write(f"#define CONV3_OUT       64\n")
                f.write(f"#define POOL1_LEN       ({FRAME_WINDOW} / 2)\n")
                f.write(f"#define POOL2_LEN       ({FRAME_WINDOW} / 4)\n")
                f.write(f"#define HAS_SE          {1 if has_se else 0}\n")
                f.write(f"#define SE_REDUCTION    {se_reduction}\n")
                f.write(f"#define SE1_HIDDEN      {se1_hidden}\n")
                f.write(f"#define SE2_HIDDEN      {se2_hidden}\n")
                f.write(f"#define CONF_THRESHOLD  {conf_threshold:.4f}f\n")
                f.write(f"#define MARGIN_THRESHOLD {margin_threshold:.4f}f\n")
                f.write(f"#define NOISE_INDEX     {noise_index}\n\n")
                f.write("static const char* class_labels[NUM_CLASSES] = {\n")
                for name in cn:
                    f.write(f'    "{name}",\n')
                f.write("};\n\n")
                f.write("static const char* class_display[NUM_CLASSES] = {\n")
                for name in cn:
                    f.write(f'    "{name}",\n')
                f.write("};\n\n")
                f.write(f"static const int class_to_cmd[NUM_CLASSES] = {{{', '.join(str(v) for v in cc)}}};\n\n")
                for name, arr in all_params.items():
                    data = arr.flatten()
                    safe = name.replace('.', '_')
                    f.write(f"static const float {safe}[{len(data)}] = {{\n")
                    for i in range(0, len(data), 8):
                        line = ", ".join(f"{float(v):12.8f}f" for v in data[i:i+8])
                        f.write(f"    {line},\n")
                    f.write("};\n\n")
                f.write("#endif\n")

            out_q_path = os.path.join(out_dir, "model_weights_q.h")
            quant_names = ["depth1.weight", "point1.weight",
                           "depth2.weight", "point2.weight", "fc.weight"]
            if has_se:
                quant_names += ["se1.fc1.weight", "se1.fc2.weight",
                                "se2.fc1.weight", "se2.fc2.weight"]
            with open(out_q_path, "w", encoding="utf-8") as f:
                f.write("#ifndef MODEL_WEIGHTS_Q_H\n#define MODEL_WEIGHTS_Q_H\n\n")
                f.write("/* Auto-generated DS-CNN + SE INT8 Quantized Weights */\n\n")
                f.write(f"#define NUM_CLASSES     {nc}\n")
                f.write(f"#define FRAME_WINDOW    {FRAME_WINDOW}\n")
                f.write(f"#define MEL_NUM_FILTERS  {MEL_FILTERS}\n")
                f.write(f"#define CONV1_OUT       32\n")
                f.write(f"#define CONV2_OUT       48\n")
                f.write(f"#define CONV3_OUT       64\n")
                f.write(f"#define POOL1_LEN       ({FRAME_WINDOW} / 2)\n")
                f.write(f"#define POOL2_LEN       ({FRAME_WINDOW} / 4)\n")
                f.write(f"#define HAS_SE          {1 if has_se else 0}\n")
                f.write(f"#define SE_REDUCTION    {se_reduction}\n")
                f.write(f"#define SE1_HIDDEN      {se1_hidden}\n")
                f.write(f"#define SE2_HIDDEN      {se2_hidden}\n")
                f.write(f"#define CONF_THRESHOLD  {conf_threshold:.4f}f\n")
                f.write(f"#define MARGIN_THRESHOLD {margin_threshold:.4f}f\n")
                f.write(f"#define NOISE_INDEX     {noise_index}\n\n")
                f.write("static const char* class_labels[NUM_CLASSES] = {\n")
                for name in cn:
                    f.write(f'    "{name}",\n')
                f.write("};\n\n")
                f.write("static const char* class_display[NUM_CLASSES] = {\n")
                for name in cn:
                    f.write(f'    "{name}",\n')
                f.write("};\n\n")
                f.write(f"static const int class_to_cmd[NUM_CLASSES] = {{{', '.join(str(v) for v in cc)}}};\n\n")
                for qname in quant_names:
                    if qname in all_params:
                        arr = all_params[qname]
                        abs_max = np.max(np.abs(arr))
                        scale = abs_max / 127.0 if abs_max > 0 else 1.0/127.0
                        q = np.round(arr/scale).clip(-128,127).astype(np.int8)
                        safe = qname.replace('.', '_')
                        f.write(f"static const int8_t {safe}_q[{len(q.flatten())}] = {{\n")
                        for i in range(0, len(q.flatten()), 16):
                            line = ",".join(f"{int(v):4d}" for v in q.flatten()[i:i+16])
                            f.write(f"    {line},\n")
                        f.write("};\n\n")
                        f.write(f"static const float {safe}_scale = {scale:.8f}f;\n\n")
                for name, arr in all_params.items():
                    if name in quant_names:
                        continue
                    data = arr.flatten()
                    safe = name.replace('.', '_')
                    f.write(f"static const float {safe}[{len(data)}] = {{\n")
                    for i in range(0, len(data), 8):
                        line = ", ".join(f"{float(v):12.8f}f" for v in data[i:i+8])
                        f.write(f"    {line},\n")
                    f.write("};\n\n")
                f.write("#endif\n")

            self._log(f"C头文件已导出: {out_path}")
            self._log(f"量化头文件已导出: {out_q_path}")
            self._log(f"文件大小: float={os.path.getsize(out_path)} bytes, "
                       f"int8={os.path.getsize(out_q_path)} bytes")
            QMessageBox.information(self, "导出成功",
                                     f"C头文件已导出到:\n{out_path}\n{out_q_path}")

        except Exception as e:
            self._log(f"[错误] 导出失败: {e}")
            QMessageBox.critical(self, "导出失败", str(e))

    def _log(self, msg):
        """添加日志消息"""
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def cleanup(self):
        """清理资源，取消正在运行的训练任务"""
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
