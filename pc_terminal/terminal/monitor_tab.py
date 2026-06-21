# MCU与PC双端推理结果的实时监控面板
import re, os, json, glob
import numpy as np
from collections import deque
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QGroupBox, QComboBox, QPushButton, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
import pyqtgraph as pg

CMD_COLORS = {
    'backward': '#FF9800', 'forward': '#4CAF50', 'start': '#F44336',
    'stop': '#F44336', 'turn': '#FFEB3B', 'noise': '#9E9E9E',
}  # 指令对应的颜色映射
MEL_WINDOW = 32  # PC推理所需的Mel帧窗口大小
MEL_FILTERS = 20  # Mel滤波器组数量
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def scan_projects():
    """扫描projects目录，返回所有可用模型列表"""
    projects_dir = os.path.join(BASE_DIR, 'projects')
    models = []
    if not os.path.isdir(projects_dir):
        return models
    for proj in sorted(os.listdir(projects_dir)):
        proj_path = os.path.join(projects_dir, proj)
        if not os.path.isdir(proj_path):
            continue
        for model in sorted(os.listdir(proj_path)):
            model_path = os.path.join(proj_path, model)
            wt_path = os.path.join(model_path, 'weights')
            eval_path = os.path.join(model_path, 'evaluation')
            ds_path = os.path.join(model_path, 'dataset')
            if not os.path.isdir(wt_path):
                continue
            pth = os.path.join(wt_path, 'model_weights.pth')
            cfg_path = os.path.join(eval_path, 'model_config.json')
            if not os.path.exists(cfg_path):
                cfg_path = os.path.join(wt_path, 'model_config.json')
            class_names_raw = []
            class_display = []
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, 'r', encoding='utf-8') as f:
                        cfg = json.load(f)
                    class_names_raw = cfg.get('class_names', [])
                    class_display = cfg.get('class_display', class_names_raw)
                except Exception:
                    class_names_raw = sorted([d for d in os.listdir(ds_path)
                                      if os.path.isdir(os.path.join(ds_path, d))]) if os.path.isdir(ds_path) else []
                    class_display = class_names_raw
            else:
                class_names_raw = sorted([d for d in os.listdir(ds_path)
                                  if os.path.isdir(os.path.join(ds_path, d))]) if os.path.isdir(ds_path) else []
                class_display = class_names_raw
            if os.path.exists(pth):
                models.append({
                    'label': f'{proj}/{model} ({len(class_display)}类)',
                    'project': proj,
                    'model': model,
                    'path': model_path,
                    'class_names': class_names_raw,
                    'class_display': class_display,
                    'weights': pth,
                })
    return models

class MonitorTab(QWidget):
    """实时监控MCU和PC端推理结果、Mel频谱和系统状态"""
    def __init__(self, serial_worker):
        super().__init__()
        self.sw = serial_worker
        self.mel_buf = deque(maxlen=200)
        self._mcu_names = []
        self._mcu_maps = []
        self._pc_model = None
        self._pc_names = []
        self._build_ui()
        self.sw.mel_received.connect(self._on_mel)
        self.sw.status_received.connect(self._on_status)
        self.sw.log_received.connect(self._on_log)
        self.sw.model_info_received.connect(self._on_model_info)

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._poll_status)
        self._status_timer.start(200)

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content = QWidget()
        layout = QHBoxLayout(content)
        left = QVBoxLayout()
        right = QVBoxLayout()

        mcu_group = QGroupBox('🔵 MCU 推理结果')
        mcu_lay = QVBoxLayout(mcu_group)
        self.mcu_cmd_label = QLabel('---')
        self.mcu_cmd_label.setAlignment(Qt.AlignCenter)
        self.mcu_cmd_label.setFont(QFont('Microsoft YaHei', 36, QFont.Bold))
        self.mcu_cmd_label.setStyleSheet('color: #c9d1d9; background: #1a1a2e; border-radius: 12px; padding: 16px;')
        self.mcu_cmd_label.setMinimumHeight(100)
        mcu_lay.addWidget(self.mcu_cmd_label)
        self.mcu_rms_label = QLabel('RMS: --')
        self.mcu_rms_label.setAlignment(Qt.AlignCenter)
        self.mcu_rms_label.setStyleSheet('color: #c9d1d9; font-size: 14px;')
        mcu_lay.addWidget(self.mcu_rms_label)
        left.addWidget(mcu_group)

        pc_group = QGroupBox('🟢 PC 推理结果')
        pc_top = QHBoxLayout()
        pc_top.addWidget(QLabel('推理模型:'))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(150)
        self.model_combo.setMaximumWidth(250)
        self.model_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        pc_top.addWidget(self.model_combo)
        self.btn_refresh_models = QPushButton('刷新')
        self.btn_refresh_models.clicked.connect(self._refresh_models)
        pc_top.addWidget(self.btn_refresh_models)
        pc_lay = QVBoxLayout(pc_group)
        pc_lay.addLayout(pc_top)
        self.pc_cmd_label = QLabel('---')
        self.pc_cmd_label.setAlignment(Qt.AlignCenter)
        self.pc_cmd_label.setFont(QFont('Microsoft YaHei', 28, QFont.Bold))
        self.pc_cmd_label.setStyleSheet('color: #c9d1d9; background: #1a1a2e; border-radius: 12px; padding: 12px;')
        self.pc_cmd_label.setMinimumHeight(70)
        pc_lay.addWidget(self.pc_cmd_label)
        self.pc_prob_label = QLabel('')
        self.pc_prob_label.setAlignment(Qt.AlignCenter)
        self.pc_prob_label.setStyleSheet('color: #c9d1d9; font-size: 12px;')
        self.pc_prob_label.setWordWrap(True)
        pc_lay.addWidget(self.pc_prob_label)
        left.addWidget(pc_group)

        self.info_group = QGroupBox('系统状态')
        info_lay = QGridLayout(self.info_group)
        self.lbl_source = QLabel('--')
        self.lbl_classes = QLabel('--')
        self.lbl_state = QLabel('--')
        self.lbl_mel_count = QLabel('0')
        self.lbl_distance = QLabel('--')
        for i, (k, v) in enumerate([
            ('模型来源', self.lbl_source), ('类别数', self.lbl_classes),
            ('系统状态', self.lbl_state), ('Mel帧数', self.lbl_mel_count),
            ('超声波', self.lbl_distance)
        ]):
            info_lay.addWidget(QLabel(k), i, 0)
            info_lay.addWidget(v, i, 1)
        left.addWidget(self.info_group)

        self.mel_plot = pg.PlotWidget(title='Mel 频谱 (MCU → PC)')
        self.mel_plot.setLabel('bottom', '时间帧')
        self.mel_plot.setLabel('left', 'Mel 频带')
        self.mel_plot.setBackground('#0f0f23')
        self.mel_img = pg.ImageItem()
        self.mel_plot.addItem(self.mel_img)
        self.mel_data = np.zeros((MEL_FILTERS, 200))
        right.addWidget(self.mel_plot, stretch=3)

        self.log_text = QLabel('')
        self.log_text.setWordWrap(True)
        self.log_text.setStyleSheet('color: #8b949e; background: #0d1117; padding: 8px; border-radius: 4px; font-family: Consolas; font-size: 11px;')
        self.log_text.setMaximumHeight(80)
        right.addWidget(self.log_text)

        layout.addLayout(left, stretch=1)
        layout.addLayout(right, stretch=2)

        scroll.setWidget(content)
        outer.addWidget(scroll)

        self._refresh_models()

    def _refresh_models(self):
        """刷新模型列表"""
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self._models = scan_projects()
        for m in self._models:
            self.model_combo.addItem(m['label'])
        self.model_combo.blockSignals(False)
        if self._models:
            self._on_model_changed(0)

    def _on_model_changed(self, idx):
        """加载选中的模型权重（含 SE + 拒识门限，与 MCU 端保持一致）"""
        if idx < 0 or idx >= len(self._models):
            return
        m = self._models[idx]
        try:
            import torch, json
            from mel_iterate import DSCNN, load_config
            nc = len(m['class_names'])
            self._pc_model = DSCNN(nc)
            state = torch.load(m['weights'], map_location='cpu')
            try:
                self._pc_model.load_state_dict(state, strict=True)
            except RuntimeError as e:
                # 兜底兼容老权重
                self._pc_model.load_state_dict(state, strict=False)
                print(f"[PC模型] 加载非严格模式: {e}")
            self._pc_model.eval()

            # 加载 cfg（拒识门限 + 类别名）
            cfg = load_config(m['weights'], nc)
            self._pc_model.conf_threshold   = cfg.get('conf_threshold', 0.7)
            self._pc_model.margin_threshold = cfg.get('margin_threshold', 0.2)
            self._pc_model.noise_index      = cfg.get('noise_index', nc - 1)

            self._pc_names = m['class_display']
            cfg_path = os.path.join(m['path'], 'evaluation', 'model_config.json')
            if not os.path.exists(cfg_path):
                cfg_path = os.path.join(m['path'], 'weights', 'model_config.json')
            if os.path.exists(cfg_path):
                with open(cfg_path, encoding='utf-8') as f:
                    cfg = json.load(f)
                if 'class_display' in cfg:
                    self._pc_names = cfg['class_display']
            if self.sw.connected and len(self._mcu_names) == nc:
                self._pc_names = self._mcu_names
            self.pc_cmd_label.setText(f'已加载: {m["model"]} ({nc}类)')
            self.pc_cmd_label.setStyleSheet('color: #4CAF50; background: #1a1a2e; border-radius: 12px; padding: 12px;')
            self.pc_cmd_label.setToolTip(
                f'{m["weights"]}\n'
                f'conf={self._pc_model.conf_threshold:.2f}, '
                f'margin={self._pc_model.margin_threshold:.2f}'
            )
        except Exception as e:
            self._pc_model = None
            err = str(e)
            short = err[:50] + ('...' if len(err) > 50 else '')
            self.pc_cmd_label.setText(f'加载失败: {short}')
            self.pc_cmd_label.setStyleSheet('color: #f44336; background: #1a1a2e; border-radius: 12px; padding: 12px;')
            self.pc_cmd_label.setToolTip(err)

    def _on_mel(self, mel):
        """接收Mel帧数据，更新频谱图并触发PC推理"""
        self.mel_buf.append(mel)
        self.lbl_mel_count.setText(str(len(self.mel_buf)))
        col = np.array(mel)
        self.mel_data = np.roll(self.mel_data, -1, axis=1)
        self.mel_data[:, -1] = col
        self.mel_img.setImage(self.mel_data.T, autoLevels=True)
        self._run_pc_inference()

    def _poll_status(self):
        if self.sw.connected:
            self.sw.query_status()

    def _on_status(self, src, nc, state, cmd, progress, distance):
        """处理MCU状态信息"""
        src_name = {0: '编译基座', 1: 'Flash'}.get(src, f'未知({src})')
        self.lbl_source.setText(src_name)
        self.lbl_classes.setText(str(nc))
        state_name = {0: 'OFF', 1: 'IDLE', 2: 'LISTENING'}.get(state, f'未知({state})')
        self.lbl_state.setText(state_name)
        self.lbl_distance.setText(f'{distance:.0f}mm')
        if not self._mcu_names:
            QTimer.singleShot(500, self.sw.query_model_info)

    def _on_model_info(self, src, nc, names, maps):
        """处理MCU模型信息"""
        self._mcu_names = names
        self._mcu_maps = maps
        src_name = {0: '编译基座', 1: 'Flash'}.get(src, f'未知({src})')
        self.lbl_source.setText(src_name)
        self.lbl_classes.setText(str(nc))
        labels = ', '.join(names)
        short_labels = labels[:30] + ('...' if len(labels) > 30 else '')
        self.info_group.setTitle(f'模型: {src_name} ({nc}类) [{short_labels}]')
        self.info_group.setToolTip(labels)
        if self._pc_model is not None:
            cur_idx = self.model_combo.currentIndex()
            if 0 <= cur_idx < len(self._models):
                self._on_model_changed(cur_idx)

    def _on_log(self, msg):
        """解析MCU日志，更新推理结果和RMS显示"""
        if msg.startswith('[DBG]'):
            m = re.search(r'\[DBG\]\s+(\S+)\s+(\d+)%\s+rms:(\d+)', msg)
            if m:
                label = m.group(1)
                conf = int(m.group(2))
                rms = int(m.group(3))
                self.mcu_rms_label.setText(f'RMS: {rms}')
                color = CMD_COLORS.get(label, '#888')
                self.mcu_cmd_label.setText(f'{label} {conf}%')
                self.mcu_cmd_label.setStyleSheet(f'color: {color}; background: #1a1a2e; border-radius: 12px; padding: 16px;')
        elif msg.startswith('[CMD]'):
            m = re.search(r'\[CMD\]\s+(\S+)', msg)
            if m:
                label = m.group(1)
                color = CMD_COLORS.get(label, '#F44336')
                self.mcu_cmd_label.setText(f'>>> {label}')
                self.mcu_cmd_label.setStyleSheet(f'color: {color}; background: #1a1a2e; border-radius: 12px; padding: 16px; font-weight: bold;')
        self.log_text.setText(msg[:120])

    def _run_pc_inference(self):
        """执行PC端模型推理：使用 predict()（含 conf/margin 拒识），与 MCU 行为一致"""
        if self._pc_model is None or len(self.mel_buf) < MEL_WINDOW:
            return
        window = list(self.mel_buf)[-MEL_WINDOW:]
        arr = np.array(window, dtype=np.float32)
        mean = arr.mean(axis=1, keepdims=True)
        std = arr.std(axis=1, keepdims=True) + 1e-6
        arr = (arr - mean) / std
        try:
            import torch
            x = torch.tensor(arr).unsqueeze(0)
            with torch.no_grad():
                # 用带拒识的 predict()，与 MCU softmax_with_reject 行为一致
                idx_t, probs_t = self._pc_model.predict(x)
                idx = int(idx_t.item())
                p = probs_t.numpy()[0]
            noise_idx = self._pc_model.noise_index
            name = self._pc_names[idx] if idx < len(self._pc_names) else f'C{idx}'
            color = CMD_COLORS.get(name, '#888')
            # 拒识后：name 是 noise 类，颜色用灰色
            if idx == noise_idx:
                color = '#9E9E9E'
                self.pc_cmd_label.setText(f'{name} {p[idx]*100:.0f}% (拒识)')
            else:
                self.pc_cmd_label.setText(f'{name} {p[idx]*100:.0f}%')
            self.pc_cmd_label.setStyleSheet(f'color: {color}; background: #1a1a2e; border-radius: 12px; padding: 12px;')
            parts = []
            for i in range(len(p)):
                n = self._pc_names[i] if i < len(self._pc_names) else f'C{i}'
                parts.append(f'{n}:{p[i]*100:.0f}%')
            self.pc_prob_label.setText('  '.join(parts))
        except Exception as e:
            print(f'PC推理异常: {e}')

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_models()
