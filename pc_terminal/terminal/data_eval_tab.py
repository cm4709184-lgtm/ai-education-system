# 数据评估与纠错模块：显示整个类的Mel热力图 + 内嵌纠错采集对比
import os
import time

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QComboBox, QLabel, QProgressBar, QSplitter, QTreeWidget,
    QTreeWidgetItem, QHeaderView, QAbstractItemView,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor

import pyqtgraph as pg

from terminal.serial_worker import SerialWorker, CMD_SET_MEL_OUTPUT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
ALLDATA_DIR = os.path.join(BASE_DIR, "alldata")
MEL_FILTERS = 20
COLLECT_FRAMES = 48


class DataEvalTab(QWidget):
    """数据评估面板：本地Mel热力图 + 内嵌纠错采集对比"""

    def __init__(self, serial_worker: SerialWorker):
        super().__init__()
        self.sw = serial_worker
        self._npy_files = []
        self._combined_data = None
        self._sample_offsets = []
        self._npy_cache = {}
        self._collecting = False
        self._collected_frames = []

        self._build_ui()
        self._connect_signals()
        self._refresh_categories()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── 顶部 ──
        top = QHBoxLayout()
        top.addWidget(QLabel("类别:"))
        self.combo_cat = QComboBox()
        self.combo_cat.setMinimumWidth(160)
        top.addWidget(self.combo_cat, 1)
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.setMinimumWidth(50)
        top.addWidget(self.btn_refresh)
        root.addLayout(top)

        # ── 主体：热力图 + 纠错 ──
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：本地 Mel 热力图
        left_group = QGroupBox("本地 Mel 频谱（整个类别）")
        left_lay = QVBoxLayout(left_group)
        left_lay.setContentsMargins(4, 14, 4, 4)

        self.mel_plot = pg.PlotWidget()
        self.mel_plot.setBackground('#0f0f23')
        self.mel_plot.setLabel('bottom', '帧索引')
        self.mel_plot.setYRange(0, 22, padding=0)
        vb = self.mel_plot.getViewBox()
        vb.setMouseEnabled(x=True, y=False)
        vb.setMenuEnabled(False)
        vb.setLimits(minXRange=50)
        self.mel_plot.setLabel('left', '滤波器索引')
        self.mel_plot.showGrid(x=True, y=True, alpha=0.15)
        self.mel_img = pg.ImageItem()
        self.mel_plot.addItem(self.mel_img)
        left_lay.addWidget(self.mel_plot, 1)

        self.lbl_hover = QLabel("悬停查看数值")
        self.lbl_hover.setStyleSheet(
            "color:#8b949e;font-size:11px;padding:2px 4px;"
            "font-family:Consolas,'Microsoft YaHei';"
        )
        left_lay.addWidget(self.lbl_hover)

        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color:#58a6ff;font-size:11px;")
        left_lay.addWidget(self.lbl_info)

        splitter.addWidget(left_group)

        # 右侧：纠错面板
        right_group = QGroupBox("纠错")
        right_lay = QVBoxLayout(right_group)
        right_lay.setContentsMargins(8, 14, 8, 8)
        right_lay.setSpacing(6)

        # 纠错按钮 + 进度
        btn_row = QHBoxLayout()
        self.btn_eval = QPushButton("  开始纠错  ")
        self.btn_eval.setStyleSheet(
            "QPushButton{background:#b94a28;color:#fff;border:1px solid #f0883e;"
            "border-radius:6px;padding:8px 20px;font-size:13px;font-weight:bold;}"
            "QPushButton:hover{background:#d4572a;}"
            "QPushButton:disabled{background:#21262d;color:#484f58;border-color:#30363d;}"
        )
        btn_row.addWidget(self.btn_eval)
        right_lay.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, COLLECT_FRAMES)
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m 帧")
        self.progress.setStyleSheet(
            "QProgressBar{background:#161b22;border:1px solid #30363d;border-radius:3px;"
            "text-align:center;color:#c9d1d9;}"
            "QProgressBar::chunk{background:#f0883e;border-radius:2px;}"
        )
        right_lay.addWidget(self.progress)

        self.lbl_collect = QLabel("")
        self.lbl_collect.setStyleSheet("color:#d29922;font-size:11px;")
        right_lay.addWidget(self.lbl_collect)

        # 采集热力图
        self.collect_plot = pg.PlotWidget()
        self.collect_plot.setBackground('#0f0f23')
        self.collect_plot.setLabel('bottom', '帧')
        self.collect_plot.setYRange(0, 22, padding=0)
        self.collect_plot.getViewBox().setMouseEnabled(x=False, y=False)
        self.collect_img = pg.ImageItem()
        self.collect_plot.addItem(self.collect_img)
        self.collect_data = np.zeros((COLLECT_FRAMES, MEL_FILTERS), dtype=np.float32)
        self.collect_img.setImage(self.collect_data, autoLevels=True)
        right_lay.addWidget(self.collect_plot, 1)

        # 对比结果
        sep = QLabel()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background:#30363d;")
        right_lay.addWidget(sep)

        self.result_tree = QTreeWidget()
        self.result_tree.setHeaderLabels(["文件名", "相似度", "帧数", "类别"])
        self.result_tree.header().setStretchLastSection(False)
        self.result_tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.result_tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.result_tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.result_tree.header().setSectionResizeMode(3, QHeaderView.Stretch)
        self.result_tree.setRootIsDecorated(False)
        self.result_tree.setAlternatingRowColors(True)
        self.result_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.result_tree.setMaximumHeight(250)
        self.result_tree.setStyleSheet(
            "QTreeWidget{background:#0d1117;color:#c9d1d9;border:1px solid #30363d;"
            "border-radius:4px;font-size:12px;}"
            "QTreeWidget::item{padding:2px 0;}"
            "QTreeWidget::item:selected{background:#1f6feb;}"
            "QTreeWidget::item:hover{background:#21262d;}"
            "QHeaderView::section{background:#161b22;color:#c9d1d9;"
            "border:1px solid #30363d;padding:3px;}"
        )
        self.result_tree.setMouseTracking(True)
        self.result_tree.currentItemChanged.connect(self._on_result_selected)
        right_lay.addWidget(self.result_tree, 1)

        self.lbl_detail = QLabel("")
        self.lbl_detail.setStyleSheet("color:#8b949e;font-size:11px;")
        self.lbl_detail.setWordWrap(True)
        right_lay.addWidget(self.lbl_detail)

        splitter.addWidget(right_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

    def _connect_signals(self):
        self.combo_cat.currentIndexChanged.connect(self._on_cat_changed)
        self.btn_refresh.clicked.connect(self._refresh_categories)
        self.btn_eval.clicked.connect(self._on_eval_toggle)
        self.sw.mel_received.connect(self._on_mel)
        self._mouse_proxy = pg.SignalProxy(
            self.mel_plot.scene().sigMouseMoved,
            rateLimit=30, slot=self._on_mouse_moved,
        )

    # ── 类别管理 ──

    def _refresh_categories(self):
        self.combo_cat.blockSignals(True)
        self.combo_cat.clear()
        if os.path.isdir(ALLDATA_DIR):
            cats = sorted(
                d for d in os.listdir(ALLDATA_DIR)
                if os.path.isdir(os.path.join(ALLDATA_DIR, d))
            )
            self.combo_cat.addItems(cats)
        self.combo_cat.blockSignals(False)
        if self.combo_cat.count() > 0:
            self._on_cat_changed(0)

    def _on_cat_changed(self, idx):
        cat = self.combo_cat.currentText().strip()
        self._load_class(cat)
        self.result_tree.clear()
        self.lbl_detail.setText("")
        self.collect_data[:] = 0
        self.collect_img.setImage(self.collect_data, autoLevels=True)
        self.lbl_collect.setText("")

    def _load_class(self, cat):
        self._npy_files = []
        self._sample_offsets = []
        self._npy_cache = {}
        cat_dir = os.path.join(ALLDATA_DIR, cat)
        if not os.path.isdir(cat_dir):
            self.mel_img.setImage(np.zeros((MEL_FILTERS, 10)))
            self.lbl_info.setText("无数据")
            return

        for f in sorted(os.listdir(cat_dir)):
            if f.endswith('.npy') and os.path.isfile(os.path.join(cat_dir, f)):
                self._npy_files.append(os.path.join(cat_dir, f))

        if not self._npy_files:
            self.mel_img.setImage(np.zeros((MEL_FILTERS, 10)))
            self.lbl_info.setText("无 .npy 文件")
            return

        gap = 2
        blank = np.full((gap, MEL_FILTERS), np.nan, dtype=np.float32)
        all_blocks = []
        self._sample_offsets = []

        for path in self._npy_files:
            try:
                data = np.load(path).astype(np.float32)
            except Exception:
                continue
            if data.ndim != 2 or data.shape[1] != MEL_FILTERS:
                continue
            self._npy_cache[path] = data
            start = sum(b.shape[0] for b in all_blocks)
            self._sample_offsets.append((start, data.shape[0], path))
            all_blocks.append(data)
            all_blocks.append(blank)

        if not all_blocks:
            self.lbl_info.setText("无有效数据")
            return

        combined = np.concatenate(all_blocks, axis=0)
        self._combined_data = combined

        self.mel_img.setImage(combined, autoLevels=True)
        self.mel_img.setRect(0, 0, combined.shape[0], MEL_FILTERS)
        self.mel_plot.getViewBox().setLimits(maxXRange=combined.shape[0])

        self.lbl_info.setText(
            f"{len(self._sample_offsets)} 个样本  |  共 {combined.shape[0]} 帧  |  "
            f"范围 [{np.nanmin(combined):.1f}, {np.nanmax(combined):.1f}]"
        )

    # ── 鼠标悬停 ──

    def _on_mouse_moved(self, evt):
        pos = evt[0]
        vb = self.mel_plot.getViewBox()
        if vb is None or self._combined_data is None:
            return
        mp = vb.mapSceneToView(pos)
        x, y = int(mp.x()), int(mp.y())
        total_frames, n_filters = self._combined_data.shape
        if 0 <= x < total_frames and 0 <= y < n_filters:
            val = self._combined_data[x, y]
            sample_name = ""
            for start, nf, path in self._sample_offsets:
                if start <= x < start + nf:
                    sample_name = os.path.basename(path)
                    break
            v_str = f"{val:.4f}" if not np.isnan(val) else "gap"
            self.lbl_hover.setText(
                f"帧:{x}  滤波器:{y}  值:{v_str}  样本:{sample_name}"
            )
        else:
            self.lbl_hover.setText("悬停查看数值")

    def _on_result_selected(self, current, previous):
        if current is None:
            return
        path = current.data(0, Qt.UserRole)
        if not path or not self._sample_offsets:
            return
        for start, nf, spath in self._sample_offsets:
            if spath == path:
                self.mel_plot.setXRange(start, start + nf, padding=0.05)
                break

    # ── 纠错采集 ──

    def _on_eval_toggle(self):
        if self._collecting:
            self._stop_collect()
            return
        if not self.sw.connected:
            self.lbl_collect.setText("未连接串口")
            self.lbl_collect.setStyleSheet("color:#f85149;font-size:11px;")
            return
        if not self._npy_files:
            self.lbl_collect.setText("当前类别没有样本数据")
            self.lbl_collect.setStyleSheet("color:#f85149;font-size:11px;")
            return

        self._collecting = True
        self._collected_frames = []
        self.collect_data[:] = 0
        self.result_tree.clear()
        self.btn_eval.setText("  停止采集  ")
        self.btn_eval.setStyleSheet(
            "QPushButton{background:#da3633;color:#fff;border:1px solid #f85149;"
            "border-radius:6px;padding:8px 20px;font-size:13px;font-weight:bold;}"
            "QPushButton:hover{background:#f85149;}"
        )
        self.progress.setValue(0)
        self.lbl_collect.setText("正在采集...")
        self.lbl_collect.setStyleSheet("color:#d29922;font-size:11px;")
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._collect_tick)
        self._timer.start(50)

    def _collect_tick(self):
        n = len(self._collected_frames)
        self.progress.setValue(n)
        self.lbl_collect.setText(f"已采集 {n}/{COLLECT_FRAMES} 帧")
        if n >= COLLECT_FRAMES:
            self._stop_collect()

    def _stop_collect(self):
        self._timer.stop()
        self._collecting = False
        self.btn_eval.setText("  开始纠错  ")
        self.btn_eval.setStyleSheet(
            "QPushButton{background:#b94a28;color:#fff;border:1px solid #f0883e;"
            "border-radius:6px;padding:8px 20px;font-size:13px;font-weight:bold;}"
            "QPushButton:hover{background:#d4572a;}"
        )
        n = len(self._collected_frames)
        if n > 0:
            self.lbl_collect.setText(f"采集完成 {n} 帧")
            self.lbl_collect.setStyleSheet("color:#3fb950;font-size:11px;")
            self._run_comparison(self._collected_frames)
        else:
            self.lbl_collect.setText("采集超时")
            self.lbl_collect.setStyleSheet("color:#f85149;font-size:11px;")

    def _on_mel(self, mel):
        if self._collecting:
            self._collected_frames.append(list(mel))
            col = np.array(mel, dtype=np.float32)
            self.collect_data = np.roll(self.collect_data, -1, axis=0)
            self.collect_data[-1] = col
            self.collect_img.setImage(self.collect_data, autoLevels=True)

    # ── 对比 ──

    @staticmethod
    def _trim_silence(data):
        """去除首尾静音帧，只保留有声音变化的部分"""
        if data.shape[0] < 3:
            return data
        # 每帧能量 = 滤波器值的标准差（静音帧std接近0，有声帧std较大）
        energy = np.std(data, axis=1)
        max_energy = np.max(energy)
        if max_energy < 0.01:
            return data
        threshold = max_energy * 0.15
        # 找到首尾超过阈值的位置
        active = np.where(energy > threshold)[0]
        if len(active) == 0:
            return data
        start = max(0, active[0] - 1)
        end = min(data.shape[0], active[-1] + 2)
        return data[start:end]

    @staticmethod
    def _resample_frames(data, target_frames):
        n_frames = data.shape[0]
        if n_frames == target_frames:
            return data
        src_idx = np.linspace(0, n_frames - 1, n_frames)
        dst_idx = np.linspace(0, n_frames - 1, target_frames)
        resampled = np.zeros((target_frames, data.shape[1]), dtype=np.float32)
        for f in range(data.shape[1]):
            resampled[:, f] = np.interp(dst_idx, src_idx, data[:, f])
        return resampled

    def _run_comparison(self, frames):
        collected = np.array(frames, dtype=np.float32)

        # 去除静音帧：只保留有声音变化的部分
        collected = self._trim_silence(collected)
        if collected.shape[0] < 5:
            self.lbl_detail.setText("采集数据有效帧不足")
            return

        TARGET_FRAMES = 32
        collected_resampled = self._resample_frames(collected, TARGET_FRAMES)
        flat_collected = collected_resampled.flatten()
        norm_a = np.linalg.norm(flat_collected)
        if norm_a == 0:
            self.lbl_detail.setText("采集数据全零")
            return
        flat_norm = flat_collected / norm_a

        results = []
        for path in self._npy_files:
            stored = self._npy_cache.get(path)
            if stored is None:
                continue
            stored_trimmed = self._trim_silence(stored)
            if stored_trimmed.shape[0] < 5:
                continue
            stored_resampled = self._resample_frames(stored_trimmed, TARGET_FRAMES)
            flat_stored = stored_resampled.flatten()
            norm_b = np.linalg.norm(flat_stored)
            sim = float(np.dot(flat_norm, flat_stored / norm_b)) if norm_b > 0 else 0.0
            sim = max(0.0, min(1.0, sim))
            results.append((path, sim, stored.shape[0]))

        if not results:
            self.lbl_detail.setText("没有可对比的样本")
            return

        results.sort(key=lambda x: -x[1])

        self.result_tree.clear()
        for path, sim, n_frames in results:
            fname = os.path.basename(path)
            display_pct = max(0, int((sim - 0.90) * 1000)) if sim >= 0.90 else 0
            parts = path.replace("\\", "/").split("/")
            cat_name = parts[-2] if len(parts) >= 2 else "?"

            item = QTreeWidgetItem(self.result_tree)
            item.setText(0, fname)
            item.setText(1, f"{display_pct}%")
            item.setText(2, str(n_frames))
            item.setText(3, cat_name)
            item.setToolTip(0, f"文件: {path}\n帧数: {n_frames}\n相似度: {display_pct}")
            item.setData(0, Qt.UserRole, path)

            if display_pct >= 80:
                item.setForeground(1, QColor("#3fb950"))
            elif display_pct >= 60:
                item.setForeground(1, QColor("#d29922"))
            else:
                item.setForeground(1, QColor("#f85149"))

        best = results[0]
        best_pct = max(0, int((best[1] - 0.90) * 1000)) if best[1] >= 0.90 else 0
        self.lbl_detail.setText(
            f"最佳: {os.path.basename(best[0])} ({best_pct})"
        )

        self.result_tree.resizeColumnToContents(0)
        self.result_tree.resizeColumnToContents(1)
        self.result_tree.resizeColumnToContents(2)
