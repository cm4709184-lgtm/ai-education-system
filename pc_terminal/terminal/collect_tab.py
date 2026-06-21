# 语音指令数据采集模块，用于录制Mel帧数据并管理数据集
import os
import time
import random
import shutil
import threading
from datetime import datetime
from collections import deque

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar, QTreeWidget,
    QTreeWidgetItem, QTextEdit, QLabel, QLineEdit, QSplitter,
    QMessageBox, QHeaderView, QAbstractItemView,
)
from PySide6.QtCore import Qt, Signal, QThread

from terminal.serial_worker import (
    SerialWorker, CMD_SET_MEL_OUTPUT, build_cmd_frame,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLDATA_DIR = os.path.join(BASE_DIR, "alldata")  # 采集数据存放目录

MEL_FILTERS = 20  # Mel滤波器组数量
DEFAULT_FPS = 32  # 默认帧率
DEFAULT_SEG_SEC = 1.5  # 默认录音段时长(秒)
DEFAULT_SAMPLES = 50  # 默认采样条数


class RecordThread(QThread):
    """后台录制线程，负责收集Mel帧数据并保存为npy文件"""
    progress = Signal(int, int)
    log = Signal(str)
    finished = Signal(bool, int, int)
    file_saved = Signal(str)

    def __init__(self, serial_worker, class_name, seg_sec, num_samples, lock, mel_queue):
        super().__init__()
        self.sw = serial_worker
        self.class_name = class_name
        self.seg_sec = seg_sec
        self.num_samples = num_samples
        self.lock = lock
        self.mel_queue = mel_queue
        self._paused = False
        self._stop_flag = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._stop_flag = True
        self._paused = False

    def run(self):
        """执行录制任务：启用Mel输出、收集帧数据、保存为npy文件"""
        save_dir = os.path.join(ALLDATA_DIR, self.class_name)
        os.makedirs(save_dir, exist_ok=True)

        self.log.emit("发送 Mel 输出启用命令...")
        self.sw.send_command(CMD_SET_MEL_OUTPUT, bytes([1]))
        time.sleep(0.3)

        self.log.emit("等待 Mel 帧数据流...")
        wait_start = time.time()
        mel_detected = False
        while time.time() - wait_start < 5.0:
            if self._stop_flag:
                self.log.emit("[已停止] 用户取消")
                self.finished.emit(False, 0, 0)
                return
            with self.lock:
                if len(self.mel_queue) > 0:
                    mel_detected = True
                    break
            time.sleep(0.1)

        if not mel_detected:
            self.log.emit("[警告] 未检测到 Mel 帧数据，请确认 MCU 已开始输出")
            self.log.emit("请确认设备已连接且 MCU 正在发送 Mel 帧")
            self.finished.emit(False, 0, 0)
            return

        self.log.emit(f"[就绪] 已检测到 Mel 帧，开始录制 {self.num_samples} 条样本")

        seg_frames = max(1, int(self.seg_sec * DEFAULT_FPS))
        saved_count = 0
        skip_count = 0

        for sample_idx in range(1, self.num_samples + 1):
            if self._stop_flag:
                self.log.emit(f"[已停止] 在第 {saved_count}/{self.num_samples} 条时停止")
                break

            while self._paused:
                if self._stop_flag:
                    break
                time.sleep(0.1)
            if self._stop_flag:
                break

            with self.lock:
                self.mel_queue.clear()

            time.sleep(0.05)

            frames = []
            deadline = time.time() + self.seg_sec + 1.0

            while len(frames) < seg_frames:
                if self._stop_flag:
                    break
                while self._paused:
                    if self._stop_flag:
                        break
                    time.sleep(0.1)
                if self._stop_flag:
                    break

                with self.lock:
                    while self.mel_queue and len(frames) < seg_frames:
                        frames.append(self.mel_queue.popleft())

                if len(frames) < seg_frames:
                    if time.time() > deadline:
                        self.log.emit(
                            f"[警告] 样本 {sample_idx} 超时，已收集 {len(frames)}/{seg_frames} 帧"
                        )
                        break
                    time.sleep(0.01)

            if self._stop_flag:
                break

            if len(frames) < 5:
                self.log.emit(f"[跳过] 样本 {sample_idx} 帧数不足 ({len(frames)})")
                skip_count += 1
                continue

            data = np.array(frames, dtype=np.float32)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            rand_hex = f"{random.randint(0, 0xFFFF):04x}"
            filename = f"seg_{ts}_{rand_hex}.npy"
            filepath = os.path.join(save_dir, filename)
            np.save(filepath, data)

            saved_count += 1
            self.file_saved.emit(filepath)
            self.progress.emit(sample_idx, self.num_samples)
            self.log.emit(
                f"[{sample_idx}/{self.num_samples}] 已保存 {filename} ({data.shape[0]}帧×{data.shape[1]}mel)"
            )

            gap = max(0.1, self.seg_sec * 0.3)
            gap_end = time.time() + gap
            while time.time() < gap_end:
                if self._stop_flag:
                    break
                time.sleep(0.05)

        self.sw.send_command(CMD_SET_MEL_OUTPUT, bytes([0]))

        if not self._stop_flag:
            self.log.emit(
                f"[完成] 类别「{self.class_name}」录制完成，"
                f"成功 {saved_count} 条，跳过 {skip_count} 条"
            )
        self.finished.emit(True, saved_count, skip_count)


class CollectTab(QWidget):
    """数据采集UI面板，提供录制、暂停、停止和数据管理功能"""
    log_signal = Signal(str)
    progress_signal = Signal(int, int)
    finished_signal = Signal(bool, int, int)

    def __init__(self, serial_worker: SerialWorker):
        super().__init__()
        self.sw = serial_worker
        self._recording = False
        self._paused = False
        self._mel_queue = deque()
        self._lock = threading.Lock()
        self._record_thread = None
        self._session_files = []

        os.makedirs(ALLDATA_DIR, exist_ok=True)

        self._build_ui()
        self._connect_signals()
        self._refresh_class_combo()
        self._refresh_tree()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_lay = QVBoxLayout(left_panel)
        left_lay.setContentsMargins(0, 0, 0, 0)

        record_group = QGroupBox("数据采集")
        rg = QVBoxLayout(record_group)

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("模式:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["录制已有类别", "录制新类别"])
        row_mode.addWidget(self.combo_mode, 1)
        rg.addLayout(row_mode)

        self.exist_widget = QWidget()
        ew_lay = QHBoxLayout(self.exist_widget)
        ew_lay.setContentsMargins(0, 0, 0, 0)
        ew_lay.addWidget(QLabel("类别:"))
        self.combo_class = QComboBox()
        ew_lay.addWidget(self.combo_class, 1)
        rg.addWidget(self.exist_widget)

        self.new_widget = QWidget()
        nw_lay = QHBoxLayout(self.new_widget)
        nw_lay.setContentsMargins(0, 0, 0, 0)
        nw_lay.addWidget(QLabel("类别名:"))
        self.edit_new_class = QLineEdit()
        self.edit_new_class.setPlaceholderText("显示名称 (如 paishou)")
        nw_lay.addWidget(self.edit_new_class, 1)
        nw_lay.addWidget(QLabel("基础命令:"))
        self.combo_base_cmd = QComboBox()
        nw_lay.addWidget(self.combo_base_cmd, 1)
        rg.addWidget(self.new_widget)
        self.new_widget.hide()

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("采样条数:"))
        self.spin_samples = QSpinBox()
        self.spin_samples.setRange(1, 9999)
        self.spin_samples.setValue(DEFAULT_SAMPLES)
        param_row.addWidget(self.spin_samples)
        param_row.addWidget(QLabel("段时长(秒):"))
        self.spin_seg = QDoubleSpinBox()
        self.spin_seg.setRange(1.5, 1.5)
        self.spin_seg.setValue(1.5)
        self.spin_seg.setEnabled(False)
        param_row.addWidget(self.spin_seg)
        rg.addLayout(param_row)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("开始录制")
        self.btn_start.setStyleSheet(
            "QPushButton{background:#238636;color:#fff;border:1px solid #2ea043;"
            "border-radius:6px;padding:6px 16px;}"
            "QPushButton:hover{background:#2ea043;}"
            "QPushButton:disabled{background:#21262d;color:#484f58;border-color:#30363d;}"
        )
        self.btn_pause = QPushButton("暂停")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setStyleSheet(
            "QPushButton{background:#da3633;color:#fff;border:1px solid #f85149;"
            "border-radius:6px;padding:6px 16px;}"
            "QPushButton:hover{background:#f85149;}"
            "QPushButton:disabled{background:#21262d;color:#484f58;border-color:#30363d;}"
        )
        for b in (self.btn_start, self.btn_pause, self.btn_stop):
            b.setMinimumWidth(80)
            btn_row.addWidget(b)
        rg.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet(
            "QProgressBar{background:#161b22;border:1px solid #30363d;border-radius:4px;"
            "text-align:center;color:#c9d1d9;height:20px;}"
            "QProgressBar::chunk{background:#58a6ff;border-radius:3px;}"
        )
        rg.addWidget(self.progress)

        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color:#c9d1d9;")
        rg.addWidget(self.lbl_status)

        left_lay.addWidget(record_group)

        log_group = QGroupBox("采集日志")
        lg_lay = QVBoxLayout(log_group)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(
            "QTextEdit{background:#0d1117;color:#c9d1d9;border:1px solid #30363d;"
            "border-radius:4px;font-family:Consolas,'Microsoft YaHei';font-size:12px;}"
        )
        self.log_view.setMinimumHeight(120)
        lg_lay.addWidget(self.log_view)
        left_lay.addWidget(log_group)

        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_lay = QVBoxLayout(right_panel)
        right_lay.setContentsMargins(0, 0, 0, 0)

        manage_group = QGroupBox("数据管理")
        mg_lay = QVBoxLayout(manage_group)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["名称", "信息"])
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setStyleSheet(
            "QTreeWidget{background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;}"
            "QTreeWidget::item:selected{background:#1f6feb;}"
            "QTreeWidget::item:hover{background:#21262d;}"
            "QHeaderView::section{background:#161b22;color:#c9d1d9;border:1px solid #30363d;padding:4px;}"
        )
        mg_lay.addWidget(self.tree)

        tree_btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.setMinimumWidth(80)
        self.btn_del_selected = QPushButton("删除选中")
        self.btn_del_selected.setMinimumWidth(80)
        self.btn_del_selected.setStyleSheet(
            "QPushButton{background:#da3633;color:#fff;border:1px solid #f85149;"
            "border-radius:6px;padding:6px 16px;}"
            "QPushButton:hover{background:#f85149;}"
        )
        self.btn_clean_empty = QPushButton("清理空文件夹")
        self.btn_clean_empty.setMinimumWidth(80)
        self.btn_clean_empty.setStyleSheet(
            "QPushButton{background:#21262d;color:#c9d1d9;border:1px solid #30363d;"
            "border-radius:6px;padding:6px 16px;}"
            "QPushButton:hover{background:#30363d;}"
        )
        tree_btn_row.addWidget(self.btn_refresh)
        tree_btn_row.addWidget(self.btn_del_selected)
        tree_btn_row.addWidget(self.btn_clean_empty)
        mg_lay.addLayout(tree_btn_row)

        self.lbl_stats = QLabel("")
        self.lbl_stats.setStyleSheet("color:#c9d1d9;font-size:12px;")
        self.lbl_stats.setWordWrap(True)
        mg_lay.addWidget(self.lbl_stats)

        right_lay.addWidget(manage_group)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)

        self._set_ctrl_state(idle=True)

    def _connect_signals(self):
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_refresh.clicked.connect(self._refresh_tree)
        self.btn_del_selected.clicked.connect(self._on_delete_selected)
        self.btn_clean_empty.clicked.connect(self._on_clean_empty)

        self.sw.mel_received.connect(self._on_mel)
        self.log_signal.connect(self._append_log)
        self.progress_signal.connect(self._update_progress)
        self.finished_signal.connect(self._on_finished)

    def _on_mode_changed(self, idx):
        if idx == 0:
            self.exist_widget.show()
            self.new_widget.hide()
            self._refresh_class_combo()
        else:
            self.exist_widget.hide()
            self.new_widget.show()
            self._scan_base_commands()

    def _refresh_class_combo(self):
        self.combo_class.clear()
        if not os.path.isdir(ALLDATA_DIR):
            return
        classes = sorted(
            d for d in os.listdir(ALLDATA_DIR)
            if os.path.isdir(os.path.join(ALLDATA_DIR, d))
        )
        self.combo_class.addItems(classes)

    def _scan_base_commands(self):
        self.combo_base_cmd.clear()
        base_cmds = ['backward', 'forward', 'start', 'stop', 'turn', 'noise']
        self.combo_base_cmd.addItems(base_cmds)

    def _refresh_tree(self):
        """刷新数据文件树，显示各类别的数据文件"""
        self.tree.clear()
        self._refresh_class_combo()

        total_files = 0
        total_classes = 0
        total_size = 0

        if not os.path.isdir(ALLDATA_DIR):
            self.lbl_stats.setText("数据目录不存在")
            return

        for cls_name in sorted(os.listdir(ALLDATA_DIR)):
            cls_path = os.path.join(ALLDATA_DIR, cls_name)
            if not os.path.isdir(cls_path):
                continue
            npy_files = sorted(
                f for f in os.listdir(cls_path)
                if f.endswith(".npy") and os.path.isfile(os.path.join(cls_path, f))
            )
            if not npy_files:
                continue

            total_classes += 1
            total_files += len(npy_files)

            cls_item = QTreeWidgetItem(self.tree)
            cls_item.setText(0, f"{cls_name} ({len(npy_files)})")
            cls_item.setText(1, "类别")
            cls_item.setData(0, Qt.UserRole, cls_path)
            cls_item.setData(0, Qt.UserRole + 1, "folder")
            cls_item.setExpanded(True)

            for fname in npy_files:
                f_path = os.path.join(cls_path, fname)
                f_size = os.path.getsize(f_path)
                total_size += f_size
                f_item = QTreeWidgetItem(cls_item)
                f_item.setText(0, fname)
                if f_size >= 1024:
                    f_item.setText(1, f"{f_size / 1024:.1f} KB")
                else:
                    f_item.setText(1, f"{f_size} B")
                f_item.setData(0, Qt.UserRole, f_path)
                f_item.setData(0, Qt.UserRole + 1, "file")

        if total_size >= 1024 * 1024:
            size_str = f"{total_size / (1024 * 1024):.1f} MB"
        elif total_size >= 1024:
            size_str = f"{total_size / 1024:.1f} KB"
        else:
            size_str = f"{total_size} B"

        self.lbl_stats.setText(
            f"共 {total_classes} 个类别，{total_files} 个数据文件，总计 {size_str}"
        )
        self.tree.collapseAll()

    def _on_delete_selected(self):
        """删除选中的数据文件或类别文件夹"""
        items = self.tree.selectedItems()
        if not items:
            QMessageBox.information(self, "提示", "请先选择要删除的项目")
            return

        folders = []
        files = []
        for item in items:
            path = item.data(0, Qt.UserRole)
            kind = item.data(0, Qt.UserRole + 1)
            if not path or not os.path.exists(path):
                continue
            if kind == "folder":
                folders.append(path)
            else:
                files.append(path)

        targets = files + folders
        if not targets:
            return

        desc_parts = []
        if folders:
            desc_parts.append(f"{len(folders)} 个类别文件夹")
        if files:
            desc_parts.append(f"{len(files)} 个数据文件")
        desc = "、".join(desc_parts)

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除 {desc} 吗？\n此操作不可撤销！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        deleted = 0
        errors = 0
        for fpath in files:
            try:
                os.remove(fpath)
                deleted += 1
            except Exception as e:
                self.log_signal.emit(f"[错误] 删除文件失败: {fpath} - {e}")
                errors += 1

        for dpath in folders:
            try:
                shutil.rmtree(dpath)
                deleted += 1
            except Exception as e:
                self.log_signal.emit(f"[错误] 删除文件夹失败: {dpath} - {e}")
                errors += 1

        self.log_signal.emit(f"[完成] 已删除 {deleted} 项，失败 {errors} 项")
        self._refresh_tree()

    def _on_clean_empty(self):
        """清理没有npy文件的空类别文件夹"""
        if not os.path.isdir(ALLDATA_DIR):
            return
        removed = 0
        for cls_name in sorted(os.listdir(ALLDATA_DIR)):
            cls_path = os.path.join(ALLDATA_DIR, cls_name)
            if not os.path.isdir(cls_path):
                continue
            has_npy = any(
                f.endswith(".npy")
                for f in os.listdir(cls_path)
                if os.path.isfile(os.path.join(cls_path, f))
            )
            if not has_npy:
                try:
                    shutil.rmtree(cls_path)
                    removed += 1
                    self.log_signal.emit(f"[清理] 删除空文件夹: {cls_name}")
                except Exception as e:
                    self.log_signal.emit(f"[错误] 清理失败: {cls_path} - {e}")

        if removed > 0:
            self.log_signal.emit(f"[完成] 共清理 {removed} 个空文件夹")
        else:
            self.log_signal.emit("[信息] 没有空文件夹需要清理")
        self._refresh_tree()

    def _set_ctrl_state(self, idle=True):
        self.btn_start.setEnabled(idle)
        self.btn_pause.setEnabled(not idle)
        self.btn_stop.setEnabled(not idle)
        self.combo_mode.setEnabled(idle)
        self.combo_class.setEnabled(idle)
        self.edit_new_class.setEnabled(idle)
        self.spin_seg.setEnabled(idle)
        self.spin_samples.setEnabled(idle)

    def _on_mel(self, mel):
        if self._recording and not self._paused:
            with self._lock:
                self._mel_queue.append(list(mel))

    def _append_log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.append(f"[{timestamp}] {msg}")
        sb = self.log_view.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def _update_progress(self, current, total):
        if total > 0:
            pct = int(current * 100 / total)
            self.progress.setValue(pct)
            self.lbl_status.setText(f"录制中: {current}/{total}")
        else:
            self.progress.setValue(0)
            self.lbl_status.setText("就绪")

    def _on_finished(self, success, saved, skipped):
        self._recording = False
        self._paused = False
        self._set_ctrl_state(idle=True)
        if success:
            self.progress.setValue(100)
            self.lbl_status.setText(f"录制完成 (成功 {saved}, 跳过 {skipped})")
        else:
            self.progress.setValue(0)
            self.lbl_status.setText("录制中断")
        self._refresh_tree()

    def _on_start(self):
        """开始录制，创建录制线程"""
        if not self.sw.connected:
            QMessageBox.warning(self, "警告", "请先连接串口设备")
            return

        mode = self.combo_mode.currentIndex()
        if mode == 0:
            cls_name = self.combo_class.currentText().strip()
            if not cls_name:
                QMessageBox.warning(self, "警告", "请选择一个类别")
                return
        else:
            display_name = self.edit_new_class.text().strip()
            if not display_name:
                QMessageBox.warning(self, "警告", "请输入新类别名称")
                return
            invalid = set('/\\:*?"<>|')
            if any(c in invalid for c in display_name):
                QMessageBox.warning(self, "警告", "类别名包含非法字符")
                return
            base_cmd = self.combo_base_cmd.currentText().strip()
            if not base_cmd:
                QMessageBox.warning(self, "警告", "请选择基础命令")
                return
            cls_name = f"{display_name}({base_cmd})"

        seg_sec = self.spin_seg.value()
        num_samples = self.spin_samples.value()

        self._recording = True
        self._paused = False
        self._mel_queue.clear()
        self._session_files = []
        self._set_ctrl_state(idle=False)
        self.progress.setValue(0)
        self.log_view.clear()
        self.btn_pause.setText("暂停")

        self.log_signal.emit(f"准备录制类别: {cls_name}")
        self.log_signal.emit(f"参数: 段时长={seg_sec}s, 采样数={num_samples}")
        self.log_signal.emit(f"保存路径: alldata/{cls_name}/")

        self._record_thread = RecordThread(
            self.sw, cls_name, seg_sec, num_samples,
            self._lock, self._mel_queue,
        )
        self._record_thread.log.connect(self.log_signal.emit)
        self._record_thread.progress.connect(self.progress_signal.emit)
        self._record_thread.finished.connect(self._on_finished)
        self._record_thread.file_saved.connect(self._session_files.append)
        self._record_thread.start()

    def _on_pause(self):
        if not self._recording or self._record_thread is None:
            return
        if not self._paused:
            self._paused = True
            self._record_thread.pause()
            self.btn_pause.setText("继续")
            self.lbl_status.setText("已暂停")
            self.log_signal.emit("[暂停] 录制已暂停")
        else:
            self._paused = False
            self._record_thread.resume()
            self.btn_pause.setText("暂停")
            self.lbl_status.setText("录制中...")
            self.log_signal.emit("[继续] 录制继续")

    def _on_stop(self):
        if self._recording and self._record_thread is not None:
            if self._paused:
                reply = QMessageBox.question(
                    self, "停止录制",
                    "是否保留已录制的数据？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                self._record_thread.stop()
                if reply == QMessageBox.StandardButton.No:
                    for fpath in self._session_files:
                        try:
                            if os.path.isfile(fpath):
                                os.remove(fpath)
                        except Exception:
                            pass
                    self.log_signal.emit(f"[丢弃] 已删除 {len(self._session_files)} 个录制文件")
                    self._session_files = []
                else:
                    self.log_signal.emit(f"[保留] 已保留 {len(self._session_files)} 个录制文件")
            else:
                self._record_thread.stop()
            self.log_signal.emit("[停止] 正在停止录制...")
