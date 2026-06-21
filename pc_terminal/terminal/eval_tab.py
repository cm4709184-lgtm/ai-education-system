# 模型评估结果展示模块，显示训练指标和混淆矩阵
import os, json

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QFormLayout, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")  # 项目根目录


def list_model_dirs(project_dir):
    """列出项目目录下所有模型子目录"""
    models = []
    if not os.path.isdir(project_dir):
        return models
    for name in os.listdir(project_dir):
        if name.startswith("model") and os.path.isdir(os.path.join(project_dir, name)):
            cfg = os.path.join(project_dir, name, "evaluation", "model_config.json")
            if os.path.isfile(cfg):
                try:
                    num = int(name[5:])
                    models.append((num, name))
                except ValueError:
                    pass
    models.sort(key=lambda x: x[0])
    return models


def scan_projects():
    """扫描projects目录，返回所有项目及其模型列表"""
    projects = []
    if not os.path.isdir(PROJECTS_DIR):
        return projects
    try:
        entries = sorted(os.listdir(PROJECTS_DIR))
    except OSError:
        return projects
    for name in entries:
        pdir = os.path.join(PROJECTS_DIR, name)
        if not os.path.isdir(pdir):
            continue
        models = list_model_dirs(pdir)
        if not models:
            continue
        projects.append({"name": name, "dir": pdir, "models": models})
    return projects


def load_model_cfg(project_dir, model_name):
    """加载模型的评估配置文件"""
    cfg_path = os.path.join(project_dir, model_name, "evaluation", "model_config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class EvalTab(QWidget):
    """展示训练模型的评估指标，包括准确率、混淆矩阵等"""
    def __init__(self):
        super().__init__()
        self._projects = []
        self._build_ui()
        self._refresh()

    def showEvent(self, event):
        super().showEvent(event)
        prev_project = self.project_combo.currentText()
        prev_model = self.model_combo.currentText()
        self._projects = scan_projects()
        self._rebuild_project_combo(prev_project, prev_model)

    def _rebuild_project_combo(self, select_project=None, select_model=None):
        self.project_combo.blockSignals(True)
        self.model_combo.blockSignals(True)
        self.project_combo.clear()
        for p in self._projects:
            self.project_combo.addItem(p["name"])
        target_idx = 0
        if select_project:
            for i, p in enumerate(self._projects):
                if p["name"] == select_project:
                    target_idx = i
                    break
        self.project_combo.setCurrentIndex(target_idx)
        self._fill_model_combo(target_idx, select_model)
        self.project_combo.blockSignals(False)
        self.model_combo.blockSignals(False)

    def _fill_model_combo(self, proj_idx, select_model=None):
        self.model_combo.clear()
        if proj_idx < 0 or proj_idx >= len(self._projects):
            return
        proj = self._projects[proj_idx]
        for num, mname in proj["models"]:
            self.model_combo.addItem(mname)
        target_idx = len(proj["models"]) - 1
        if select_model:
            for j, (_, mname) in enumerate(proj["models"]):
                if mname == select_model:
                    target_idx = j
                    break
        self.model_combo.setCurrentIndex(target_idx)
        self._load_and_display(proj_idx, target_idx)

    def _load_and_display(self, proj_idx, model_idx):
        """加载指定模型配置并显示评估结果"""
        if proj_idx < 0 or proj_idx >= len(self._projects):
            return
        proj = self._projects[proj_idx]
        if model_idx < 0 or model_idx >= len(proj["models"]):
            return
        _, model_name = proj["models"][model_idx]
        cfg = load_model_cfg(proj["dir"], model_name)
        if cfg is None:
            return
        self._display_cfg(cfg)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("项目:"))
        self.project_combo = QComboBox()
        self.project_combo.currentIndexChanged.connect(self._on_project_changed)
        top_row.addWidget(self.project_combo, stretch=1)
        top_row.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        top_row.addWidget(self.model_combo, stretch=1)
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.clicked.connect(self._refresh)
        top_row.addWidget(self.btn_refresh)
        layout.addLayout(top_row)

        splitter = QSplitter(Qt.Vertical)

        summary_group = QGroupBox("模型摘要")
        summary_lay = QFormLayout(summary_group)
        self.lbl_classes = QLabel("--")
        self.lbl_acc = QLabel("--")
        self.lbl_epochs = QLabel("--")
        self.lbl_lr = QLabel("--")
        self.lbl_bs = QLabel("--")
        summary_lay.addRow("类别数:", self.lbl_classes)
        summary_lay.addRow("总准确率:", self.lbl_acc)
        summary_lay.addRow("Epochs:", self.lbl_epochs)
        summary_lay.addRow("学习率:", self.lbl_lr)
        summary_lay.addRow("Batch Size:", self.lbl_bs)
        splitter.addWidget(summary_group)

        per_class_group = QGroupBox("各类别准确率")
        per_class_lay = QVBoxLayout(per_class_group)
        self.per_class_table = QTableWidget()
        self.per_class_table.setColumnCount(3)
        self.per_class_table.setHorizontalHeaderLabels(["类别", "显示名", "准确率"])
        self.per_class_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.per_class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.per_class_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.per_class_table.setEditTriggers(QTableWidget.NoEditTriggers)
        per_class_lay.addWidget(self.per_class_table)
        splitter.addWidget(per_class_group)

        cm_group = QGroupBox("混淆矩阵 (行=实际, 列=预测)")
        cm_lay = QVBoxLayout(cm_group)
        self.cm_table = QTableWidget()
        self.cm_table.setEditTriggers(QTableWidget.NoEditTriggers)
        cm_lay.addWidget(self.cm_table)
        splitter.addWidget(cm_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 3)
        layout.addWidget(splitter)

    def _refresh(self):
        self._projects = scan_projects()
        self._rebuild_project_combo()

    def select_project(self, project_name):
        self._projects = scan_projects()
        self._rebuild_project_combo(select_project=project_name)

    def _on_project_changed(self, idx):
        self._projects = scan_projects()
        self._rebuild_project_combo(select_project=self.project_combo.currentText())

    def _on_model_changed(self, idx):
        pidx = self.project_combo.currentIndex()
        self._load_and_display(pidx, idx)

    def _display_cfg(self, cfg):
        """解析配置并更新UI显示摘要、各类别准确率和混淆矩阵"""
        try:
            nc = cfg.get("num_classes", 0)
            class_names = cfg.get("class_names", [])
            class_display = cfg.get("class_display", class_names)
            best_acc = cfg.get("best_accuracy", 0.0)
            epochs = cfg.get("epochs", 0)
            lr = cfg.get("learning_rate", 0)
            bs = cfg.get("batch_size", 0)
            cm = cfg.get("confusion_matrix", [])
            per_class_acc = cfg.get("per_class_acc", [])

            self.lbl_classes.setText(str(nc))
            self.lbl_acc.setText(f"{best_acc*100:.1f}%")
            self.lbl_epochs.setText(str(epochs))
            self.lbl_lr.setText(str(lr))
            self.lbl_bs.setText(str(bs))

            self.per_class_table.setRowCount(0)
            for i in range(min(len(per_class_acc), nc)):
                row = self.per_class_table.rowCount()
                self.per_class_table.insertRow(row)
                self.per_class_table.setItem(row, 0, QTableWidgetItem(class_names[i] if i < len(class_names) else "?"))
                self.per_class_table.setItem(row, 1, QTableWidgetItem(class_display[i] if i < len(class_display) else "?"))
                acc_item = QTableWidgetItem(f"{per_class_acc[i]*100:.1f}%")
                if per_class_acc[i] >= 0.9:
                    acc_item.setForeground(QColor("#3fb950"))
                elif per_class_acc[i] >= 0.7:
                    acc_item.setForeground(QColor("#d29922"))
                else:
                    acc_item.setForeground(QColor("#f85149"))
                self.per_class_table.setItem(row, 2, acc_item)

            self.cm_table.setRowCount(0)
            self.cm_table.setColumnCount(0)
            if cm and nc > 0:
                self.cm_table.setColumnCount(nc)
                self.cm_table.setRowCount(nc)
                col_labels = [class_display[i] if i < len(class_display) else "?" for i in range(nc)]
                row_labels = [class_display[i] if i < len(class_display) else "?" for i in range(nc)]
                self.cm_table.setHorizontalHeaderLabels(col_labels)
                self.cm_table.setVerticalHeaderLabels(row_labels)
                for i in range(min(len(cm), nc)):
                    for j in range(min(len(cm[i]), nc)):
                        val = cm[i][j]
                        item = QTableWidgetItem(str(val))
                        item.setTextAlignment(Qt.AlignCenter)
                        if i == j and val > 0:
                            item.setBackground(QColor("#238636"))
                        elif val > 0:
                            item.setBackground(QColor("#da3633"))
                        self.cm_table.setItem(i, j, item)
                for col in range(nc):
                    self.cm_table.setColumnWidth(col, 60)
        except Exception as e:
            self.lbl_classes.setText("错误")
            self.lbl_acc.setText(str(e)[:50])
            self.lbl_epochs.setText("--")
            self.lbl_lr.setText("--")
            self.lbl_bs.setText("--")
            self.per_class_table.setRowCount(0)
            self.cm_table.setRowCount(0)
            self.cm_table.setColumnCount(0)
