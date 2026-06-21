# 设置标签页：串口配置、工作路径和版本信息展示
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QPushButton, QLabel, QFormLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from terminal.serial_worker import list_serial_ports


APP_NAME = '智能语音小车 · 控制终端'  # 应用名称
APP_VERSION = 'v1.0.0'  # 当前版本号

PATHS = {  # 关键工作目录路径配置
    '数据集目录': r'd:\15\ai_training\alldataset',
    '训练输出目录': r'd:\15\ai_training\runs',
    'Core/Inc 目录': r'd:\15\Core\Inc',
}

BAUD_RATES = ['921600', '460800', '115200']  # 可选波特率列表


class SettingsTab(QWidget):
    """设置页面：串口参数配置、工作路径查看和版本信息展示"""
    def __init__(self, serial_worker):
        super().__init__()
        self.sw = serial_worker
        self._build_ui()

    def _build_ui(self):
        """构建设置页面：串口设置、工作路径、版本信息三个分组"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        serial_group = QGroupBox('串口设置')
        serial_form = QFormLayout(serial_group)
        serial_form.setSpacing(10)

        port_row = QHBoxLayout()
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.setMinimumWidth(200)
        self._refresh_ports()
        port_row.addWidget(self.port_combo)
        self.btn_refresh = QPushButton('扫描端口')
        self.btn_refresh.setFixedWidth(90)
        self.btn_refresh.clicked.connect(self._refresh_ports)
        port_row.addWidget(self.btn_refresh)
        port_row.addStretch()
        serial_form.addRow('串口:', port_row)

        self.baud_combo = QComboBox()
        self.baud_combo.addItems(BAUD_RATES)
        self.baud_combo.setCurrentText('921600')
        serial_form.addRow('波特率:', self.baud_combo)

        layout.addWidget(serial_group)

        path_group = QGroupBox('工作路径')
        path_form = QFormLayout(path_group)
        path_form.setSpacing(8)
        for label, path in PATHS.items():
            lbl = QLabel(path)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lbl.setStyleSheet('color: #8b949e; font-family: Consolas, monospace;')
            lbl.setToolTip(path)
            path_form.addRow(f'{label}:', lbl)
        layout.addWidget(path_group)

        version_group = QGroupBox('版本信息')
        version_form = QFormLayout(version_group)
        version_form.setSpacing(8)
        lbl_name = QLabel(APP_NAME)
        lbl_name.setFont(QFont('Microsoft YaHei', 10, QFont.Bold))
        version_form.addRow('应用名称:', lbl_name)
        version_form.addRow('版本号:', QLabel(APP_VERSION))
        layout.addWidget(version_group)

        layout.addStretch()

    def _refresh_ports(self):
        """扫描系统串口并更新下拉列表，优先选中COM口"""
        self.port_combo.clear()
        ports = list_serial_ports()
        self.port_combo.addItems(ports)
        if ports:
            for p in ports:
                if 'COM' in p:
                    self.port_combo.setCurrentText(p)
                    break
