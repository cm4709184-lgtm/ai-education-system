#!/usr/bin/env python3
# 应用主入口：主窗口框架，集成串口连接、多标签页导航和状态栏
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QComboBox, QPushButton,
    QLabel, QStatusBar, QToolBar, QWidget, QHBoxLayout
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QFont

from terminal.serial_worker import SerialWorker, list_serial_ports
from terminal.monitor_tab import MonitorTab
from terminal.collect_tab import CollectTab
from terminal.model_tab import ModelTab
from terminal.eval_tab import EvalTab
from terminal.data_eval_tab import DataEvalTab
from terminal.settings_tab import SettingsTab

APP_STYLE = """  # 全局深色主题样式表
QMainWindow { background: #0d1117; }
QTabWidget::pane { border: 1px solid #30363d; background: #0d1117; }
QTabBar::tab {
    background: #161b22; color: #8b949e; padding: 8px 16px;
    border: 1px solid #30363d; border-bottom: none; border-radius: 4px 4px 0 0;
    margin-right: 2px;
}
QTabBar::tab:selected { background: #0d1117; color: #58a6ff; border-bottom: 2px solid #58a6ff; }
QTabBar::tab:hover { background: #21262d; }
QGroupBox {
    color: #58a6ff; border: 1px solid #30363d; border-radius: 6px;
    margin-top: 8px; padding-top: 16px; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLabel { color: #c9d1d9; }
QPushButton {
    background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
    border-radius: 6px; padding: 6px 16px;
}
QPushButton:hover { background: #30363d; }
QPushButton:pressed { background: #161b22; }
QComboBox {
    background: #161b22; color: #c9d1d9; border: 1px solid #30363d;
    border-radius: 4px; padding: 4px 8px;
}
QStatusBar { background: #161b22; color: #c9d1d9; border-top: 1px solid #30363d; }
QToolBar { background: #161b22; border-bottom: 1px solid #30363d; spacing: 8px; padding: 4px; }
"""

class MainWindow(QMainWindow):
    """主窗口：集成串口连接管理、多标签页和状态栏"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle('智能语音小车 · 控制终端')
        self.setMinimumSize(1000, 700)
        self.sw = SerialWorker()
        self._build_toolbar()
        self._build_tabs()
        self._build_statusbar()
        self._refresh_ports()
        self.sw.connection_changed.connect(self._on_conn)
        self.sw.log_received.connect(self._on_log)
        self.sw.model_info_received.connect(self._on_model_info)
        self._model_refresh_timer = QTimer(self)
        self._model_refresh_timer.timeout.connect(self._auto_refresh_model)
        self._model_refresh_timer.start(10000)

    def _build_toolbar(self):
        """构建顶部工具栏：串口选择、连接按钮、模型信息显示"""
        tb = QToolBar('连接')
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addWidget(QLabel('  串口: '))
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.setMinimumWidth(120)
        tb.addWidget(self.port_combo)

        self.btn_refresh = QPushButton('🔄')
        self.btn_refresh.setMinimumWidth(36)
        self.btn_refresh.clicked.connect(self._refresh_ports)
        tb.addWidget(self.btn_refresh)

        self.btn_conn = QPushButton('连接')
        self.btn_conn.clicked.connect(self._toggle_conn)
        tb.addWidget(self.btn_conn)

        tb.addSeparator()
        self.lbl_model = QLabel('  模型: --')
        self.lbl_model.setStyleSheet('color: #58a6ff; font-weight: bold;')
        self.lbl_model.setToolTip('模型: --')
        self.lbl_model.setMaximumWidth(400)
        tb.addWidget(self.lbl_model)

        self.btn_query = QPushButton('查询状态')
        self.btn_query.clicked.connect(self.sw.query_status)
        tb.addWidget(self.btn_query)

    def _build_tabs(self):
        """构建标签页：实时监控、数据采集、训练&模型、模型测评、设置"""
        self.tabs = QTabWidget()
        self.tabs.addTab(MonitorTab(self.sw), '📊 实时监控')
        self.tabs.addTab(CollectTab(self.sw), '🎤 数据采集')
        model_tab = ModelTab(self.sw)
        eval_tab = EvalTab()
        model_tab.project_selected.connect(eval_tab.select_project)
        self.tabs.addTab(model_tab, '🧠 训练 & 模型')
        self.tabs.addTab(eval_tab, '📈 模型测评')
        self.tabs.addTab(DataEvalTab(self.sw), '🔬 数据测评')
        self.tabs.addTab(SettingsTab(self.sw), '⚙ 设置')
        self.setCentralWidget(self.tabs)

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lbl_conn = QLabel('未连接')
        self.lbl_baud = QLabel('921600')
        self.lbl_sys = QLabel('')
        sb.addWidget(self.lbl_conn)
        sb.addWidget(QLabel('  │  '))
        sb.addWidget(self.lbl_baud)
        sb.addWidget(QLabel('  │  '))
        sb.addWidget(self.lbl_sys)

    def _refresh_ports(self):
        self.port_combo.clear()
        ports = list_serial_ports()
        self.port_combo.addItems(ports)
        if ports:
            for p in ports:
                if 'COM' in p:
                    self.port_combo.setCurrentText(p)
                    break

    def _toggle_conn(self):
        """切换串口连接状态"""
        if self.sw.connected:
            self.sw.disconnect_port()
        else:
            port = self.port_combo.currentText().strip()
            if port:
                self.sw.connect_port(port)

    def _on_conn(self, ok, info):
        if ok:
            self.btn_conn.setText('断开')
            self.lbl_conn.setText(f'已连接 {info}')
            self.lbl_conn.setStyleSheet('color: #4CAF50;')
            self.sw.query_status()
            QTimer.singleShot(500, self.sw.query_model_info)
        else:
            self.btn_conn.setText('连接')
            self.lbl_conn.setText('未连接')
            self.lbl_conn.setStyleSheet('color: #f44336;')

    def _on_model_info(self, src, nc, names, maps):
        """收到模型信息后更新工具栏显示"""
        src_name = {0: '编译基座', 1: 'Flash'}.get(src, '?')
        text = f'  模型: {src_name} ({nc}类) [{", ".join(names)}]'
        self.lbl_model.setText(text)
        self.lbl_model.setToolTip(text.strip())

    def _on_log(self, msg):
        self.lbl_sys.setText(msg[:77] + '...' if len(msg) > 80 else msg)

    def _auto_refresh_model(self):
        """定时自动刷新模型信息"""
        if self.sw.connected:
            self.sw.query_model_info()

    def closeEvent(self, event):
        self.sw.disconnect_port()
        event.accept()

def main():
    """应用入口：初始化QApplication并显示主窗口"""
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    app.setFont(QFont('Microsoft YaHei', 9))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
