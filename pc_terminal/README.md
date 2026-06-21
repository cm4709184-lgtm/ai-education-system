# PC 端训练与监控终端

## 文件说明

| 文件 | 作用 |
|------|------|
| `terminal/main.py` | 程序入口（PyQt5） |
| `terminal/monitor_tab.py` | 实时监控 tab |
| `terminal/collect_tab.py` | 数据采集 tab |
| `terminal/model_tab.py` | 模型训练 tab |
| `terminal/eval_tab.py` | 模型测评 tab |
| `terminal/data_eval_tab.py` | 数据集测评 tab |
| `terminal/settings_tab.py` | 推理设置 tab |
| `terminal/serial_worker.py` | 串口通信子线程 |
| `mel_iterate.py` | Mel 频谱提取 |
| `regenerate_model_weights_h.py` | PyTorch 模型 → C 头文件 |
| `启动终端.bat` | Windows 一键启动 |
| `requirements.txt` | Python 依赖 |

## 运行

```bash
pip install -r requirements.txt
python terminal/main.py
```

## 数据目录（git 忽略）

- `projects/`：训练数据集项目
- `alldata/`：累积训练样本
