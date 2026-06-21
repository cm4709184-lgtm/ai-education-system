# MAX9814 AGC 麦克风模块 · Arduino 采集程序

## 接线

| MAX9814 | Arduino |
|---------|---------|
| VCC     | 5V      |
| GND     | GND     |
| OUT     | A0      |
| GAIN    | 悬空 (40dB) |
| AR      | 悬空 |

## 采样参数
- 采样率 16 kHz
- 单声道 16-bit
- 串口 921600 bps

## 用途
PC 端采集训练数据。固件端用板载 ADC 直接采样，不使用此代码。
