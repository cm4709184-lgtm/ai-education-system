# 如何补全 Drivers/ 和 Middlewares/

本仓库**不包含** STM32 HAL 库和 FreeRTOS 源码。需要在本地用 STM32CubeIDE 重新生成。

## 步骤

### 1. 安装 STM32CubeIDE
下载 [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html)（1.13.0 或更新版本）

### 2. 导入工程
1. STM32CubeIDE → File → Open Projects from File System
2. 导入目录：选择 `firmware/` 目录
3. STM32CubeIDE 会自动识别 `15.ioc` 并生成 Drivers/ 和 Middlewares/

### 3. 验证
- 编译（Ctrl+B）应无错误
- Drivers/STM32F4xx_HAL_Driver/ 和 Middlewares/Third_Party/FreeRTOS/Source/ 自动出现

## 为什么不上传 HAL/FreeRTOS？

- 体积：HAL 库 ~300 文件，FreeRTOS ~30 文件，远超精简仓库大小
- 维护：ST 官方持续更新，git submodule 跟踪更稳定
- 标准做法：STM32 生态默认通过 .ioc 工程文件自动生成

## 所需软件包（STM32CubeIDE 首次打开 .ioc 时提示安装）

- STM32CubeF4 Firmware Package V1.27.1 或更新
- FreeRTOS（包含在 STM32CubeF4 中）
