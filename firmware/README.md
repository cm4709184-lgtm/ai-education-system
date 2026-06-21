# STM32F407 端固件（STM32CubeIDE 工程）

## 工程结构

```
firmware/
├── Core/Inc/, Core/Src/, Core/Startup/
├── Drivers/          # HAL + CMSIS
├── Middlewares/      # FreeRTOS
├── HardWare/         # 板级驱动
├── USB_HOST/
├── 15.ioc
├── STM32F407VGTX_FLASH.ld
└── STM32F407VGTX_RAM.ld
```

## 双模型架构

| 区域 | 地址 | 大小 |
|------|------|------|
| 编译基座 | `0x08000000` | 896 KB |
| 用户数据 | `0x080C0000` | 64 KB |
| Flash 模型 | `0x080E0000` | 128 KB（含 40 字节 Header） |

## 编译

1. STM32CubeIDE 导入 `15.ioc`
2. Build → ST-Link Debug 烧录基座到 `0x08000000`

## Flash 模型烧录

详见根目录 [README.md](../README.md) 第三节。
