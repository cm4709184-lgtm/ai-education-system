/**
 * @file    OLED.c
 * @brief   SSD1306 OLED 驱动 (I2C, 128x64)
 * @author  北京林业大学 理学院 无敌小车车
 * @advisor 张立
 */
#include "OLED.h" // 请确保该头文件已包含必要的函数声明，或直接将声明放在这里
#include "OLED_Font.h"
#include "main.h"   // 包含 GPIO 定义
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
/***************************************************************************************
 * 程序说明：
 *   适用于：STM32F103 + 0.96寸OLED(SSD1306) + HAL库
 *   接线：PB6 -> SCL, PB7 -> SDA
 ***************************************************************************************/

/* 用户配置区：引脚定义 */
#define OLED_SCL_PORT GPIOE
#define OLED_SCL_PIN  GPIO_PIN_11

#define OLED_SDA_PORT GPIOE
#define OLED_SDA_PIN  GPIO_PIN_12
/***************************************************************************************/

/* 全局变量 */
// OLED显存数组 (8页 x 128列)
uint8_t OLED_DisplayBuf[8][128];

/* 内部函数声明 */
static void OLED_W_SCL(uint8_t BitValue);
static void OLED_W_SDA(uint8_t BitValue);
static void OLED_GPIO_Init(void);
static void OLED_I2C_Delay(void);
static void OLED_I2C_Start(void);
static void OLED_I2C_Stop(void);
static void OLED_I2C_SendByte(uint8_t Byte);
static void OLED_WriteCommand(uint8_t Command);
static void OLED_WriteData(uint8_t *Data, uint8_t Count);

/**
  * 函 数：OLED写SCL高低电平
  */
static void OLED_W_SCL(uint8_t BitValue)
{
    HAL_GPIO_WritePin(OLED_SCL_PORT, OLED_SCL_PIN, (GPIO_PinState)BitValue);
    OLED_I2C_Delay();
}

/**
  * 函 数：OLED写SDA高低电平
  */
static void OLED_W_SDA(uint8_t BitValue)
{
    HAL_GPIO_WritePin(OLED_SDA_PORT, OLED_SDA_PIN, (GPIO_PinState)BitValue);
    OLED_I2C_Delay();
}

/**
  * 函 数：模拟I2C简单延时
  *         根据主频调整，F103默认72MHz时，此循环约为几微秒
  */
static void OLED_I2C_Delay(void)
{
	__NOP();
}

/**
  * 函 数：OLED引脚初始化
  */
static void OLED_GPIO_Init(void)
{

    OLED_W_SCL(1);
    OLED_W_SDA(1);
}

/**
  * 函 数：I2C起始信号
  */
static void OLED_I2C_Start(void)
{
    OLED_W_SDA(1); // 拉高数据线
    OLED_W_SCL(1); // 拉高时钟线
    OLED_W_SDA(0); // 数据线由高变低，此时时钟线为高 -> 起始信号
    OLED_W_SCL(0); // 拉低时钟线，准备发送数据
}

/**
  * 函 数：I2C停止信号
  */
static void OLED_I2C_Stop(void)
{
    OLED_W_SDA(0); // 拉低数据线
    OLED_W_SCL(1); // 拉高时钟线
    OLED_W_SDA(1); // 数据线由低变高，此时时钟线为高 -> 停止信号
}

/**
  * 函 数：I2C发送一个字节
  */
static void OLED_I2C_SendByte(uint8_t Byte)
{
    uint8_t i;
    for (i = 0; i < 8; i++)
    {
        // 按位取出数据，高位在前
        if (Byte & (0x80 >> i))
            OLED_W_SDA(1);
        else
            OLED_W_SDA(0);
        OLED_W_SCL(1); // 拉高时钟，OLED读取数据
        OLED_W_SCL(0); // 拉低时钟，准备下一位
    }
    // 释放应答时钟 (SSD1306通常不需要严格处理应答，简单释放即可)
    OLED_W_SCL(1);
    OLED_W_SCL(0);
}

/**
  * 函 数：OLED写命令
  */
static void OLED_WriteCommand(uint8_t Command)
{
    OLED_I2C_Start();
    OLED_I2C_SendByte(0x78);    // OLED I2C从机地址 (7位地址0x3C左移1位)
    OLED_I2C_SendByte(0x00);    // 控制字节：Co=0, D/C#=0 (写命令)
    OLED_I2C_SendByte(Command); // 发送命令
    OLED_I2C_Stop();
}

/**
  * 函 数：OLED写数据
  */
static void OLED_WriteData(uint8_t *Data, uint8_t Count)
{
    uint8_t i;
    OLED_I2C_Start();
    OLED_I2C_SendByte(0x78);    // OLED I2C从机地址
    OLED_I2C_SendByte(0x40);    // 控制字节：Co=0, D/C#=1 (写数据)
    for (i = 0; i < Count; i++)
    {
        OLED_I2C_SendByte(Data[i]);
    }
    OLED_I2C_Stop();
}

/**
  * 函 数：OLED初始化
  */
void OLED_Init(void)
{
    // 1. 初始化GPIO
    OLED_GPIO_Init();

    // 2. 硬件复位 (可选，如果没有接RST引脚可注释此部分)
    // HAL_GPIO_WritePin(OLED_RST_PORT, OLED_RST_PIN, GPIO_PIN_RESET);
    // HAL_Delay(100);
    // HAL_GPIO_WritePin(OLED_RST_PORT, OLED_RST_PIN, GPIO_PIN_SET);

    // 3. 发送初始化配置命令
    OLED_WriteCommand(0xAE); // 关显示

    OLED_WriteCommand(0xD5); // 设置时钟分频
    OLED_WriteCommand(0x80); // 分频因子

    OLED_WriteCommand(0xA8); // 设置驱动路数
    OLED_WriteCommand(0x3F); // 1/64 Duty (对应64行)

    OLED_WriteCommand(0xD3); // 设置显示偏移
    OLED_WriteCommand(0x00); // 0偏移

    OLED_WriteCommand(0x40); // 设置显示开始行

    OLED_WriteCommand(0x8D); // 充电泵设置
    OLED_WriteCommand(0x14); // 开充电泵

    OLED_WriteCommand(0x20); // 内存地址模式
    OLED_WriteCommand(0x02); // 行地址模式

    OLED_WriteCommand(0xA1); // 段重映射 (左右翻转，根据屏幕实际方向可改为A0)
    OLED_WriteCommand(0xC8); // COM扫描方向 (上下翻转)

    OLED_WriteCommand(0xDA); // 设置COM引脚硬件配置
    OLED_WriteCommand(0x12); // Alternative COM Pin config

    OLED_WriteCommand(0x81); // 对比度控制
    OLED_WriteCommand(0xCF); // 设置对比度值

    OLED_WriteCommand(0xD9); // 预充电周期
    OLED_WriteCommand(0xF1);

    OLED_WriteCommand(0xDB); // VCOMH 取消选择级别
    OLED_WriteCommand(0x30); // 0.83 * Vcc

    OLED_WriteCommand(0xA4); // 全局显示开启
    OLED_WriteCommand(0xA6); // 正常显示 (非反色)

    OLED_WriteCommand(0xAF); // 开显示

    // 4. 清屏
    OLED_Clear();
    OLED_Update();
}

/**
  * 函 数：OLED设置光标位置
  */
void OLED_SetCursor(uint8_t Page, uint8_t X)
{
    OLED_WriteCommand(0xB0 | Page); // 设置页地址
    OLED_WriteCommand(0x10 | ((X >> 4) & 0x0F)); // 设置列地址高4位
    OLED_WriteCommand(0x00 | (X & 0x0F));        // 设置列地址低4位
}

/**
  * 函 数：OLED更新显存到屏幕 (全屏)
  */
void OLED_Update(void)
{
    uint8_t i;
    for (i = 0; i < 8; i++)
    {
        OLED_SetCursor(i, 0);
        OLED_WriteData(OLED_DisplayBuf[i], 128);
    }
}

/**
  * 函 数：OLED清屏 (仅清空显存)
  */
void OLED_Clear(void)
{
    memset(OLED_DisplayBuf, 0, sizeof(OLED_DisplayBuf));
}

// =================================================================
// 以下是显示功能函数 (ShowChar, ShowString, Printf等)
// 为了篇幅，此处展示核心逻辑，通常直接复用即可
// =================================================================

/**
  * 函 数：OLED显示一个字符
  */
void OLED_DrawPoint(int16_t X, int16_t Y)
{
    if (X >= 0 && X < 128 && Y >= 0 && Y < 64)
    {
        OLED_DisplayBuf[Y/8][X] |= (1 << (Y%8));
    }
}
void OLED_ShowChar(int16_t X, int16_t Y, char Char, uint8_t FontSize)
{
    uint8_t i, j;
    const uint8_t *font;

    if (Char < ' ' || Char > '~') return;

    if (FontSize == OLED_8X16)
    {
        font = OLED_F8x16[Char - ' ']; // 16 bytes per char
        // 8x16: 每行1字节，共16行
        for (j = 0; j < 16; j++)
        {
            for (i = 0; i < 8; i++)
            {
                if (font[j] & (0x80 >> i)) // 从高位到低位（左到右）
                {
                    OLED_DrawPoint(X + i, Y + j);
                }
            }
        }
    }
    else // OLED_6X8
    {
        const uint8_t *font = OLED_F6x8[Char - ' '];
        for (uint8_t col = 0; col < 6; col++) {
            uint8_t data = font[col];
            for (uint8_t row = 0; row < 8; row++) {
                // 关键：检查 bit 'row' (bit0 = top, bit7 = bottom)
                if (data & (1 << row)) {
                    OLED_DrawPoint(X + col, Y + row);
                }
            }
        }
    }
  }

/**
  * 函 数：OLED画点 (显存操作)
  */


// 其他函数 (OLED_ShowString, OLED_Printf, OLED_ShowNum 等)
// 逻辑与上一轮对话中的 HAL 版本一致，调用 OLED_ShowChar 即可，此处省略以节省篇幅
// ==================== 1. 显示字符串 ====================
void OLED_ShowString(int16_t X, int16_t Y, const char *str, uint8_t FontSize)
{
    uint8_t width = (FontSize == OLED_8X16) ? 8 : 6;
    while (*str)
    {
        if (X >= 128) break; // 超出屏幕宽度
        OLED_ShowChar(X, Y, *str, FontSize);
        X += width;
        str++;
    }
}

// ==================== 2. 格式化打印（简化版）====================
void OLED_Printf(int16_t X, int16_t Y, uint8_t FontSize, const char *format, ...)
{
    char buffer[64];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    OLED_ShowString(X, Y, buffer, FontSize);
}

// ==================== 3. 显示整数 ====================
void OLED_ShowNum(int16_t X, int16_t Y, int32_t num, uint8_t len, uint8_t FontSize)
{
    char buf[12];
    if (num < 0) {
        OLED_ShowChar(X, Y, '-', FontSize);
        X += (FontSize == OLED_8X16) ? 8 : 6;
        num = -num;
    }
    snprintf(buf, sizeof(buf), "%ld", (long)num);
    // 补前导零
    if (len > strlen(buf)) {
        char temp[12] = {0};
        memset(temp, '0', len - strlen(buf));
        strcat(temp, buf);
        strcpy(buf, temp);
    }
    OLED_ShowString(X, Y, buf, FontSize);
}

// ==================== 4. 显示浮点数 ====================
void OLED_ShowFloat(int16_t X, int16_t Y, float num, uint8_t decimal, uint8_t FontSize)
{
    if (decimal > 5) decimal = 5; // 限制小数位

    // 处理负数
    if (num < 0) {
        OLED_ShowChar(X, Y, '-', FontSize);
        X += (FontSize == OLED_8X16) ? 8 : 6;
        num = -num;
    }

    // 分离整数和小数部分
    uint32_t integer = (uint32_t)num;
    float frac = num - integer;

    // 缩放小数部分
    uint32_t frac_part = 0;
    float scale = 1.0f;
    for (uint8_t i = 0; i < decimal; i++) scale *= 10.0f;
    frac_part = (uint32_t)(frac * scale + 0.5f); // 四舍五入

    // 防止进位溢出（如 0.99999 → 1.00000）
    if (frac_part >= (uint32_t)scale) {
        integer++;
        frac_part = 0;
    }

    // 显示整数部分
    char buf[12];
    snprintf(buf, sizeof(buf), "%lu", (unsigned long)integer);
    OLED_ShowString(X, Y, buf, FontSize);
    X += strlen(buf) * ((FontSize == OLED_8X16) ? 8 : 6);

    // 显示小数点
    if (decimal > 0) {
        OLED_ShowChar(X, Y, '.', FontSize);
        X += (FontSize == OLED_8X16) ? 8 : 6;

        // 补前导零并显示小数
        snprintf(buf, sizeof(buf), "%lu", (unsigned long)frac_part);
        uint8_t len = strlen(buf);
        while (len < decimal) {
            OLED_ShowChar(X, Y, '0', FontSize);
            X += (FontSize == OLED_8X16) ? 8 : 6;
            len++;
        }
        OLED_ShowString(X, Y, buf, FontSize);
    }
}




// ==================== 5. 画线（Bresenham算法）====================




// 修复版：带坐标裁剪的画线
void OLED_DrawLine(int16_t x0, int16_t y0, int16_t x1, int16_t y1)
{
    // 简单裁剪：完全在屏幕外则返回
    if (x0 < 0 && x1 < 0) return;
    if (x0 >= 128 && x1 >= 128) return;
    if (y0 < 0 && y1 < 0) return;
    if (y0 >= 64 && y1 >= 64) return;

    int16_t dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int16_t dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int16_t err = dx + dy, e2;

    for (;;) {
        OLED_DrawPoint(x0, y0);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// 修复版：带边界检查的矩形


// ==================== 6. 画矩形 ====================

void OLED_DrawRect(int16_t x, int16_t y, int16_t w, int16_t h)
{
    if (w <= 0 || h <= 0) return;
    // 裁剪到屏幕内（简化版）
    int16_t x1 = x + w - 1;
    int16_t y1 = y + h - 1;
    if (x >= 128 || y >= 64 || x1 < 0 || y1 < 0) return;

    OLED_DrawLine(x, y, x1, y);
    OLED_DrawLine(x, y1, x1, y1);
    OLED_DrawLine(x, y, x, y1);
    OLED_DrawLine(x1, y, x1, y1);
}

// ==================== 7. 填充矩形 ====================
void OLED_FillRect(int16_t x, int16_t y, int16_t w, int16_t h)
{
    if (w <= 0 || h <= 0) return;
    for (int16_t dy = 0; dy < h; dy++)
        for (int16_t dx = 0; dx < w; dx++)
            OLED_DrawPoint(x + dx, y + dy);
}

// ==================== 8. 反显指定区域 ====================
void OLED_ReverseArea(uint8_t page_start, uint8_t page_end, uint8_t col_start, uint8_t col_end)
{
    for (uint8_t page = page_start; page <= page_end; page++) {
        for (uint8_t col = col_start; col <= col_end; col++) {
            OLED_DisplayBuf[page][col] = ~OLED_DisplayBuf[page][col];
        }
    }
    OLED_Update(); // 立即刷新
}

// ==================== 音频播放状态显示 ====================
void OLED_ShowAudioPlaying(const char* track_name)
{
    OLED_ReverseArea(0, 0, 0, 127);
    OLED_Printf(0, 0, OLED_6X8, ">>%s", track_name);
    OLED_Update();
}

void OLED_ClearAudioStatus(void)
{
    OLED_DisplayBuf[0][0] = 0;
    OLED_DisplayBuf[0][1] = 0;
    for (int col = 0; col < 128; col++)
        OLED_DisplayBuf[0][col] = 0;
    OLED_Update();
}
