/**
 * @file    OLED.h
 * @brief   SSD1306 OLED 驱动接口
 * @author  北京林业大学 理学院 无敌小车车
 * @advisor 张立
 */

#ifndef __OLED_H
#define __OLED_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32f4xx_hal.h"
#include <stdint.h>

/* 函数接口声明 */
void OLED_Init(void);
void OLED_Clear(void);
void OLED_Update(void);
void OLED_SetCursor(uint8_t Page, uint8_t X);
void OLED_DrawPoint(int16_t X, int16_t Y);
void OLED_ShowString(int16_t X, int16_t Y, const char *str, uint8_t FontSize);
void OLED_Printf(int16_t X, int16_t Y, uint8_t FontSize, const char *format, ...);
void OLED_ShowNum(int16_t X, int16_t Y, int32_t num, uint8_t len, uint8_t FontSize);
void OLED_ShowFloat(int16_t X, int16_t Y, float num, uint8_t decimal, uint8_t FontSize);
void OLED_DrawLine(int16_t x0, int16_t y0, int16_t x1, int16_t y1);
void OLED_DrawRect(int16_t x, int16_t y, int16_t w, int16_t h);
void OLED_FillRect(int16_t x, int16_t y, int16_t w, int16_t h);
void OLED_ReverseArea(uint8_t page_start, uint8_t page_end, uint8_t col_start, uint8_t col_end);
// 显示字符/字符串
void OLED_ShowChar(int16_t X, int16_t Y, char Char, uint8_t FontSize);
void OLED_ShowString(int16_t X, int16_t Y, const char *str, uint8_t FontSize);

// 格式化输出 (类似printf)
void OLED_Printf(int16_t X, int16_t Y, uint8_t FontSize, const char *format, ...);

// 音频播放状态显示
void OLED_ShowAudioPlaying(const char* track_name);
void OLED_ClearAudioStatus(void);

/* 字体大小宏定义 */
#define OLED_6X8  6
#define OLED_8X16 16

#ifdef __cplusplus
}
#endif
#endif

