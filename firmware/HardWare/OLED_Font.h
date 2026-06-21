/**
 * @file    OLED_Font.h
 * @brief   OLED 字库声明
 * @author  北京林业大学 理学院 无敌小车车
 * @advisor 张立
 */

#ifndef __OLED_FONT_H
#define __OLED_FONT_H

#include "stm32f4xx_hal.h"

// 定义字体大小常量
#define OLED_6X8  6
#define OLED_8X16 16

// 正确：声明为二维数组，第二维的大小必须与定义文件一致
extern const uint8_t OLED_F6x8[][6];
extern const uint8_t OLED_F8x16[][16];

// 汉字字模数据 (示例：可扩展)
// extern const typFNT_GB16 gImage_示例[];

#endif
