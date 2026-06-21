/**
 * @file    hcsr04.h
 * @brief   HCSR04 超声波测距驱动接口
 * @author  北京林业大学 理学院 无敌小车车
 * @advisor 张立
 */

#ifndef HCSR04_H
#define HCSR04_H

#include "main.h"

// 声明获取距离的函数，返回单位为毫米 (mm)
float HCSR04_GetDistance(void);

#endif
