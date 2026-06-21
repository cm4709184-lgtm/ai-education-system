/**
 * @file    hcsr04.c
 * @brief   HCSR04 超声波测距驱动
 * @author  北京林业大学 理学院 无敌小车车
 * @advisor 张立
 */

#include "hcsr04.h"
#include "stm32f4xx_hal.h"

#define TRIG_PIN        GPIO_PIN_4
#define TRIG_PORT       GPIOB
#define ECHO_PIN        GPIO_PIN_5
#define ECHO_PORT       GPIOB

// DWT 延时初始化
static void DWT_Init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static void Delay_us(uint16_t us)
{
    static uint8_t initialized = 0;
    if (!initialized) {
        DWT_Init();
        initialized = 1;
    }
    uint32_t start = DWT->CYCCNT;
    uint32_t ticks = us * (SystemCoreClock / 1000000);
    while ((DWT->CYCCNT - start) < ticks);
}

float HCSR04_GetDistance(void)
{
    uint32_t start_time, end_time;
    uint32_t timeout;

    // 1. 发送 12us 触发脉冲
    HAL_GPIO_WritePin(TRIG_PORT, TRIG_PIN, GPIO_PIN_SET);
    Delay_us(12);
    HAL_GPIO_WritePin(TRIG_PORT, TRIG_PIN, GPIO_PIN_RESET);

    // 2. 等待 Echo 变为高电平（超时 50ms）
    timeout = 50000;
    while (HAL_GPIO_ReadPin(ECHO_PORT, ECHO_PIN) == GPIO_PIN_RESET && --timeout);
    if (timeout == 0) return -1.0f;

    // 3. 记录高电平开始时刻
    start_time = DWT->CYCCNT;

    // 4. 等待 Echo 变为低电平
    timeout = 50000;
    while (HAL_GPIO_ReadPin(ECHO_PORT, ECHO_PIN) == GPIO_PIN_SET && --timeout);
    if (timeout == 0) return -1.0f;

    // 5. 记录结束时刻
    end_time = DWT->CYCCNT;

    // 6. 计算高电平时间 (us) 并转换为距离 (mm)
    uint32_t ticks = end_time - start_time;
    uint32_t high_time_us = ticks / (SystemCoreClock / 1000000);
    return (float)high_time_us * 0.17f;
}
