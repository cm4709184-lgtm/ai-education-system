#include "motor.h"
#include "gpio.h"

/* L298N #1 - Left motors (Front-Left + Rear-Left) */
/* ENA/ENB 用 L298N 板上跳线短接，无需代码控制 */
#define L1_IN1_PIN      GPIO_PIN_8    /* PC8 */
#define L1_IN1_PORT     GPIOC
#define L1_IN2_PIN      GPIO_PIN_9    /* PC9 */
#define L1_IN2_PORT     GPIOC
#define L1_IN3_PIN      GPIO_PIN_6    /* PC6 */
#define L1_IN3_PORT     GPIOC
#define L1_IN4_PIN      GPIO_PIN_11   /* PC11 */
#define L1_IN4_PORT     GPIOC

/* L298N #2 - Right motors (Front-Right + Rear-Right) */
/* ENA/ENB 用 L298N 板上跳线短接，无需代码控制 */
#define L2_IN1_PIN      GPIO_PIN_13   /* PB13 */
#define L2_IN1_PORT     GPIOB
#define L2_IN2_PIN      GPIO_PIN_14   /* PB14 */
#define L2_IN2_PORT     GPIOB
#define L2_IN3_PIN      GPIO_PIN_7    /* PB7  */
#define L2_IN3_PORT     GPIOB
#define L2_IN4_PIN      GPIO_PIN_8    /* PB8  */
#define L2_IN4_PORT     GPIOB

static void Motor_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();

    /* L298N #1 方向引脚 - PC8 PC9 PC6 PC11 */
    GPIO_InitStruct.Pin = L1_IN1_PIN | L1_IN2_PIN | L1_IN3_PIN | L1_IN4_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

    /* L298N #2 方向引脚 - PB13 PB14 PB7 PB8 */
    GPIO_InitStruct.Pin = L2_IN1_PIN | L2_IN2_PIN | L2_IN3_PIN | L2_IN4_PIN;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
}

static void Motor_SetLeft(uint8_t dir)
{
    switch (dir)
    {
        case 0: /* stop */
            HAL_GPIO_WritePin(L1_IN1_PORT, L1_IN1_PIN, GPIO_PIN_SET);
            HAL_GPIO_WritePin(L1_IN2_PORT, L1_IN2_PIN, GPIO_PIN_SET);
            HAL_GPIO_WritePin(L1_IN3_PORT, L1_IN3_PIN, GPIO_PIN_SET);
            HAL_GPIO_WritePin(L1_IN4_PORT, L1_IN4_PIN, GPIO_PIN_SET);
            break;
        case 1: /* forward */
            HAL_GPIO_WritePin(L1_IN1_PORT, L1_IN1_PIN, GPIO_PIN_RESET);
            HAL_GPIO_WritePin(L1_IN2_PORT, L1_IN2_PIN, GPIO_PIN_SET);
            HAL_GPIO_WritePin(L1_IN3_PORT, L1_IN3_PIN, GPIO_PIN_SET);
            HAL_GPIO_WritePin(L1_IN4_PORT, L1_IN4_PIN, GPIO_PIN_RESET);
            break;
        case 2: /* backward */
            HAL_GPIO_WritePin(L1_IN1_PORT, L1_IN1_PIN, GPIO_PIN_SET);
            HAL_GPIO_WritePin(L1_IN2_PORT, L1_IN2_PIN, GPIO_PIN_RESET);
            HAL_GPIO_WritePin(L1_IN3_PORT, L1_IN3_PIN, GPIO_PIN_RESET);
            HAL_GPIO_WritePin(L1_IN4_PORT, L1_IN4_PIN, GPIO_PIN_SET);
            break;
    }
}

static void Motor_SetRight(uint8_t dir)
{
    switch (dir)
    {
        case 0: /* stop */
            HAL_GPIO_WritePin(L2_IN1_PORT, L2_IN1_PIN, GPIO_PIN_SET);
            HAL_GPIO_WritePin(L2_IN2_PORT, L2_IN2_PIN, GPIO_PIN_SET);
            HAL_GPIO_WritePin(L2_IN3_PORT, L2_IN3_PIN, GPIO_PIN_SET);
            HAL_GPIO_WritePin(L2_IN4_PORT, L2_IN4_PIN, GPIO_PIN_SET);
            break;
        case 1: /* forward */
            HAL_GPIO_WritePin(L2_IN1_PORT, L2_IN1_PIN, GPIO_PIN_RESET);
            HAL_GPIO_WritePin(L2_IN2_PORT, L2_IN2_PIN, GPIO_PIN_SET);
            HAL_GPIO_WritePin(L2_IN3_PORT, L2_IN3_PIN, GPIO_PIN_RESET);
            HAL_GPIO_WritePin(L2_IN4_PORT, L2_IN4_PIN, GPIO_PIN_SET);
            break;
        case 2: /* backward */
            HAL_GPIO_WritePin(L2_IN1_PORT, L2_IN1_PIN, GPIO_PIN_SET);
            HAL_GPIO_WritePin(L2_IN2_PORT, L2_IN2_PIN, GPIO_PIN_RESET);
            HAL_GPIO_WritePin(L2_IN3_PORT, L2_IN3_PIN, GPIO_PIN_SET);
            HAL_GPIO_WritePin(L2_IN4_PORT, L2_IN4_PIN, GPIO_PIN_RESET);
            break;
    }
}

void Motor_Init(void)
{
    Motor_GPIO_Init();
    Motor_Stop();
}

void Motor_Forward(void)
{
    Motor_SetLeft(1);
    Motor_SetRight(1);
}

void Motor_Backward(void)
{
    Motor_SetLeft(2);
    Motor_SetRight(2);
}

void Motor_TurnLeft(void)
{
    Motor_SetLeft(2);
    Motor_SetRight(1);
}

void Motor_Stop(void)
{
    Motor_SetLeft(0);
    Motor_SetRight(0);
}
