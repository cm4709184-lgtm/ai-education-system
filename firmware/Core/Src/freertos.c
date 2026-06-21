/**
 * @file    freertos.c
 * @brief   FreeRTOS 5任务初始化 — Audio/Inference/CarState/CarControl/Display
 * @author  北京林业大学 理学院 无敌小车车
 * @advisor 张立
 */

/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * File Name          : freertos.c
  * Description        : Code for freertos applications
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "FreeRTOS.h"
#include "task.h"
#include "main.h"
#include "cmsis_os.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN Variables */

/* USER CODE END Variables */
/* Definitions for tasks */
osThreadId_t audioTaskHandle;
const osThreadAttr_t audioTask_attributes = {
  .name = "audioTask",
  .stack_size = 1024 * 3,
  .priority = (osPriority_t) osPriorityNormal,
};

osThreadId_t inferenceTaskHandle;
const osThreadAttr_t inferenceTask_attributes = {
  .name = "inferenceTask",
  .stack_size = 2048 * 3,
  .priority = (osPriority_t) osPriorityNormal,
};

osThreadId_t carStateTaskHandle;
const osThreadAttr_t carStateTask_attributes = {
  .name = "carStateTask",
  .stack_size = 512 * 3,
  .priority = (osPriority_t) osPriorityNormal,
};

osThreadId_t carControlTaskHandle;
const osThreadAttr_t carControlTask_attributes = {
  .name = "carControlTask",
  .stack_size = 512 * 3,
  .priority = (osPriority_t) osPriorityNormal,
};

osThreadId_t displayTaskHandle;
const osThreadAttr_t displayTask_attributes = {
  .name = "displayTask",
  .stack_size = 1024 * 2,
  .priority = (osPriority_t) osPriorityBelowNormal,
};

osThreadId_t serialTaskHandle;
const osThreadAttr_t serialTask_attributes = {
  .name = "serialTask",
  .stack_size = 512 * 4,
  .priority = (osPriority_t) osPriorityNormal,
};

osThreadId_t flashUpdateTaskHandle;
const osThreadAttr_t flashUpdateTask_attributes = {
  .name = "flashUpdateTask",
  .stack_size = 512 * 4,
  .priority = (osPriority_t) osPriorityAboveNormal,
};

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN FunctionPrototypes */

/* USER CODE END FunctionPrototypes */

void MX_FREERTOS_Init(void); /* (MISRA C 2004 rule 8.1) */

/**
  * @brief  FreeRTOS initialization
  * @param  None
  * @retval None
  */
void MX_FREERTOS_Init(void) {
  /* USER CODE BEGIN Init */
  App_InitQueues();
  /* USER CODE END Init */

  /* USER CODE BEGIN RTOS_MUTEX */
  /* USER CODE END RTOS_MUTEX */

  /* USER CODE BEGIN RTOS_SEMAPHORES */
  /* USER CODE END RTOS_SEMAPHORES */

  /* USER CODE BEGIN RTOS_TIMERS */
  /* USER CODE END RTOS_TIMERS */

  /* USER CODE BEGIN RTOS_QUEUES */
  /* USER CODE END RTOS_QUEUES */

  /* Create the thread(s) */
  audioTaskHandle = osThreadNew(AudioTaskEntry, NULL, &audioTask_attributes);
  inferenceTaskHandle = osThreadNew(InferenceTaskEntry, NULL, &inferenceTask_attributes);
  carStateTaskHandle = osThreadNew(CarStateTaskEntry, NULL, &carStateTask_attributes);
  carControlTaskHandle = osThreadNew(CarControlTaskEntry, NULL, &carControlTask_attributes);
  displayTaskHandle = osThreadNew(DisplayTaskEntry, NULL, &displayTask_attributes);
  serialTaskHandle = osThreadNew(SerialTaskEntry, NULL, &serialTask_attributes);
  flashUpdateTaskHandle = osThreadNew(FlashUpdateTaskEntry, NULL, &flashUpdateTask_attributes);

  /* USER CODE BEGIN RTOS_THREADS */
  /* USER CODE END RTOS_THREADS */

  /* USER CODE BEGIN RTOS_EVENTS */
  /* USER CODE END RTOS_EVENTS */

}

/* Private application code --------------------------------------------------*/
/* USER CODE BEGIN Application */

/* USER CODE END Application */

