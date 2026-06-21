/**
 * @file    main.c
 * @brief   离线语音控制小车 — FreeRTOS多任务 + 纯C CNN推理
 * @author  北京林业大学 理学院 无敌小车车
 * @advisor 张立
 */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "cmsis_os.h"
#include "dma.h"
#include "i2c.h"
#include "i2s.h"
#include "spi.h"
#include "usart.h"
#include "usb_host.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stm32f4_discovery.h"
#include "stm32f4_discovery_audio.h"
#include "stm32f4_discovery_accelerometer.h"
#include "pdm2pcm_glo.h"  /* 保留（BSP音频库依赖） */
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include "OLED.h"
#include "hcsr04.h"

#include "mel_spectrogram.h"
#include "inference.h"
#include "model_weights.h"
#include "motor.h"
#include "model_loader.h"
#include "serial_protocol.h"
#include "flash_update.h"

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define AUDIO_SAMPLING_FREQUENCY        32000
#define AUDIO_VOLUME                    100

#define PCM_SUB_BUFFER_SIZE             512   /* >= MEL_HOP_SIZE*2 for mel spectrogram */
#define ADC_BUFFER_SIZE                 6144  /* 12KB 缓冲区 */

#define HIGH_PASS_TAP                   2122358088

/* ADC (MAX9814 模拟麦克风) */
#define ADC_PIN                         GPIO_PIN_0
#define ADC_PORT                        GPIOB
#define ADC_CHANNEL                     ADC_CHANNEL_8

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
extern DMA_HandleTypeDef hdma_i2s3_ext_rx;  /* 保留（BSP依赖） */
extern I2S_HandleTypeDef hi2s2;              /* 保留（BSP依赖） */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
volatile int current_label = 1;

/* ADC 双缓冲（MAX9814 模拟麦克风） */
static uint16_t adc_buffer[ADC_BUFFER_SIZE];
static TIM_HandleTypeDef htim2;

bool is_recording_active = false;

int16_t pcm_play_buffer[2][PCM_SUB_BUFFER_SIZE * 2];
volatile uint8_t play_active_buffer = 0;
volatile uint8_t play_next_ready = 0;

int16_t pcm_mono_buffer[PCM_SUB_BUFFER_SIZE];

volatile bool adc_half_ready = false;
volatile bool adc_full_ready = false;

/* 诊断：每 N 次打印一次统计 */
static volatile uint32_t diag_counter = 0;
#define DIAG_INTERVAL  100  /* 每100个半缓冲区打印一次 */

typedef enum {
    STATE_IDLE = 0,
    STATE_RECORDING
} RecordState_t;

volatile RecordState_t record_state = STATE_IDLE;
volatile bool button_pressed = false;
volatile uint8_t serial_mel_output_enabled = 1;

volatile SystemState_t system_state = SYS_OFF;
volatile CarCommand_t  car_command   = CMD_NONE;
volatile float         ultrasonic_distance = 999.0f;
volatile int16_t       accel_xyz[3] = {0};
volatile char          last_inference_label[24] = "none";
volatile char          last_confirmed_label[24] = "---";
volatile float         last_confidence = 0.0f;
volatile float         confirmed_confidence = 0.0f;
volatile uint32_t      last_inference_us = 0;
volatile int           infer_reset_flag = 0;

#define LISTENING_DURATION_MS  3000
#define OBSTACLE_THRESHOLD_MM   100

static const char* cmd_name[] = {"NONE","FORWARD","BACKWARD","TURN","START","STOP"};
static volatile float current_rms = 0.0f;
static volatile float current_agc_gain = 1.0f;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
void PeriphCommonClock_Config(void);
void MX_FREERTOS_Init(void);
/* USER CODE BEGIN PFP */
static void ProcessAudio(void);
static void uart_printf(const char* fmt, ...);

static void safe_strncpy_volatile(volatile char *dest, const char *src, size_t n)
{
    for (size_t i = 0; i < n && src[i] != '\0'; i++)
        dest[i] = src[i];
    if (n > 0) dest[n-1] = '\0';
}
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/**
  * @brief 音频系统初始化 — ADC 模拟麦克风 + I2S3 播放
  */
static void ADC_Mic_Init(void);

static void Audio_Init(void)
{
  /* ── 诊断：打印音频配置参数 ── */
  uart_printf("\n[AUDIO] ====== 音频配置 ======\n");
  uart_printf("[AUDIO] SAMPLING_FREQ=%d\n", AUDIO_SAMPLING_FREQUENCY);
  uart_printf("[AUDIO] PCM_BUF=%d\n", PCM_SUB_BUFFER_SIZE);
  uart_printf("[AUDIO] INPUT=ADC (MAX9814 on PB0)\n");

  /* 验证 PLLI2S 时钟 */
  RCC_PeriphCLKInitTypeDef clkchk;
  HAL_RCCEx_GetPeriphCLKConfig(&clkchk);
  uint32_t plln = clkchk.PLLI2S.PLLI2SN;
  uint32_t pllr = clkchk.PLLI2S.PLLI2SR;
  uart_printf("[AUDIO] PLLI2SN=%lu PLLI2SR=%lu\n", plln, pllr);

  uint32_t i2sclk = 1000000UL * plln / pllr;
  uart_printf("[AUDIO] I2SCLK=%lu Hz\n", i2sclk);

  /* I2S3 播放验算 */
  uint32_t tmp_out = (((i2sclk / 256) * 10) / AUDIO_SAMPLING_FREQUENCY + 5) / 10;
  uint32_t odd_out = tmp_out & 1;
  uint32_t div_out = (tmp_out - odd_out) / 2;
  uart_printf("[AUDIO] I2S3: tmp=%lu div=%lu odd=%lu\n", tmp_out, div_out, odd_out);
  if (div_out < 2)
    uart_printf("[AUDIO] !! I2S3 div<2 -> HAL_ERROR !!\n");

  uart_printf("[AUDIO] ========================\n");

  /* 1. 初始化 ADC 麦克风 */
  ADC_Mic_Init();

  /* 2. 初始化音频输出（耳机） */
  if (BSP_AUDIO_OUT_Init(OUTPUT_DEVICE_HEADPHONE, AUDIO_VOLUME, AUDIO_SAMPLING_FREQUENCY) != AUDIO_OK)
  {
    while (1) { BSP_LED_Toggle(LED4); HAL_Delay(100); }
  }

  /* 3. 启动播放 */
  memset(pcm_play_buffer[0], 0, PCM_SUB_BUFFER_SIZE * 2 * sizeof(int16_t));
  BSP_AUDIO_OUT_Play((uint16_t*)pcm_play_buffer[0], PCM_SUB_BUFFER_SIZE * 2 * sizeof(int16_t));
}

/**
  * @brief ADC1 (PB0) + DMA + TIM2 初始化（直接寄存器操作）
  *        TIM2 触发 16kHz 采样，DMA 循环填充 adc_buffer
  */
static void ADC_Mic_Init(void)
{
  /* GPIO: PB0 模拟输入 */
  __HAL_RCC_GPIOB_CLK_ENABLE();
  GPIO_InitTypeDef gpio = {0};
  gpio.Pin  = ADC_PIN;
  gpio.Mode = GPIO_MODE_ANALOG;
  gpio.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(ADC_PORT, &gpio);

  /* DMA 清除 */
  __HAL_RCC_DMA2_CLK_ENABLE();
  DMA2_Stream0->CR  = 0;
  for (volatile int t = 0; t < 10000 && (DMA2_Stream0->CR & DMA_SxCR_EN); t++);
  DMA2_Stream0->PAR  = (uint32_t)&ADC1->DR;
  DMA2_Stream0->M0AR = (uint32_t)adc_buffer;
  DMA2_Stream0->NDTR = ADC_BUFFER_SIZE;
  DMA2_Stream0->CR   = DMA_PDATAALIGN_HALFWORD
                      | DMA_MDATAALIGN_HALFWORD
                      | DMA_MINC_ENABLE
                      | DMA_CIRCULAR
                      | DMA_PRIORITY_HIGH
                      | DMA_SxCR_HTIE           /* Half-transfer interrupt */
                      | DMA_SxCR_TCIE;          /* Transfer-complete interrupt */
  /* 显式设置 Channel 0（清 bits 27:25） */
  DMA2_Stream0->CR &= ~(0x07 << 25);
  DMA2_Stream0->CR |= DMA_SxCR_EN;

  /* NVIC: DMA2 Stream0 */
  HAL_NVIC_SetPriority(DMA2_Stream0_IRQn, 6, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream0_IRQn);

  /* ADC1 校准（用最小配置） */
  __HAL_RCC_ADC1_CLK_ENABLE();
  ADC1->CR1 = 0;
  ADC1->CR2 = (1 << 0);  /* 仅 ADON */
  for (volatile int d = 0; d < 10000; d++);
  ADC1->CR2 |= (1 << 3);
  for (volatile int t = 0; t < 100000 && (ADC1->CR2 & (1 << 3)); t++);
  ADC1->CR2 |= (1 << 2);
  for (volatile int t = 0; t < 100000 && (ADC1->CR2 & (1 << 2)); t++);

  /* ADC1 最终配置：连续转换 + DMA */
  ADC1->CR1 = 0;
  ADC1->SMPR2 = (0x04 << 24);  /* Ch8: 84 cycles (42MHz/96=437.5kHz, buf=2.34ms) */
  ADC1->SQR1 = 0;              /* 1 conversion */
  ADC1->SQR3 = 8;              /* Ch8 */
  ADC1->CR2 = (1 << 0)         /* ADON */
            | (1 << 1)          /* CONT: 连续转换 */
            | (1 << 8)          /* DMA */
            | (1 << 9);         /* DDS */

  /* 启动 ADC 连续转换 + DMA */
  ADC1->CR2 |= (1 << 30);      /* SWSTART */
}

/* DMA2 Stream0 中断（直接检查标志，无需 HAL ADC 库） */
void DMA2_Stream0_IRQHandler(void)
{
  uint32_t lisr = DMA2->LISR;
  if (lisr & DMA_LISR_TCIF0)  /* Transfer complete */
  {
    DMA2->LIFCR = DMA_LIFCR_CTCIF0;
    adc_full_ready = true;
  }
  if (lisr & DMA_LISR_HTIF0)  /* Half transfer */
  {
    DMA2->LIFCR = DMA_LIFCR_CHTIF0;
    adc_half_ready = true;
  }
}

/**
  * @brief 音频后处理：RMS-AGC + 噪声门限
  *        RMS检测 + 极慢时间常数，避免尖锐失真
  *        目标使远距离（1.7-5m）增益到近距离（40cm）音量水平
  */
#define AGC_RMS_TARGET    1500
#define AGC_MIN_GAIN      0.3f
#define AGC_MAX_GAIN      5.0f
#define AGC_NOISE_FLOOR   500    /* RMS低于此视为静音，不更新增益 */

static void ApplyAudioProcessing(int16_t *pcm, uint32_t size)
{
    static float agc_gain = 1.0f;
    static float smooth_rms = 0.0f;
    float sum_sq = 0.0f;

    for (uint32_t i = 0; i < size; i++)
    {
        int32_t s = pcm[i];
        sum_sq += (float)(s * s);
    }
    float rms = sqrtf(sum_sq / (float)size);

    /* 慢速RMS平滑 */
    float coeff = (rms > smooth_rms) ? 0.02f : 0.10f;
    smooth_rms = smooth_rms * (1.0f - coeff) + rms * coeff;
    current_rms = smooth_rms;

    /* AGC 增益：目标 RMS 3200 */
    if (smooth_rms > AGC_NOISE_FLOOR)
    {
        float desired = (float)AGC_RMS_TARGET / (smooth_rms + 1.0f);
        if (desired < AGC_MIN_GAIN) desired = AGC_MIN_GAIN;
        if (desired > AGC_MAX_GAIN) desired = AGC_MAX_GAIN;
        agc_gain = agc_gain * 0.75f + desired * 0.25f;
    }
    current_agc_gain = agc_gain;

    /* 静音门限 + 增益应用 */
    for (uint32_t i = 0; i < size; i++)
    {
        int32_t processed;
        if (pcm[i] > -AGC_NOISE_FLOOR && pcm[i] < AGC_NOISE_FLOOR)
            processed = 0;
        else
            processed = (int32_t)(pcm[i] * agc_gain);

        if (processed > 32767)  processed = 32767;
        if (processed < -32768) processed = -32768;
        pcm[i] = (int16_t)processed;
    }
}

/**
  * @brief 单声道转立体声
  */
static void MonoToStereo(const int16_t *mono, int16_t *stereo, uint32_t mono_len)
{
    for (uint32_t i = 0; i < mono_len; i++)
    {
        stereo[i * 2]     = mono[i];
        stereo[i * 2 + 1] = mono[i];
    }
}

/**
  * @brief 用户按钮按下回调
  */
void BSP_PB_Callback(Button_TypeDef Button)
{
    (void)Button;
}

static uint8_t uart_rx_byte;

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2)
    {
        SerialProtocol_RxByte(uart_rx_byte);
        HAL_UART_Receive_IT(&huart2, &uart_rx_byte, 1);
    }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* Configure the peripherals common clocks */
  PeriphCommonClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_I2C1_Init();
  MX_I2S2_Init();
  MX_I2S3_Init();
  MX_SPI1_Init();
  MX_USART2_UART_Init();
  /* USER CODE BEGIN 2 */
  BSP_LED_Init(LED3);
  BSP_LED_Init(LED4);
  BSP_LED_Init(LED5);
  BSP_LED_Init(LED6);

  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  OLED_Init();
  OLED_Clear();
  OLED_Printf(0, 0, OLED_8X16, "Audio System");
  OLED_Printf(0, 16, OLED_6X8, "Ready...");
  OLED_Update();

  /* LED 快速闪烁指示初始化 */
  BSP_LED_On(LED3); BSP_LED_On(LED4); BSP_LED_On(LED5); BSP_LED_On(LED6);
  HAL_Delay(200);
  BSP_LED_Off(LED3); BSP_LED_Off(LED4); BSP_LED_Off(LED5); BSP_LED_Off(LED6);

  BSP_PB_Init(BUTTON_KEY, BUTTON_MODE_EXTI);

  Audio_Init();
  MelSpectrogram_Init();
  Inference_Init();
  ModelLoader_Init();
  ModelLoader_DumpModelInfo("BOOT");  /* 启动自检：打印 SE 权重指针 + 前几个值，便于排查错位 */

  if (g_active_model.source == MODEL_SOURCE_COMPILE)
    BSP_LED_On(LED6);
  else
    BSP_LED_Off(LED6);
  SerialProtocol_Init();
  FlashUpdate_Init();
  HAL_UART_Receive_IT(&huart2, &uart_rx_byte, 1);
  BSP_ACCELERO_Init();
  Motor_Init();
  Motor_Stop();

  /* USER CODE END 2 */

  /* Init scheduler */
  osKernelInitialize();  /* Call init function for freertos objects (in cmsis_os2.c) */
  MX_FREERTOS_Init();

  /* Start scheduler */
  osKernelStart();

  /* We should never get here as control is now taken by the scheduler */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    /* 主循环逻辑已迁移至 4 个 FreeRTOS 任务中并行运行 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief Peripherals Common Clock Configuration
  * @retval None
  */
void PeriphCommonClock_Config(void)
{
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  /** Initializes the peripherals clock
  */
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_I2S;
  PeriphClkInitStruct.PLLI2S.PLLI2SN = 256;
  PeriphClkInitStruct.PLLI2S.PLLI2SR = 1;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

void BSP_AUDIO_IN_ClockConfig(I2S_HandleTypeDef *hi2s, uint32_t AudioFreq, void *Params)
{
    RCC_PeriphCLKInitTypeDef rccclkinit;
    HAL_RCCEx_GetPeriphCLKConfig(&rccclkinit);
    rccclkinit.PeriphClockSelection = RCC_PERIPHCLK_I2S;
    rccclkinit.PLLI2S.PLLI2SN = 256;
    rccclkinit.PLLI2S.PLLI2SR = 1;
    HAL_RCCEx_PeriphCLKConfig(&rccclkinit);
}

void BSP_AUDIO_OUT_ClockConfig(I2S_HandleTypeDef *hi2s, uint32_t AudioFreq, void *Params)
{
    RCC_PeriphCLKInitTypeDef rccclkinit;
    HAL_RCCEx_GetPeriphCLKConfig(&rccclkinit);
    rccclkinit.PeriphClockSelection = RCC_PERIPHCLK_I2S;
    rccclkinit.PLLI2S.PLLI2SN = 256;
    rccclkinit.PLLI2S.PLLI2SR = 1;
    HAL_RCCEx_PeriphCLKConfig(&rccclkinit);
}
/* USER CODE BEGIN 4 */
/**
  * @brief 串口格式化输出函数
  * @details 通过USART2发送格式化字符串，支持可变参数
  * @param fmt 格式化字符串
  * @param ... 可变参数列表
  * @retval 无
  * @note 使用USART2，波特率921600
  * @warning 内部缓冲区大小为128字节，超过会被截断
  */
void uart_printf(const char* fmt, ...)
{
    uint8_t send_buf[512] = {0};  /* 发送缓冲区 */
    va_list args;                 /* 可变参数列表 */
    va_start(args, fmt);          /* 初始化可变参数 */
    vsnprintf((char*)send_buf, sizeof(send_buf)-1, fmt, args);  /* 格式化字符串 */
    va_end(args);                 /* 结束可变参数处理 */
    HAL_UART_Transmit(&huart2, send_buf, strlen((char*)send_buf), HAL_MAX_DELAY);  /* 发送数据 */
}

/* 录音 I2S 时钟配置重写（固定 16kHz） */


static void ProcessAudio(void)
{
  int16_t *pcm_mono = pcm_mono_buffer;
  int16_t *pcm_stereo;
  uint8_t target_buffer;

  /* 轮询 DMA NDTR 获取 ADC 数据（连续模式，~168kHz ADC → 16kHz 降采样） */
  /* 轮询 DMA NDTR，5:1 直接抽取（保留高频） */
  static uint16_t last_pos = 0;
  static uint16_t sample_idx = 0;
  static uint16_t skip = 0;

  uint16_t pos = (uint16_t)(ADC_BUFFER_SIZE - DMA2_Stream0->NDTR);
  uint16_t available = (uint16_t)((pos - last_pos + ADC_BUFFER_SIZE) % ADC_BUFFER_SIZE);
  if (available == 0) return;

  for (uint16_t i = 0; i < available; i++)
  {
    uint16_t idx = (last_pos + i) % ADC_BUFFER_SIZE;
    if (++skip >= 26)  /*26:1: 437.5kHz/ = 16.25kHz (匹配32kHz播放) */
    {
      skip = 0;
      pcm_mono[sample_idx++] = (int16_t)((int32_t)adc_buffer[idx] - 2048);
      if (sample_idx >= PCM_SUB_BUFFER_SIZE)
      {
        /* 暂时禁用所有滤波，测试原始数据 */

        /* ApplyAudioProcessing(pcm_mono, PCM_SUB_BUFFER_SIZE); */

        /* 计算 RMS */
        {
          float sum_sq = 0.0f;
          for (int k = 0; k < PCM_SUB_BUFFER_SIZE; k++)
            sum_sq += (float)pcm_mono[k] * pcm_mono[k];
          current_rms = sqrtf(sum_sq / PCM_SUB_BUFFER_SIZE);
        }

        /* 平滑噪声门：RMS<200 时逐步衰减 */
        {
          static float gate_gain = 1.0f;
          float target = (current_rms >= 100.0f) ? 1.0f : 0.0f;
          /* 快速打开(0.2)，慢速关闭(0.01) */
          float alpha = (target > gate_gain) ? 0.2f : 0.01f;
          gate_gain += alpha * (target - gate_gain);
          for (int k = 0; k < PCM_SUB_BUFFER_SIZE; k++)
            pcm_mono[k] = (int16_t)(pcm_mono[k] * gate_gain);
        }

        MelSpectrogram_ProcessBlock(pcm_mono, PCM_SUB_BUFFER_SIZE);

        target_buffer = play_active_buffer ^ 1;
        pcm_stereo = pcm_play_buffer[target_buffer];
        MonoToStereo(pcm_mono, pcm_stereo, PCM_SUB_BUFFER_SIZE);
        __disable_irq();
        play_next_ready = 1;
        __enable_irq();
        BSP_LED_Toggle(LED4);
        sample_idx = 0;
      }
    }
  }
  last_pos = pos;
}

/* I2S2 录音回调（不再使用，保留空实现） */
void BSP_AUDIO_IN_HalfTransfer_CallBack(void) {}
void BSP_AUDIO_IN_TransferComplete_CallBack(void) {}
void BSP_AUDIO_IN_Error_Callback(void) {}

/* 播放回调 */
void BSP_AUDIO_OUT_HalfTransfer_CallBack(void) {}

void BSP_AUDIO_OUT_TransferComplete_CallBack(void)
{
  uint8_t next_ready;
  __disable_irq();
  next_ready = play_next_ready;
  if (next_ready) {
    play_next_ready = 0;
  }
  __enable_irq();

  if (next_ready)
  {
    play_active_buffer ^= 1;
    BSP_AUDIO_OUT_ChangeBuffer((uint16_t*)pcm_play_buffer[play_active_buffer],
                               PCM_SUB_BUFFER_SIZE * 2);
  }
  else
  {
    BSP_AUDIO_OUT_ChangeBuffer((uint16_t*)pcm_play_buffer[play_active_buffer],
                               PCM_SUB_BUFFER_SIZE * 2);
  }
}

/* 错误回调 */
void BSP_AUDIO_OUT_Error_CallBack(void) { while(1); }

static osMessageQueueId_t infCmdQueue;
static osMessageQueueId_t carCmdQueue;

void App_InitQueues(void)
{
  infCmdQueue = osMessageQueueNew(16, sizeof(CarCommand_t), NULL);
  carCmdQueue = osMessageQueueNew(16, sizeof(CarCommand_t), NULL);
}

void AudioTaskEntry(void *argument)
{
  MX_USB_HOST_Init();
  for (;;)
  {
    ProcessAudio();
    osDelay(1);
  }
}

#define INF_HIST_SIZE 3
static int inf_hist_buf[INF_HIST_SIZE];
static int inf_hist_write = 0;
static float prev_probs[MODEL_MAX_CLASSES];

static void InfHist_Push(int label)
{
  inf_hist_buf[inf_hist_write] = label;
  inf_hist_write = (inf_hist_write + 1) % INF_HIST_SIZE;
}

static inline int IsNoise(int idx)
{
  return (CarCommand_t)g_active_model.class_map[idx] == CMD_NONE;
}

static int FindNoiseIdx(void)
{
  for (int i = 0; i < (int)g_active_model.num_classes; i++)
    if (IsNoise(i)) return i;
  return g_active_model.num_classes - 1;
}

static int InfHist_HasNonNoise(void)
{
  for (int i = 0; i < INF_HIST_SIZE; i++)
    if (!IsNoise(inf_hist_buf[i]))
      return 1;
  return 0;
}

/* 按 CMD 编号聚合投票：多类可映射到同一命令（如 pai1+start → CMD_START） */
static int InfHist_DominantCmd(void)
{
  int cnt[MODEL_MAX_CLASSES] = {0};
  int cmd_best_class[MODEL_MAX_CLASSES] = {0};
  for (int i = 0; i < INF_HIST_SIZE; i++)
  {
    int lbl = inf_hist_buf[i];
    if (lbl >= 0 && lbl < (int)g_active_model.num_classes && !IsNoise(lbl)) {
      int cmd_idx = (int)g_active_model.class_map[lbl];
      if (cmd_idx >= 0 && cmd_idx < MODEL_MAX_CLASSES) {
        cnt[cmd_idx]++;
        cmd_best_class[cmd_idx] = lbl;
      }
    }
  }

  int best = -1, best_cnt = 0;
  for (int c = 0; c < MODEL_MAX_CLASSES; c++)
  {
    if (cnt[c] > best_cnt)
    {
      best_cnt = cnt[c];
      best = cmd_best_class[c];
    }
  }

  /* 3帧中需>=2帧为同一命令（约67%，取59%兼容浮点） */
  if (best >= 0 && best_cnt * 100 >= INF_HIST_SIZE * 59)
    return best;
  return -1;
}

static void InfHist_Print(void)
{
  uart_printf("[");
  for (int i = 0; i < INF_HIST_SIZE; i++)
  {
    int idx = (inf_hist_write + i) % INF_HIST_SIZE;
    uart_printf("%s", g_active_model.class_names[inf_hist_buf[idx]]);
    if (i < INF_HIST_SIZE - 1) uart_printf(" ");
  }
  uart_printf("]");
}

void InferenceTaskEntry(void *argument)
{
  static float inf_win[INF_FRAME_WINDOW * INF_MEL_FILTERS];
  static int inf_count = 0;

  for (;;)
  {
    if (infer_reset_flag)
    {
      infer_reset_flag = 0;
      inf_count = 0;
      safe_strncpy_volatile(last_confirmed_label, "---", 24);
      confirmed_confidence = 0.0f;
      __disable_irq();
      for (int i = 0; i < INF_HIST_SIZE; i++)
        inf_hist_buf[i] = FindNoiseIdx();
      inf_hist_write = 0;
      __enable_irq();
      __disable_irq();
      mel_frame_count = 0;
      mel_frame_read_idx = 0;
      __enable_irq();
    }

    __disable_irq();
    int count = mel_frame_count;
    if (count > 0)
    {
      if (count > MEL_MAX_FRAMES / 2)
        count = MEL_MAX_FRAMES / 2;
      mel_frame_count -= count;
    }
    __enable_irq();

    if (count > 0)
    {
      int idx = mel_frame_read_idx;

      /* 串口 mel 输出（仅 SYS_OFF 状态） */
      if (system_state == SYS_OFF && serial_mel_output_enabled)
      {
        for (int f = 0; f < count; f++)
        {
          int slot = (idx + f) & (MEL_MAX_FRAMES - 1);
          /* 关中断复制，防止音频任务覆盖 */
          float local[MEL_NUM_FILTERS];
          __disable_irq();
          memcpy(local, mel_frames[slot], sizeof(local));
          __enable_irq();
          SerialProtocol_SendMelFrame(local, MEL_NUM_FILTERS);
        }
      }

      if (system_state == SYS_OFF)
      {
        /* OFF 状态：仅丢弃 */
      }
      else
      {
        for (int f = 0; f < count; f++)
        {
          int slot = (idx + f) & (MEL_MAX_FRAMES - 1);
          if (inf_count < INF_FRAME_WINDOW)
          {
            for (int m = 0; m < MEL_NUM_FILTERS; m++)
              inf_win[inf_count * INF_MEL_FILTERS + m] = mel_frames[slot][m];
            inf_count++;
          }
        }
        if (inf_count >= INF_FRAME_WINDOW)
        {
          float conf = 0.0f;
          uint32_t t0 = DWT->CYCCNT;
          int raw = Inference_Run(inf_win, &conf);
          uint32_t t1 = DWT->CYCCNT;
          last_inference_us = (t1 - t0) / 168;
          safe_strncpy_volatile(last_inference_label, g_active_model.class_display[raw], 24);
          last_confidence = conf;
          int capped = raw;
          float base_thresh = g_active_model.thresholds[raw];
          int is_start = ((CarCommand_t)g_active_model.class_map[raw] == CMD_START);
          float array_thresh = is_start ? base_thresh + 0.20f : base_thresh;
          float fast_thresh  = is_start ? 0.60f : array_thresh;
          if (conf < array_thresh)
            capped = FindNoiseIdx();

          int conf_pct = (int)(conf * 100.0f);

          uart_printf("[DBG] %s %d%% rms:%.0f\n", g_active_model.class_display[raw], conf_pct, (double)current_rms);

          if (conf >= fast_thresh && !IsNoise(raw))
          {
            uart_printf("[CMD] %s (fast %d%%)\n", g_active_model.class_display[raw], conf_pct);
              CarCommand_t cmd = (CarCommand_t)g_active_model.class_map[raw];
            safe_strncpy_volatile(last_confirmed_label, g_active_model.class_display[raw], 24);
            confirmed_confidence = conf;
            osMessageQueuePut(infCmdQueue, &cmd, 0, 0);
            for (int i = 0; i < INF_HIST_SIZE; i++)
              inf_hist_buf[i] = FindNoiseIdx();
            inf_hist_write = 0;
          }
          else
          {
            /* 通道 3: 跃升检测 — 相邻帧置信度跳变 ≥ 0.20 */
            int delta_cmd = -1;
            float cur_probs[MODEL_MAX_CLASSES] = {0};
            Inference_GetProbs(cur_probs, MODEL_MAX_CLASSES);
            for (int ci = 0; ci < (int)g_active_model.num_classes; ci++)
            {
              float delta = cur_probs[ci] - prev_probs[ci];
              if (delta >= 0.20f && !IsNoise(ci))
              {
                delta_cmd = ci;
                break;
              }
              prev_probs[ci] = cur_probs[ci];
            }
            if (delta_cmd >= 0)
            {
              uart_printf("[CMD] %s (delta %d%%)\n", g_active_model.class_display[delta_cmd],
                          (int)(cur_probs[delta_cmd] * 100.0f));
              CarCommand_t cmd = (CarCommand_t)g_active_model.class_map[delta_cmd];
              safe_strncpy_volatile(last_confirmed_label, g_active_model.class_display[delta_cmd], 24);
              confirmed_confidence = cur_probs[delta_cmd];
              osMessageQueuePut(infCmdQueue, &cmd, 0, 0);
              for (int i = 0; i < INF_HIST_SIZE; i++)
                inf_hist_buf[i] = FindNoiseIdx();
              inf_hist_write = 0;
            }
            else
            {
            InfHist_Push(capped);
            int vote = InfHist_DominantCmd();
            if (vote >= 0)
            {
              uart_printf("[CMD] %s (%d%%)\n", g_active_model.class_display[vote], (int)(conf * 100.0f));
              CarCommand_t cmd = (CarCommand_t)g_active_model.class_map[vote];
              safe_strncpy_volatile(last_confirmed_label, g_active_model.class_display[vote], 24);
              confirmed_confidence = conf;
              osMessageQueuePut(infCmdQueue, &cmd, 0, 0);
              for (int i = 0; i < INF_HIST_SIZE; i++)
                inf_hist_buf[i] = FindNoiseIdx();
              inf_hist_write = 0;
            }
            }
          }

          for (int i = 0; i < INF_FRAME_WINDOW / 2; i++)
            for (int m = 0; m < MEL_NUM_FILTERS; m++)
              inf_win[i * INF_MEL_FILTERS + m] =
                  inf_win[(i + INF_FRAME_WINDOW / 2) * INF_MEL_FILTERS + m];
          inf_count = INF_FRAME_WINDOW / 2;
        }
      }
      mel_frame_read_idx = (idx + count) & (MEL_MAX_FRAMES - 1);
    }

    osDelay(1);
  }
}

void CarStateTaskEntry(void *argument)
{
  uint32_t listening_start_tick = 0;
  uint32_t last_button_tick = 0;
  bool btn_was_pressed = false;
  uint32_t btn_press_start = 0;
  bool btn_long_triggered = false;

  for (;;)
  {
    bool btn_now = (HAL_GPIO_ReadPin(B1_GPIO_Port, B1_Pin) == GPIO_PIN_SET);

    if (btn_now && !btn_was_pressed)
    {
      btn_press_start = osKernelGetTickCount();
      btn_long_triggered = false;
    }

    if (btn_now && btn_was_pressed && !btn_long_triggered)
    {
      if (osKernelGetTickCount() - btn_press_start > 1000)
      {
        btn_long_triggered = true;
        if (g_active_model.source == MODEL_SOURCE_COMPILE)
        {
          if (ModelLoader_LoadSlot(MODEL_SOURCE_SLOT_A))
          {
            BSP_LED_Off(LED6);
            uart_printf("[MODEL] -> Flash (%dc)\n", g_active_model.num_classes);
          }
          else
          {
            uart_printf("[MODEL] Flash load failed!\n");
          }
        }
        else
        {
          ModelLoader_Fallback();
          BSP_LED_On(LED6);
          uart_printf("[MODEL] -> Compiled Base (%dc)\n", g_active_model.num_classes);
        }
      }
    }

    if (!btn_now && btn_was_pressed && !btn_long_triggered)
    {
      uint32_t held = osKernelGetTickCount() - btn_press_start;
      if (held >= 50 && held < 1000)
      {
        uint32_t now = osKernelGetTickCount();
        if (now - last_button_tick >= 200)
        {
          last_button_tick = now;
          if (system_state > SYS_OFF)
          {
            system_state = SYS_OFF;
            car_command = CMD_NONE;
            BSP_LED_Off(LED3);
            BSP_LED_Off(LED4);
            uart_printf("[SYS] OFF\n");
          }
          else
          {
            system_state = SYS_IDLE;
            infer_reset_flag = 1;
            BSP_LED_On(LED3);
            uart_printf("[SYS] IDLE\n");
          }
        }
      }
    }

    btn_was_pressed = btn_now;

    if (system_state == SYS_IDLE || system_state == SYS_LISTENING)
    {
      for (int drain = 0; drain < 8; drain++)
      {
        CarCommand_t cmd;
        if (osMessageQueueGet(infCmdQueue, &cmd, NULL, 0) != osOK)
          break;

        if (system_state == SYS_IDLE)
        {
          if (cmd == CMD_START)
          {
            /* 只有车在动（FORWARD/BACKWARD/TURN）时才静默
               - car_command == CMD_STOP 时立即可启动
               - 防止车还在跑就切到 LISTEN 模式 */
            bool car_is_moving = (car_command == CMD_FORWARD ||
                                  car_command == CMD_BACKWARD ||
                                  car_command == CMD_TURN);
            if (!car_is_moving)
            {
              Motor_Stop();
              car_command = CMD_STOP;
              system_state = SYS_LISTENING;
              BSP_LED_Off(LED3);
              BSP_LED_On(LED4);
              listening_start_tick = osKernelGetTickCount();
              uart_printf("[SYS] LISTENING (3s)\n");
              goto skip_queue;
            }
            else
            {
              uart_printf("[SYS] start ignored: car still moving (%s)\n", cmd_name[car_command]);
            }
          }
        }
        else if (system_state == SYS_LISTENING)
        {
          if (cmd != CMD_START)
          {
            uint32_t elapsed = osKernelGetTickCount() - listening_start_tick;
            if (elapsed < 500)
            {
              uart_printf("[SYS] cooldown, ignoring %s\n", cmd > CMD_STOP ? "?" : cmd_name[cmd]);
              continue;
            }
            car_command = cmd;
            system_state = SYS_IDLE;
            infer_reset_flag = 1;
            BSP_LED_On(LED3);
            BSP_LED_Off(LED4);
            switch(cmd)
            {
              default: break;
            }
            osMessageQueuePut(carCmdQueue, &cmd, 0, 0);
            uart_printf("[SYS] IDLE (cmd:%s)\n", cmd > CMD_STOP ? "?" : cmd_name[cmd]);
          }
          else
          {
            listening_start_tick = osKernelGetTickCount();
            uart_printf("[SYS] LISTENING (reset timer)\n");
          }
        }
      }
    }
  skip_queue:

    if (system_state == SYS_LISTENING)
    {
      uint32_t elapsed = osKernelGetTickCount() - listening_start_tick;
      if (elapsed >= LISTENING_DURATION_MS)
      {
        system_state = SYS_IDLE;
        infer_reset_flag = 1;
        BSP_LED_On(LED3);
        BSP_LED_Off(LED4);
        uart_printf("[SYS] IDLE (timeout)\n");
      }
    }

    osDelay(10);
  }
}

void CarControlTaskEntry(void *argument)
{
  uint32_t last_sensor_tick = 0;
  uint32_t last_sonar_tick = 0;
  CarCommand_t prev = CMD_NONE;

  uint32_t last_tap_tick = 0;
  int tap_count = 0;
  uint32_t lift_tick = 0;
  int lift_reported = 0;
  uint32_t auto_begin_tick = 0;

  for (;;)
  {
    CarCommand_t cmd;
    if (osMessageQueueGet(carCmdQueue, &cmd, NULL, 10) == osOK)
    {
      if (cmd != prev)
      {
        prev = cmd;
        lift_reported = 0;

        switch (cmd)
        {
          case CMD_FORWARD:
            uart_printf("[CAR] GO FORWARD\n");
            Motor_Forward();
            auto_begin_tick = osKernelGetTickCount();
            break;
          case CMD_BACKWARD:
            uart_printf("[CAR] GO BACKWARD\n");
            Motor_Backward();
            auto_begin_tick = osKernelGetTickCount();
            break;
          case CMD_TURN:
            uart_printf("[CAR] TANK TURN\n");
            Motor_TurnLeft();
            auto_begin_tick = osKernelGetTickCount();
            break;
          case CMD_STOP:
            uart_printf("[CAR] STOP\n");
            Motor_Stop();
            break;
          default:
            break;
        }
      }
    }

    /* ── Auto-stop after 5s ── */
    if (auto_begin_tick != 0 && (car_command == CMD_FORWARD || car_command == CMD_BACKWARD || car_command == CMD_TURN))
    {
      if (osKernelGetTickCount() - auto_begin_tick >= 5000)
      {
        uart_printf("[CAR] AUTO-STOP (5s timeout)\n");
        car_command = CMD_STOP;
        prev = CMD_STOP;
        Motor_Stop();
        auto_begin_tick = 0;
      }
    }

    uint32_t now = osKernelGetTickCount();

    /* ── Accelerometer: 50Hz ── */
    if (now - last_sensor_tick >= 20)
    {
      last_sensor_tick = now;

      BSP_ACCELERO_GetXYZ(accel_xyz);
      int ax = accel_xyz[0], ay = accel_xyz[1], az = accel_xyz[2];
      int mag_sq = ax*ax + ay*ay + az*az;

      /* ── Double-tap wake (disabled, use button instead) ── */
      #if 0
      if (mag_sq > 5760000 && now - last_tap_tick > 200)
      {
        last_tap_tick = now;
        tap_count++;
      }

      if (tap_count >= 2 && now - last_tap_tick < 500)
      {
        tap_count = 0;
        if (system_state == SYS_OFF)
        {
          system_state = SYS_IDLE;
          infer_reset_flag = 1;
          BSP_LED_On(LED3);
          uart_printf("[SYS] IDLE (double-tap)\n");
        }
      }

      if (tap_count > 0 && now - last_tap_tick > 500)
        tap_count = 0;
      #endif

      /* ── Pickup / tilt detection (only when powered on) ── */
      /* Tilt > 45° (az_abs < 700) held for 300ms → picked up / tipped over */
      if (system_state != SYS_OFF)
      {
        int az_abs = az > 0 ? az : -az;

        if (az_abs < 700 && !lift_reported)
        {
          if (lift_tick == 0)
            lift_tick = now;
          else if (now - lift_tick > 300)
          {
            lift_reported = 1;
            lift_tick = 0;
            if (car_command != CMD_STOP)
            {
              car_command = CMD_STOP;
              prev = CMD_STOP;
              Motor_Stop();
            }
            uart_printf("[ACCEL] PICKED UP! az=%d\n", az);
          }
        }
        else
        {
          lift_tick = 0;
          if (az_abs > 900) lift_reported = 0;
        }
      }

      /* ── Collision detection (only while moving) ── */
      {
        static int prev_ax = 0, prev_ay = 0, prev_az = 0;
        if (prev_ax != 0) /* skip first read */
        {
          int moving = (car_command == CMD_FORWARD || car_command == CMD_BACKWARD || car_command == CMD_TURN);
          if (moving && system_state != SYS_OFF)
          {
            int dx = ax - prev_ax;
            int dy = ay - prev_ay;
            if (dx > 1500 || dx < -1500 || dy > 1500 || dy < -1500)
            {
              uart_printf("[ACCEL] COLLISION! dx=%d dy=%d az=%d\n", dx, dy, az);
              car_command = CMD_STOP;
              prev = CMD_STOP;
              Motor_Stop();
            }
          }
        }
        prev_ax = ax; prev_ay = ay; prev_az = az;
      }
    }

    /* ── Ultrasonic + STAT: 200ms ── */
    if (now - last_sonar_tick >= 200)
    {
      last_sonar_tick = now;

      float dist = HCSR04_GetDistance();
      ultrasonic_distance = dist;

      (void)dist;

      if (car_command == CMD_FORWARD && system_state != SYS_OFF
          && dist > 0 && dist < OBSTACLE_THRESHOLD_MM)
      {
        car_command = CMD_STOP;
        prev = CMD_STOP;
        Motor_Stop();
        uart_printf("[CAR] OBSTACLE! STOP (%.0fmm)\n", dist);
      }
    }

    osDelay(10);
  }
}

void DisplayTaskEntry(void *argument)
{
  for (;;)
  {
    const char *state_str =
      system_state==SYS_OFF?"OFF":system_state==SYS_LISTENING?"LISTEN":"IDLE";

    const char *cmd_str =
      car_command==CMD_FORWARD?"forward":car_command==CMD_BACKWARD?"backward":
      car_command==CMD_TURN?"turn":car_command==CMD_START?"start":
      car_command==CMD_STOP?"stop":"---";

    int conf_pct = (int)(confirmed_confidence * 100.0f);
    if (conf_pct < 0) conf_pct = 0;
    if (conf_pct > 100) conf_pct = 100;

    OLED_Clear();

    OLED_Printf(0, 0,  OLED_6X8, "S:%s", state_str);
    OLED_Printf(0, 10, OLED_6X8, "%s %d%%", last_confirmed_label, conf_pct);
    OLED_Printf(0, 20, OLED_6X8, "CMD:%s", cmd_str);
    OLED_Printf(0, 30, OLED_6X8, "DIS:%.0fmm", ultrasonic_distance);

    OLED_Update();

    osDelay(200);
  }
}

void SerialTaskEntry(void *argument)
{
  (void)argument;
  for (;;)
  {
    SerialProtocol_Process();

    CmdFrame_t cmd;
    while (SerialProtocol_GetCmd(&cmd))
    {
      switch (cmd.cmd)
      {
        case CMD_UPLOAD_START:
        {
          if (cmd.payload_len >= 5) {
            if (flash_update_state != FLASH_UPDATE_IDLE) {
              FlashUpdate_Init();
            }
            uint8_t slot = cmd.payload[0];
            uint32_t size = *(uint32_t *)(cmd.payload + 1);
            uint8_t state_before = flash_update_state;
            FlashUpdate_Start(slot, size);
            uint8_t resp[8];
            resp[0] = (flash_update_state == FLASH_UPDATE_RECEIVING) ? 0x01 : 0x00;
            resp[1] = flash_update_state;
            resp[2] = state_before;
            resp[3] = (uint8_t)(size & 0xFF);
            resp[4] = (uint8_t)((size >> 8) & 0xFF);
            resp[5] = slot;
            extern volatile uint8_t flash_update_last_error;
            resp[6] = flash_update_last_error;
            resp[7] = (uint8_t)(MODEL_AREA_SIZE & 0xFF);
            SerialProtocol_SendFrame(0x81, resp, 8);
          }
          break;
        }
        case CMD_UPLOAD_DATA:
        {
          if (flash_update_state == FLASH_UPDATE_RECEIVING && cmd.payload_len > 0) {
            FlashUpdate_ReceiveData(cmd.payload, cmd.payload_len);
            uint8_t ack = 0x01;
            SerialProtocol_SendFrame(0x82, &ack, 1);
          }
          break;
        }
        case CMD_UPLOAD_FINISH:
        {
          if (cmd.payload_len >= 4) {
            uint32_t crc = *(uint32_t *)cmd.payload;
            FlashUpdate_Finish(crc);
          }
          break;
        }
        case CMD_SWITCH_MODEL:
        {
          if (cmd.payload_len >= 1) {
            uint8_t slot = cmd.payload[0];
            bool ok = ModelLoader_LoadSlot(slot);
            if (!ok) ModelLoader_Fallback();
            infer_reset_flag = 1;
            if (g_active_model.source == MODEL_SOURCE_COMPILE)
              BSP_LED_On(LED6);
            else
              BSP_LED_Off(LED6);
            ModelLoader_DumpModelInfo("SWITCH");  /* 切换模型后再次自检 */
            uint8_t resp[2] = { ok ? 0x01 : 0x00, g_active_model.source };
            SerialProtocol_SendFrame(0x84, resp, 2);
          }
          break;
        }
        case CMD_RESET_MODEL:
        {
          ModelLoader_Fallback();
          ModelLoader_DumpModelInfo("RESET");  /* 复位后再次自检 */
          BSP_LED_On(LED6);
          uint8_t ack = 0x01;
          SerialProtocol_SendFrame(0x86, &ack, 1);
          break;
        }
        case CMD_QUERY_STATUS:
        {
          uint8_t resp[32];
          uint8_t idx = 0;
          resp[idx++] = g_active_model.source;
          resp[idx++] = g_active_model.num_classes;
          resp[idx++] = (uint8_t)system_state;
          resp[idx++] = (uint8_t)car_command;
          resp[idx++] = (uint8_t)flash_update_state;
          resp[idx++] = flash_update_progress;
          memcpy(&resp[idx], &ultrasonic_distance, 4);
          idx += 4;
          resp[idx++] = (uint8_t)car_command;
          SerialProtocol_SendFrame(0x85, resp, idx);
          break;
        }
        case CMD_SET_MEL_OUTPUT:
        {
          if (cmd.payload_len >= 1) {
            serial_mel_output_enabled = cmd.payload[0];
          }
          uint8_t ack = serial_mel_output_enabled;
          SerialProtocol_SendFrame(0x87, &ack, 1);
          break;
        }
        case CMD_QUERY_MODEL_INFO:
        {
          uint8_t resp[260];
          uint8_t idx = 0;
          resp[idx++] = g_active_model.source;
          resp[idx++] = g_active_model.num_classes;
          resp[idx++] = (uint8_t)system_state;
          for (int i = 0; i < g_active_model.num_classes && i < MODEL_MAX_CLASSES; i++) {
            for (int j = 0; j < MODEL_CLASS_NAME_LEN; j++) {
              resp[idx++] = g_active_model.class_display[i][j];
            }
          }
          for (int i = 0; i < g_active_model.num_classes && i < MODEL_MAX_CLASSES; i++) {
            resp[idx++] = g_active_model.class_map[i];
          }
          SerialProtocol_SendFrame(0x88, resp, idx);
          break;
        }
        default:
          break;
      }
    }

    osDelay(10);
  }
}

/* USER CODE END 4 */

/**
  * @brief  Period elapsed callback in non blocking mode
  * @note   This function is called  when TIM14 interrupt took place, inside
  * HAL_TIM_IRQHandler(). It makes a direct call to HAL_IncTick() to increment
  * a global variable "uwTick" used as application time base.
  * @param  htim : TIM handle
  * @retval None
  */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  /* USER CODE BEGIN Callback 0 */

  /* USER CODE END Callback 0 */
  if (htim->Instance == TIM14)
  {
    HAL_IncTick();
  }
  /* USER CODE BEGIN Callback 1 */

  /* USER CODE END Callback 1 */
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
