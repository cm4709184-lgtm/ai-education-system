#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include "model_header.h"
#include "model_weights.h"   /* 引入 HAS_SE / SE1_HIDDEN / SE2_HIDDEN / CONF_THRESHOLD / MARGIN_THRESHOLD / NOISE_INDEX */

#define INF_FRAME_WINDOW  32
#define INF_MEL_FILTERS   20

#define CONV1_OUT         32
#define CONV2_OUT         48
#define CONV3_OUT         64

#define POOL1_LEN         (INF_FRAME_WINDOW / 2)
#define POOL2_LEN         (INF_FRAME_WINDOW / 4)

/* SE 配置（从 model_weights.h 引入，未定义则按 0 关闭） */
#ifndef HAS_SE
#define HAS_SE 0
#endif
#if HAS_SE
#ifndef SE1_HIDDEN
#define SE1_HIDDEN  (CONV2_OUT / 4)
#endif
#ifndef SE2_HIDDEN
#define SE2_HIDDEN  (CONV3_OUT / 4)
#endif
#endif

#ifndef QUANTIZED_INFERENCE
#define QUANTIZED_INFERENCE 0
#endif

void Inference_Init(void);
int  Inference_Run(const float *mel_window, float *confidence);
uint8_t Inference_GetNumClasses(void);
void Inference_GetProbs(float *buf, int max_classes);

#endif
