#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <stdint.h>
#include <stdbool.h>
#include "model_header.h"

#define MODEL_SOURCE_COMPILE  0
#define MODEL_SOURCE_SLOT_A   1
#define MODEL_SOURCE_SLOT_B   2

typedef struct {
    uint8_t  source;
    uint8_t  num_classes;
    const float *conv1_weight;  const float *conv1_bias;
    const float *bn1_weight;    const float *bn1_bias;
    const float *bn1_rmean;     const float *bn1_rvar;
    const float *depth1_weight; const float *depth1_bias;
    const float *bn_d1_weight;  const float *bn_d1_bias;
    const float *bn_d1_rmean;   const float *bn_d1_rvar;
    const float *point1_weight; const float *point1_bias;
    const float *bn_p1_weight;  const float *bn_p1_bias;
    const float *bn_p1_rmean;   const float *bn_p1_rvar;
    const float *depth2_weight; const float *depth2_bias;
    const float *bn_d2_weight;  const float *bn_d2_bias;
    const float *bn_d2_rmean;   const float *bn_d2_rvar;
    const float *point2_weight; const float *point2_bias;
    const float *bn_p2_weight;  const float *bn_p2_bias;
    const float *bn_p2_rmean;   const float *bn_p2_rvar;
    const float *fc_weights;
    const float *fc_bias;
    /* ===== SE 通道注意力权重指针（仅当 has_se=1 时有效） ===== */
    const float *se1_fc1_weight; const float *se1_fc1_bias;
    const float *se1_fc2_weight; const float *se1_fc2_bias;
    const float *se2_fc1_weight; const float *se2_fc1_bias;
    const float *se2_fc2_weight; const float *se2_fc2_bias;
    /* ===== 推理配置 ===== */
    uint8_t  has_se;
    uint8_t  noise_index;
    uint8_t  se1_hidden;            /* = CONV2_OUT / se_reduction */
    uint8_t  se2_hidden;            /* = CONV3_OUT / se_reduction */
    float    conf_threshold;        /* 置信度门限 */
    float    margin_threshold;      /* top1-top2 差门限 */
    uint8_t  class_map[MODEL_MAX_CLASSES];
    float    thresholds[MODEL_MAX_CLASSES];
    char     class_names[MODEL_MAX_CLASSES][MODEL_CLASS_NAME_LEN];
    char     class_display[MODEL_MAX_CLASSES][MODEL_CLASS_NAME_LEN];
    bool     valid;
} ActiveModel_t;

extern ActiveModel_t g_active_model;

void ModelLoader_Init(void);
bool ModelLoader_LoadSlot(uint8_t slot);
void ModelLoader_Fallback(void);
bool ModelLoader_WriteSlot(uint8_t slot, const uint8_t *data, uint32_t size);
const char* ModelLoader_GetSourceName(void);

/* 启动自检：打印当前激活模型的 SE 权重指针 + 前几个值，便于排查错位 */
void ModelLoader_DumpModelInfo(const char *tag);

#endif
