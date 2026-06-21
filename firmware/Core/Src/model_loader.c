#include "model_loader.h"
#include "model_weights.h"
#include <string.h>
#include "stm32f4xx_hal.h"

ActiveModel_t g_active_model;

static uint32_t CalcCRC32_Simple(const void *data, uint32_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    uint32_t crc = 0xFFFFFFFF;
    for (uint32_t i = 0; i < len; i++) {
        crc ^= p[i];
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
    }
    return ~crc;
}

static bool ValidateSlot(uint32_t base_addr)
{
    const ModelHeader_t *hdr = (const ModelHeader_t *)base_addr;
    if (hdr->magic != MODEL_MAGIC) return false;
    if (hdr->version != MODEL_VERSION) return false;
    if (hdr->num_classes < 2 || hdr->num_classes > MODEL_MAX_CLASSES) return false;
    if (hdr->gap_dims != MODEL_GAP_DIMS) return false;
    if (hdr->base_weights_offset < 40 || hdr->base_weights_offset > 256) return false;
    /* base 权重上限 = MODEL_AREA_SIZE - 头(40) - FC - meta(550)
     * DS-CNN+SE 最大约 50KB，留余量放 0xF000 (60KB) */
    if (hdr->base_weights_size == 0 || hdr->base_weights_size > 0xF000) return false;
    uint32_t hdr_crc = CalcCRC32_Simple(hdr, offsetof(ModelHeader_t, header_crc));
    if (hdr_crc != hdr->header_crc) return false;
    return true;
}

/* ===== 启动自检：打印 SE 权重指针 + 前几个值 ===== */
void ModelLoader_DumpModelInfo(const char *tag)
{
    extern int printf(const char *fmt, ...);
    printf("\r\n[MLD] %s | num=%u has_se=%u noise_idx=%u src=%u\r\n",
           tag ? tag : "?",
           g_active_model.num_classes,
           g_active_model.has_se,
           g_active_model.noise_index,
           g_active_model.source);
    printf("[MLD] conf=%.3f margin=%.3f se1_h=%u se2_h=%u\r\n",
           g_active_model.conf_threshold,
           g_active_model.margin_threshold,
           g_active_model.se1_hidden,
           g_active_model.se2_hidden);
    if (g_active_model.has_se) {
        /* SE1 fc1 weight：前 4 个值（验证是不是真 SE1 权重指针，没错位） */
        const float *w = g_active_model.se1_fc1_weight;
        const float *b = g_active_model.se1_fc1_bias;
        const float *w2 = g_active_model.se1_fc2_weight;
        const float *b2 = g_active_model.se1_fc2_bias;
        if (w && b && w2 && b2) {
            printf("[MLD] SE1 fc1_w[0..3]=[%.4f,%.4f,%.4f,%.4f] b[0..3]=[%.4f,%.4f,%.4f,%.4f]\r\n",
                   w[0], w[1], w[2], w[3], b[0], b[1], b[2], b[3]);
            printf("[MLD] SE1 fc2_w[0..3]=[%.4f,%.4f,%.4f,%.4f] b[0..3]=[%.4f,%.4f,%.4f,%.4f]\r\n",
                   w2[0], w2[1], w2[2], w2[3], b2[0], b2[1], b2[2], b2[3]);
        } else {
            printf("[MLD] [警告] SE1 权重指针为 NULL\r\n");
        }
        const float *w3 = g_active_model.se2_fc1_weight;
        const float *b3 = g_active_model.se2_fc1_bias;
        const float *w4 = g_active_model.se2_fc2_weight;
        const float *b4 = g_active_model.se2_fc2_bias;
        if (w3 && b3 && w4 && b4) {
            printf("[MLD] SE2 fc1_w[0..3]=[%.4f,%.4f,%.4f,%.4f] b[0..3]=[%.4f,%.4f,%.4f,%.4f]\r\n",
                   w3[0], w3[1], w3[2], w3[3], b3[0], b3[1], b3[2], b3[3]);
            printf("[MLD] SE2 fc2_w[0..3]=[%.4f,%.4f,%.4f,%.4f] b[0..3]=[%.4f,%.4f,%.4f,%.4f]\r\n",
                   w4[0], w4[1], w4[2], w4[3], b4[0], b4[1], b4[2], b4[3]);
        } else {
            printf("[MLD] [警告] SE2 权重指针为 NULL\r\n");
        }
        /* 关键自检：SE1 fc1 权重 + SE1 fc2 权重 + SE1 fc1 bias 三者地址必须不同
         * （如果 SE 错位了，SE2_fc1_weight 会等于 SE1_fc2_weight，地址就相同） */
        if ((const void *)g_active_model.se2_fc1_weight == (const void *)g_active_model.se1_fc2_weight) {
            printf("[MLD] [严重] SE2_fc1_weight 地址 == SE1_fc2_weight 地址，权重错位！\r\n");
        }
    }
}

static void SetCompiledWeights(void)
{
    g_active_model.conv1_weight  = conv1_weight;
    g_active_model.conv1_bias    = conv1_bias;
    g_active_model.bn1_weight    = bn1_weight;
    g_active_model.bn1_bias      = bn1_bias;
    g_active_model.bn1_rmean     = bn1_running_mean;
    g_active_model.bn1_rvar      = bn1_running_var;
    g_active_model.depth1_weight = depth1_weight;
    g_active_model.depth1_bias   = depth1_bias;
    g_active_model.bn_d1_weight  = bn_d1_weight;
    g_active_model.bn_d1_bias    = bn_d1_bias;
    g_active_model.bn_d1_rmean   = bn_d1_running_mean;
    g_active_model.bn_d1_rvar    = bn_d1_running_var;
    g_active_model.point1_weight = point1_weight;
    g_active_model.point1_bias   = point1_bias;
    g_active_model.bn_p1_weight  = bn_p1_weight;
    g_active_model.bn_p1_bias    = bn_p1_bias;
    g_active_model.bn_p1_rmean   = bn_p1_running_mean;
    g_active_model.bn_p1_rvar    = bn_p1_running_var;
    g_active_model.depth2_weight = depth2_weight;
    g_active_model.depth2_bias   = depth2_bias;
    g_active_model.bn_d2_weight  = bn_d2_weight;
    g_active_model.bn_d2_bias    = bn_d2_bias;
    g_active_model.bn_d2_rmean   = bn_d2_running_mean;
    g_active_model.bn_d2_rvar    = bn_d2_running_var;
    g_active_model.point2_weight = point2_weight;
    g_active_model.point2_bias   = point2_bias;
    g_active_model.bn_p2_weight  = bn_p2_weight;
    g_active_model.bn_p2_bias    = bn_p2_bias;
    g_active_model.bn_p2_rmean   = bn_p2_running_mean;
    g_active_model.bn_p2_rvar    = bn_p2_running_var;
    g_active_model.fc_weights    = fc_weight;
    g_active_model.fc_bias       = fc_bias;
    /* SE 权重（来自 model_weights.h 的编译期常量） */
#if HAS_SE
    g_active_model.se1_fc1_weight = se1_fc1_weight;
    g_active_model.se1_fc1_bias   = se1_fc1_bias;
    g_active_model.se1_fc2_weight = se1_fc2_weight;
    g_active_model.se1_fc2_bias   = se1_fc2_bias;
    g_active_model.se2_fc1_weight = se2_fc1_weight;
    g_active_model.se2_fc1_bias   = se2_fc1_bias;
    g_active_model.se2_fc2_weight = se2_fc2_weight;
    g_active_model.se2_fc2_bias   = se2_fc2_bias;
    g_active_model.has_se         = 1;
    g_active_model.se1_hidden     = (uint8_t)(SE1_HIDDEN);
    g_active_model.se2_hidden     = (uint8_t)(SE2_HIDDEN);
#else
    g_active_model.se1_fc1_weight = NULL;
    g_active_model.se2_fc1_weight = NULL;
    g_active_model.has_se         = 0;
    g_active_model.se1_hidden     = 0;
    g_active_model.se2_hidden     = 0;
#endif
    /* 推理门限（来自 model_weights.h） */
#ifdef CONF_THRESHOLD
    g_active_model.conf_threshold   = (float)(CONF_THRESHOLD);
    g_active_model.margin_threshold = (float)(MARGIN_THRESHOLD);
#else
    g_active_model.conf_threshold   = 0.0f;
    g_active_model.margin_threshold = 0.0f;
#endif
#ifdef NOISE_INDEX
    g_active_model.noise_index = (uint8_t)(NOISE_INDEX);
#else
    g_active_model.noise_index = 0xFF;
#endif
}

static void SetFlashWeights(uint32_t base, const ModelHeader_t *hdr, const ModelMeta_t *meta)
{
    const float *w = (const float *)(base + hdr->base_weights_offset);
    uint32_t off = 0;

    g_active_model.conv1_weight  = w + off; off += CONV1_OUT * 20 * 5;
    g_active_model.conv1_bias    = w + off; off += CONV1_OUT;
    g_active_model.bn1_weight    = w + off; off += CONV1_OUT;
    g_active_model.bn1_bias      = w + off; off += CONV1_OUT;
    g_active_model.bn1_rmean     = w + off; off += CONV1_OUT;
    g_active_model.bn1_rvar      = w + off; off += CONV1_OUT;
    g_active_model.depth1_weight = w + off; off += CONV1_OUT * 3;
    g_active_model.depth1_bias   = w + off; off += CONV1_OUT;
    g_active_model.bn_d1_weight  = w + off; off += CONV1_OUT;
    g_active_model.bn_d1_bias    = w + off; off += CONV1_OUT;
    g_active_model.bn_d1_rmean   = w + off; off += CONV1_OUT;
    g_active_model.bn_d1_rvar    = w + off; off += CONV1_OUT;
    g_active_model.point1_weight = w + off; off += CONV2_OUT * CONV1_OUT;
    g_active_model.point1_bias   = w + off; off += CONV2_OUT;
    g_active_model.bn_p1_weight  = w + off; off += CONV2_OUT;
    g_active_model.bn_p1_bias    = w + off; off += CONV2_OUT;
    g_active_model.bn_p1_rmean   = w + off; off += CONV2_OUT;
    g_active_model.bn_p1_rvar    = w + off; off += CONV2_OUT;
    /* SE1（按 Flash 二进制中 weight_order 顺序） */
    g_active_model.se1_fc1_weight = w + off; off += CONV2_OUT * meta->se1_hidden;
    g_active_model.se1_fc1_bias   = w + off; off += meta->se1_hidden;
    g_active_model.se1_fc2_weight = w + off; off += meta->se1_hidden * CONV2_OUT;
    g_active_model.se1_fc2_bias   = w + off; off += CONV2_OUT;

    g_active_model.depth2_weight = w + off; off += CONV2_OUT * 3;
    g_active_model.depth2_bias   = w + off; off += CONV2_OUT;
    g_active_model.bn_d2_weight  = w + off; off += CONV2_OUT;
    g_active_model.bn_d2_bias    = w + off; off += CONV2_OUT;
    g_active_model.bn_d2_rmean   = w + off; off += CONV2_OUT;
    g_active_model.bn_d2_rvar    = w + off; off += CONV2_OUT;
    g_active_model.point2_weight = w + off; off += CONV3_OUT * CONV2_OUT;
    g_active_model.point2_bias   = w + off; off += CONV3_OUT;
    g_active_model.bn_p2_weight  = w + off; off += CONV3_OUT;
    g_active_model.bn_p2_bias    = w + off; off += CONV3_OUT;
    g_active_model.bn_p2_rmean   = w + off; off += CONV3_OUT;
    g_active_model.bn_p2_rvar    = w + off; off += CONV3_OUT;
    /* SE2 */
    g_active_model.se2_fc1_weight = w + off; off += CONV3_OUT * meta->se2_hidden;
    g_active_model.se2_fc1_bias   = w + off; off += meta->se2_hidden;
    g_active_model.se2_fc2_weight = w + off; off += meta->se2_hidden * CONV3_OUT;
    g_active_model.se2_fc2_bias   = w + off; off += CONV3_OUT;

    g_active_model.fc_weights = (const float *)(base + hdr->fc_weights_offset);
    g_active_model.fc_bias    = (const float *)(base + hdr->fc_bias_offset);
    /* SE 配置 */
    g_active_model.has_se         = meta->has_se;
    g_active_model.se1_hidden     = meta->se1_hidden;
    g_active_model.se2_hidden     = meta->se2_hidden;
    g_active_model.conf_threshold   = meta->conf_threshold;
    g_active_model.margin_threshold = meta->margin_threshold;
    g_active_model.noise_index      = meta->noise_index;
}

void ModelLoader_Init(void)
{
    memset(&g_active_model, 0, sizeof(g_active_model));
    g_active_model.source = MODEL_SOURCE_COMPILE;
    g_active_model.num_classes = NUM_CLASSES;
    g_active_model.valid = true;

    for (int i = 0; i < NUM_CLASSES && i < MODEL_MAX_CLASSES; i++) {
        strncpy(g_active_model.class_names[i], class_labels[i], MODEL_CLASS_NAME_LEN - 1);
        g_active_model.class_names[i][MODEL_CLASS_NAME_LEN - 1] = '\0';
        strncpy(g_active_model.class_display[i], class_display[i], MODEL_CLASS_NAME_LEN - 1);
        g_active_model.class_display[i][MODEL_CLASS_NAME_LEN - 1] = '\0';
        g_active_model.class_map[i] = (uint8_t)class_to_cmd[i];
        g_active_model.thresholds[i] = 0.35f;
    }

    SetCompiledWeights();

    if (ValidateSlot(MODEL_A_ADDR)) {
        ModelLoader_LoadSlot(MODEL_SOURCE_SLOT_A);
    } else if (ValidateSlot(MODEL_B_ADDR)) {
        ModelLoader_LoadSlot(MODEL_SOURCE_SLOT_B);
    }
}

bool ModelLoader_LoadSlot(uint8_t slot)
{
    uint32_t base = (slot == MODEL_SOURCE_SLOT_A) ? MODEL_A_ADDR : MODEL_B_ADDR;
    if (!ValidateSlot(base)) return false;

    const ModelHeader_t *hdr = (const ModelHeader_t *)base;
    const ModelMeta_t *meta = (const ModelMeta_t *)(base + hdr->meta_offset);

    uint32_t meta_crc = CalcCRC32_Simple(meta, offsetof(ModelMeta_t, meta_crc));
    if (meta_crc != meta->meta_crc) return false;

    __disable_irq();
    g_active_model.source = slot;
    g_active_model.num_classes = hdr->num_classes;

    SetFlashWeights(base, hdr, meta);

    for (int i = 0; i < hdr->num_classes && i < MODEL_MAX_CLASSES; i++) {
        strncpy(g_active_model.class_names[i], meta->class_names[i], MODEL_CLASS_NAME_LEN - 1);
        g_active_model.class_names[i][MODEL_CLASS_NAME_LEN - 1] = '\0';
        strncpy(g_active_model.class_display[i], meta->class_display[i], MODEL_CLASS_NAME_LEN - 1);
        g_active_model.class_display[i][MODEL_CLASS_NAME_LEN - 1] = '\0';
        g_active_model.class_map[i] = meta->class_map[i];
        g_active_model.thresholds[i] = meta->thresholds[i];
    }
    g_active_model.valid = true;
    __enable_irq();

    return true;
}

void ModelLoader_Fallback(void)
{
    __disable_irq();
    g_active_model.source = MODEL_SOURCE_COMPILE;
    g_active_model.num_classes = NUM_CLASSES;

    SetCompiledWeights();

    for (int i = 0; i < NUM_CLASSES && i < MODEL_MAX_CLASSES; i++) {
        strncpy(g_active_model.class_names[i], class_labels[i], MODEL_CLASS_NAME_LEN - 1);
        g_active_model.class_names[i][MODEL_CLASS_NAME_LEN - 1] = '\0';
        strncpy(g_active_model.class_display[i], class_display[i], MODEL_CLASS_NAME_LEN - 1);
        g_active_model.class_display[i][MODEL_CLASS_NAME_LEN - 1] = '\0';
        g_active_model.class_map[i] = (uint8_t)class_to_cmd[i];
        g_active_model.thresholds[i] = 0.35f;
    }
    g_active_model.valid = true;
    __enable_irq();
}

bool ModelLoader_WriteSlot(uint8_t slot, const uint8_t *data, uint32_t size)
{
    uint32_t base = (slot == MODEL_SOURCE_SLOT_A) ? MODEL_A_ADDR : MODEL_B_ADDR;
    uint32_t sector = (slot == MODEL_SOURCE_SLOT_A) ? MODEL_A_SECTOR : MODEL_B_SECTOR;

    if (size > MODEL_AREA_SIZE) return false;

    HAL_FLASH_Unlock();

    FLASH_EraseInitTypeDef erase;
    erase.TypeErase = FLASH_TYPEERASE_SECTORS;
    erase.Sector = sector;
    erase.NbSectors = 1;
    erase.VoltageRange = FLASH_VOLTAGE_RANGE_3;
    uint32_t sectorError;
    if (HAL_FLASHEx_Erase(&erase, &sectorError) != HAL_OK) {
        HAL_FLASH_Lock();
        return false;
    }

    for (uint32_t i = 0; i < size; i += 4) {
        uint32_t word = 0xFFFFFFFF;
        uint32_t remain = size - i;
        if (remain >= 4) {
            word = *(const uint32_t *)(data + i);
        } else {
            memcpy(&word, data + i, remain);
        }
        if (HAL_FLASH_Program(FLASH_TYPEPROGRAM_WORD, base + i, word) != HAL_OK) {
            HAL_FLASH_Lock();
            return false;
        }
    }

    HAL_FLASH_Lock();
    return true;
}

const char* ModelLoader_GetSourceName(void)
{
    switch (g_active_model.source) {
        case MODEL_SOURCE_SLOT_A: return "Slot A";
        case MODEL_SOURCE_SLOT_B: return "Slot B";
        default: return "Base";
    }
}
