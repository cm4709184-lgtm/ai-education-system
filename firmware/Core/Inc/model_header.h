#ifndef MODEL_HEADER_H
#define MODEL_HEADER_H

#include <stdint.h>

#define MODEL_MAGIC           0xDEADBEEF
#define MODEL_VERSION         1
#define MODEL_MAX_CLASSES     10
#define MODEL_CLASS_NAME_LEN  24
#define MODEL_GAP_DIMS        64

#define MODEL_A_ADDR          0x080E0000
#define MODEL_B_ADDR          0x080C0000
#define MODEL_A_SECTOR        FLASH_SECTOR_11
#define MODEL_B_SECTOR        FLASH_SECTOR_10

/* 模型区域大小：升级到 DS-CNN + SE 后 base 权重 ~50KB + FC + meta = ~53KB
 * 原来 0xA000(40KB) 装不下，扩到 0xE000(56KB)，仍小于 sector 的 128KB */
#define MODEL_AREA_SIZE       0xE000

#define MODEL_STATUS_INVALID   0
#define MODEL_STATUS_UPDATING  1
#define MODEL_STATUS_VALID     2

#define PARAM_LOG_ADDR         0x08048000
#define PARAM_LOG_SECTOR       FLASH_SECTOR_6
#define PARAM_LOG_SIZE         0x4000

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t  version;
    uint8_t  num_classes;
    uint8_t  gap_dims;
    uint8_t  reserved0;
    uint32_t base_weights_offset;
    uint32_t base_weights_size;
    uint32_t fc_weights_offset;
    uint32_t fc_weights_size;
    uint32_t fc_bias_offset;
    uint32_t fc_bias_size;
    uint32_t meta_offset;
    uint32_t header_crc;
} ModelHeader_t;

typedef struct __attribute__((packed)) {
    char     class_names[MODEL_MAX_CLASSES][MODEL_CLASS_NAME_LEN];
    char     class_display[MODEL_MAX_CLASSES][MODEL_CLASS_NAME_LEN];
    uint8_t  class_map[MODEL_MAX_CLASSES];
    float    thresholds[MODEL_MAX_CLASSES];
    /* ===== 新增：SE 注意力 + 推理门限配置 ===== */
    float    conf_threshold;        /* 拒识门限1：top1 置信度低于此值判 noise */
    float    margin_threshold;      /* 拒识门限2：top1-top2 差小于此值判 noise */
    uint8_t  noise_index;           /* noise 类在 class_labels 中的索引 */
    uint8_t  has_se;                /* 是否包含 SE 通道注意力 */
    uint8_t  se_reduction;          /* SE 压缩比 r */
    uint8_t  se1_hidden;            /* SE1 隐层维度 = CONV2_OUT / r */
    uint8_t  se2_hidden;            /* SE2 隐层维度 = CONV3_OUT / r */
    uint8_t  reserved[3];           /* 4 字节对齐填充 */
    uint32_t meta_crc;
} ModelMeta_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t  active_slot;
    uint8_t  slot_a_status;
    uint8_t  slot_b_status;
    uint8_t  reserved0;
    uint32_t slot_a_crc;
    uint32_t slot_b_crc;
    uint32_t run_count;
    uint32_t reserved[4];
    uint32_t param_crc;
} ParamLog_t;

uint32_t ModelHeader_CalcCRC32(const ModelHeader_t *hdr);
uint32_t ModelMeta_CalcCRC32(const ModelMeta_t *meta);
uint32_t ParamLog_CalcCRC32(const ParamLog_t *log);
uint32_t CalcCRC32(const void *data, uint32_t len);

#endif
