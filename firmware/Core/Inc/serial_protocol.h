#ifndef SERIAL_PROTOCOL_H
#define SERIAL_PROTOCOL_H

#include <stdint.h>
#include <stdbool.h>

#define SERIAL_FRAME_START    0x7E
#define SERIAL_FRAME_END      0x7F
#define SERIAL_ESCAPE         0x7D
#define SERIAL_ESCAPE_XOR     0x20

#define SERIAL_RX_BUF_SIZE    4096
#define SERIAL_TX_BUF_SIZE    512

#define SERIAL_TYPE_MEL       0x01
#define SERIAL_TYPE_INFER     0x02
#define SERIAL_TYPE_STATUS    0x03
#define SERIAL_TYPE_LOG       0x04

#define SERIAL_CMD_START      0xAA
#define SERIAL_CMD_SECOND     0x55

#define CMD_UPLOAD_START      0x01
#define CMD_UPLOAD_DATA       0x02
#define CMD_UPLOAD_FINISH     0x03
#define CMD_SWITCH_MODEL      0x04
#define CMD_QUERY_STATUS      0x05
#define CMD_RESET_MODEL       0x06
#define CMD_SET_MEL_OUTPUT    0x07
#define CMD_QUERY_MODEL_INFO  0x08

typedef enum {
    RX_STATE_IDLE,
    RX_STATE_TYPE,
    RX_STATE_LEN,
    RX_STATE_PAYLOAD,
    RX_STATE_CRC,
    RX_STATE_END
} SerialRxState_t;

typedef struct {
    SerialRxState_t state;
    uint8_t  type;
    uint8_t  len;
    uint8_t  payload[256];
    uint8_t  payload_idx;
    uint8_t  crc;
    bool     escaped;
} SerialRxParser_t;

typedef struct {
    uint8_t buf[SERIAL_RX_BUF_SIZE];
    volatile uint32_t head;
    volatile uint32_t tail;
} SerialRingBuf_t;

typedef struct {
    uint8_t  cmd;
    uint8_t  payload[256];
    uint16_t payload_len;
    uint32_t crc32;
} CmdFrame_t;

void SerialProtocol_Init(void);
void SerialProtocol_RxByte(uint8_t byte);
bool SerialProtocol_GetCmd(CmdFrame_t *cmd);

void SerialProtocol_SendFrame(uint8_t type, const uint8_t *payload, uint8_t len);
void SerialProtocol_SendMelFrame(const float *mel_data, uint8_t dims);
void SerialProtocol_SendInferResult(const float *probs, uint8_t num_classes,
                                     uint32_t inference_us, float agc_gain,
                                     uint32_t free_ram);
void SerialProtocol_SendStatus(uint8_t state, uint8_t cmd, float distance,
                                int16_t accel_x, int16_t accel_y, int16_t accel_z);
void SerialProtocol_SendLog(const char *fmt, ...);

void SerialProtocol_Process(void);

extern volatile uint8_t serial_dma_rx_buf[SERIAL_RX_BUF_SIZE];

#endif
