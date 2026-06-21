#include "serial_protocol.h"
#include "usart.h"
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

static SerialRxParser_t rx_parser;
static SerialRingBuf_t rx_ring;
static CmdFrame_t cmd_queue[8];
static volatile uint8_t cmd_head = 0;
static volatile uint8_t cmd_tail = 0;

volatile uint8_t serial_dma_rx_buf[SERIAL_RX_BUF_SIZE];

static uint8_t CRC8_Calc(const uint8_t *data, uint32_t len)
{
    uint8_t crc = 0;
    for (uint32_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0x8C & (-(crc & 1)));
    }
    return crc;
}

void SerialProtocol_Init(void)
{
    memset(&rx_parser, 0, sizeof(rx_parser));
    memset(&rx_ring, 0, sizeof(rx_ring));
    cmd_head = 0;
    cmd_tail = 0;
}

static void RingBuf_Push(uint8_t byte)
{
    uint32_t next = (rx_ring.head + 1) % SERIAL_RX_BUF_SIZE;
    if (next != rx_ring.tail) {
        rx_ring.buf[rx_ring.head] = byte;
        rx_ring.head = next;
    }
}

static bool RingBuf_Pop(uint8_t *byte)
{
    if (rx_ring.head == rx_ring.tail) return false;
    *byte = rx_ring.buf[rx_ring.tail];
    rx_ring.tail = (rx_ring.tail + 1) % SERIAL_RX_BUF_SIZE;
    return true;
}

static uint8_t HexCharToNibble(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
}

static void ParseUpstreamByte(uint8_t byte)
{
    switch (rx_parser.state) {
    case RX_STATE_IDLE:
        if (byte == SERIAL_FRAME_START) {
            rx_parser.state = RX_STATE_TYPE;
            rx_parser.escaped = false;
        }
        break;
    case RX_STATE_TYPE:
        rx_parser.type = byte;
        rx_parser.state = RX_STATE_LEN;
        break;
    case RX_STATE_LEN:
        rx_parser.len = byte;
        rx_parser.payload_idx = 0;
        rx_parser.crc = 0;
        rx_parser.state = (byte > 0) ? RX_STATE_PAYLOAD : RX_STATE_CRC;
        break;
    case RX_STATE_PAYLOAD:
        rx_parser.payload[rx_parser.payload_idx++] = byte;
        if (rx_parser.payload_idx >= rx_parser.len)
            rx_parser.state = RX_STATE_CRC;
        break;
    case RX_STATE_CRC:
        rx_parser.crc = byte;
        rx_parser.state = RX_STATE_END;
        break;
    case RX_STATE_END:
        if (byte == SERIAL_FRAME_END) {
            uint8_t check_buf[2 + 256];
            check_buf[0] = rx_parser.type;
            check_buf[1] = rx_parser.len;
            memcpy(check_buf + 2, rx_parser.payload, rx_parser.len);
            if (CRC8_Calc(check_buf, 2 + rx_parser.len) == rx_parser.crc) {
                // valid frame received - handled upstream
            }
        }
        rx_parser.state = RX_STATE_IDLE;
        break;
    }
}

static void ParseCommandByte(uint8_t byte)
{
    static uint8_t cmd_buf[260];
    static uint8_t cmd_state = 0;
    static uint8_t cmd_idx = 0;
    static uint8_t cmd_len = 0;

    switch (cmd_state) {
    case 0:
        if (byte == SERIAL_CMD_START) cmd_state = 1;
        break;
    case 1:
        if (byte == SERIAL_CMD_SECOND) cmd_state = 2;
        else cmd_state = 0;
        break;
    case 2:
        cmd_buf[0] = byte;
        cmd_idx = 1;
        cmd_state = 3;
        break;
    case 3:
        cmd_len = byte;
        cmd_buf[1] = byte;
        cmd_idx = 2;
        cmd_state = (cmd_len > 0) ? 4 : 7;
        break;
    case 4:
        cmd_buf[cmd_idx++] = byte;
        if (cmd_idx >= (uint8_t)(cmd_len + 2)) cmd_state = 7;
        break;
    case 7:
        cmd_buf[cmd_idx++] = byte;
        if (cmd_idx >= (uint8_t)(cmd_len + 2 + 4)) {
            uint8_t next = (cmd_head + 1) % 8;
            if (next != cmd_tail) {
                cmd_queue[cmd_head].cmd = cmd_buf[0];
                cmd_queue[cmd_head].payload_len = cmd_len;
                if (cmd_len > 0)
                    memcpy(cmd_queue[cmd_head].payload, cmd_buf + 2, cmd_len);
                memcpy(&cmd_queue[cmd_head].crc32, cmd_buf + cmd_len + 2, 4);
                cmd_head = next;
            }
            cmd_state = 0;
        }
        break;
    }
}

void SerialProtocol_RxByte(uint8_t byte)
{
    RingBuf_Push(byte);
}

bool SerialProtocol_GetCmd(CmdFrame_t *cmd)
{
    if (cmd_head == cmd_tail) return false;
    *cmd = cmd_queue[cmd_tail];
    cmd_tail = (cmd_tail + 1) % 8;
    return true;
}

static void SendByte_Escaped(uint8_t byte, uint8_t *buf, uint8_t *idx)
{
    if (byte == SERIAL_FRAME_START || byte == SERIAL_FRAME_END || byte == SERIAL_ESCAPE) {
        buf[(*idx)++] = SERIAL_ESCAPE;
        buf[(*idx)++] = byte ^ SERIAL_ESCAPE_XOR;
    } else {
        buf[(*idx)++] = byte;
    }
}

void SerialProtocol_SendFrame(uint8_t type, const uint8_t *payload, uint8_t len)
{
    uint8_t frame[520];
    uint8_t idx = 0;

    frame[idx++] = SERIAL_FRAME_START;

    uint8_t check_buf[2 + 256];
    check_buf[0] = type;
    check_buf[1] = len;
    memcpy(check_buf + 2, payload, len);
    uint8_t crc = CRC8_Calc(check_buf, 2 + len);

    SendByte_Escaped(type, frame, &idx);
    SendByte_Escaped(len, frame, &idx);
    for (uint8_t i = 0; i < len; i++)
        SendByte_Escaped(payload[i], frame, &idx);
    SendByte_Escaped(crc, frame, &idx);
    frame[idx++] = SERIAL_FRAME_END;

    HAL_UART_Transmit(&huart2, frame, idx, HAL_MAX_DELAY);
}

void SerialProtocol_SendLog(const char *fmt, ...)
{
    char buf[256];
    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    if (n > 0) {
        if (n > 255) n = 255;
        SerialProtocol_SendFrame(SERIAL_TYPE_LOG, (uint8_t *)buf, (uint8_t)n);
    }
}

void SerialProtocol_Process(void)
{
    uint8_t byte;
    while (RingBuf_Pop(&byte)) {
        ParseCommandByte(byte);
    }
}

void SerialProtocol_SendMelFrame(const float *mel_data, uint8_t dims)
{
    uint8_t buf[20 * 4];
    for (uint8_t i = 0; i < dims; i++) {
        memcpy(&buf[i * 4], &mel_data[i], 4);
    }
    SerialProtocol_SendFrame(SERIAL_TYPE_MEL, buf, dims * 4);
}

void SerialProtocol_SendInferResult(const float *probs, uint8_t num_classes,
                                     uint32_t inference_us, float agc_gain,
                                     uint32_t free_ram)
{
    uint8_t buf[64];
    uint8_t idx = 0;
    for (uint8_t i = 0; i < num_classes && idx + 4 <= 64; i++) {
        int16_t p = (int16_t)(probs[i] * 10000);
        buf[idx++] = (uint8_t)(p & 0xFF);
        buf[idx++] = (uint8_t)(p >> 8);
    }
    if (idx + 4 <= 64) {
        buf[idx++] = (uint8_t)(inference_us & 0xFF);
        buf[idx++] = (uint8_t)((inference_us >> 8) & 0xFF);
    }
    if (idx + 2 <= 64) {
        int16_t agc = (int16_t)(agc_gain * 100);
        buf[idx++] = (uint8_t)(agc & 0xFF);
        buf[idx++] = (uint8_t)(agc >> 8);
    }
    if (idx + 4 <= 64) {
        buf[idx++] = (uint8_t)(free_ram & 0xFF);
        buf[idx++] = (uint8_t)((free_ram >> 8) & 0xFF);
        buf[idx++] = (uint8_t)((free_ram >> 16) & 0xFF);
        buf[idx++] = (uint8_t)((free_ram >> 24) & 0xFF);
    }
    SerialProtocol_SendFrame(SERIAL_TYPE_INFER, buf, idx);
}

void SerialProtocol_SendStatus(uint8_t state, uint8_t cmd, float distance,
                                int16_t accel_x, int16_t accel_y, int16_t accel_z)
{
    uint8_t buf[32];
    uint8_t idx = 0;
    buf[idx++] = state;
    buf[idx++] = cmd;
    int16_t dist = (int16_t)distance;
    buf[idx++] = (uint8_t)(dist & 0xFF);
    buf[idx++] = (uint8_t)(dist >> 8);
    buf[idx++] = (uint8_t)(accel_x & 0xFF);
    buf[idx++] = (uint8_t)(accel_x >> 8);
    buf[idx++] = (uint8_t)(accel_y & 0xFF);
    buf[idx++] = (uint8_t)(accel_y >> 8);
    buf[idx++] = (uint8_t)(accel_z & 0xFF);
    buf[idx++] = (uint8_t)(accel_z >> 8);
    SerialProtocol_SendFrame(SERIAL_TYPE_STATUS, buf, idx);
}
