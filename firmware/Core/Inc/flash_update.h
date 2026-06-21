#ifndef FLASH_UPDATE_H
#define FLASH_UPDATE_H

#include <stdint.h>
#include <stdbool.h>

#define FLASH_UPLOAD_BUF_SIZE  2048

typedef enum {
    FLASH_UPDATE_IDLE,
    FLASH_UPDATE_RECEIVING,
    FLASH_UPDATE_VALIDATING,
    FLASH_UPDATE_WRITING,
    FLASH_UPDATE_DONE,
    FLASH_UPDATE_ERROR
} FlashUpdateState_t;

extern volatile FlashUpdateState_t flash_update_state;
extern volatile uint8_t  flash_update_progress;
extern volatile uint32_t flash_update_expected_size;
extern volatile uint32_t flash_update_received_size;
extern volatile uint8_t  flash_update_target_slot;

void FlashUpdate_Init(void);
void FlashUpdate_Start(uint8_t slot, uint32_t size);
void FlashUpdate_ReceiveData(const uint8_t *data, uint16_t len);
void FlashUpdate_Finish(uint32_t crc32);
void FlashUpdateTaskEntry(void *argument);

#endif
