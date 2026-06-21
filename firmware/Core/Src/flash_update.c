#include "flash_update.h"
#include "model_header.h"
#include "model_loader.h"
#include <string.h>
#include "stm32f4xx_hal.h"
#include "cmsis_os.h"
#include "OLED.h"

volatile FlashUpdateState_t flash_update_state = FLASH_UPDATE_IDLE;
volatile uint8_t  flash_update_progress = 0;
volatile uint32_t flash_update_expected_size = 0;
volatile uint32_t flash_update_received_size = 0;
volatile uint8_t  flash_update_target_slot = 0;
volatile uint8_t  flash_update_last_error = 0;

static uint8_t chunk_buf[FLASH_UPLOAD_BUF_SIZE];
static volatile uint32_t chunk_write_pos = 0;
static volatile uint32_t flash_write_offset = 0;
static volatile uint32_t target_base_addr = 0;
static volatile bool upload_complete = false;
static volatile uint32_t upload_crc32 = 0;

void FlashUpdate_Init(void)
{
    flash_update_state = FLASH_UPDATE_IDLE;
    flash_update_progress = 0;
    flash_update_expected_size = 0;
    flash_update_received_size = 0;
    chunk_write_pos = 0;
    flash_write_offset = 0;
    upload_complete = false;
}

static bool FlashWriteChunk(uint32_t addr, const uint8_t *data, uint32_t len)
{
    for (uint32_t i = 0; i < len; i += 4) {
        uint32_t word = 0xFFFFFFFF;
        uint32_t remain = len - i;
        if (remain >= 4) {
            word = *(const uint32_t *)(data + i);
        } else {
            memcpy(&word, data + i, remain);
        }
        if (HAL_FLASH_Program(FLASH_TYPEPROGRAM_WORD, addr + i, word) != HAL_OK)
            return false;
    }
    return true;
}

void FlashUpdate_Start(uint8_t slot, uint32_t size)
{
    if (flash_update_state != FLASH_UPDATE_IDLE) return;
    if (size == 0 || size > MODEL_AREA_SIZE) return;

    flash_update_target_slot = slot;
    flash_update_expected_size = size;
    flash_update_received_size = 0;
    flash_update_progress = 0;
    chunk_write_pos = 0;
    flash_write_offset = 0;
    upload_complete = false;
    target_base_addr = (slot == MODEL_SOURCE_SLOT_A) ? MODEL_A_ADDR : MODEL_B_ADDR;

    uint32_t sector = (slot == MODEL_SOURCE_SLOT_A) ? MODEL_A_SECTOR : MODEL_B_SECTOR;
    HAL_FLASH_Unlock();
    __HAL_FLASH_CLEAR_FLAG(FLASH_FLAG_EOP | FLASH_FLAG_OPERR | FLASH_FLAG_WRPERR |
                            FLASH_FLAG_PGAERR | FLASH_FLAG_PGPERR | FLASH_FLAG_PGSERR);
    FLASH_EraseInitTypeDef erase;
    erase.TypeErase = FLASH_TYPEERASE_SECTORS;
    erase.Sector = sector;
    erase.NbSectors = 1;
    erase.VoltageRange = FLASH_VOLTAGE_RANGE_3;
    uint32_t sectorError = 0xFFFFFFFF;
    HAL_StatusTypeDef ret = HAL_FLASHEx_Erase(&erase, &sectorError);
    HAL_FLASH_Lock();

    if (ret != HAL_OK) {
        flash_update_last_error = (uint8_t)ret;
        flash_update_state = FLASH_UPDATE_IDLE;
        return;
    }
    flash_update_last_error = 0;

    flash_update_state = FLASH_UPDATE_RECEIVING;
}

void FlashUpdate_ReceiveData(const uint8_t *data, uint16_t len)
{
    if (flash_update_state != FLASH_UPDATE_RECEIVING) return;
    if (flash_update_received_size + len > flash_update_expected_size) return;

    uint16_t offset = 0;
    while (offset < len) {
        uint16_t space = FLASH_UPLOAD_BUF_SIZE - chunk_write_pos;
        uint16_t copy = (len - offset < space) ? (len - offset) : space;
        memcpy(chunk_buf + chunk_write_pos, data + offset, copy);
        chunk_write_pos += copy;
        offset += copy;

        if (chunk_write_pos >= FLASH_UPLOAD_BUF_SIZE) {
            HAL_FLASH_Unlock();
            bool ok = FlashWriteChunk(target_base_addr + flash_write_offset,
                                       chunk_buf, FLASH_UPLOAD_BUF_SIZE);
            HAL_FLASH_Lock();
            if (!ok) {
                flash_update_state = FLASH_UPDATE_ERROR;
                return;
            }
            flash_write_offset += FLASH_UPLOAD_BUF_SIZE;
            chunk_write_pos = 0;
        }
    }

    flash_update_received_size += len;
    flash_update_progress = (uint8_t)((flash_update_received_size * 100) / flash_update_expected_size);
}

void FlashUpdate_Finish(uint32_t crc32)
{
    if (flash_update_state != FLASH_UPDATE_RECEIVING) return;
    if (flash_update_received_size != flash_update_expected_size) {
        flash_update_state = FLASH_UPDATE_ERROR;
        return;
    }

    if (chunk_write_pos > 0) {
        HAL_FLASH_Unlock();
        bool ok = FlashWriteChunk(target_base_addr + flash_write_offset,
                                   chunk_buf, chunk_write_pos);
        HAL_FLASH_Lock();
        if (!ok) {
            flash_update_state = FLASH_UPDATE_ERROR;
            return;
        }
    }

    upload_crc32 = crc32;
    upload_complete = true;
    flash_update_state = FLASH_UPDATE_VALIDATING;
}

void FlashUpdateTaskEntry(void *argument)
{
    (void)argument;

    for (;;) {
        osDelay(100);

        if (!upload_complete) continue;
        upload_complete = false;

        flash_update_state = FLASH_UPDATE_WRITING;

        OLED_Clear();
        OLED_Printf(0, 0, OLED_8X16, "Verifying...");
        OLED_Update();

        const ModelHeader_t *hdr = (const ModelHeader_t *)target_base_addr;
        if (hdr->magic != MODEL_MAGIC || hdr->version != MODEL_VERSION) {
            flash_update_state = FLASH_UPDATE_ERROR;
            ModelLoader_Fallback();
            OLED_Clear();
            OLED_Printf(0, 0, OLED_8X16, "INVALID");
            OLED_Printf(0, 16, OLED_6X8, "Bad header");
            OLED_Update();
            osDelay(2000);
            flash_update_state = FLASH_UPDATE_IDLE;
            continue;
        }

        if (ModelLoader_LoadSlot(flash_update_target_slot)) {
            flash_update_state = FLASH_UPDATE_DONE;
            flash_update_progress = 100;
            OLED_Clear();
            OLED_Printf(0, 0, OLED_8X16, "DONE!");
            OLED_Printf(0, 16, OLED_6X8, "%s %dc",
                        ModelLoader_GetSourceName(),
                        g_active_model.num_classes);
            OLED_Update();
            osDelay(1500);
        } else {
            flash_update_state = FLASH_UPDATE_ERROR;
            ModelLoader_Fallback();
            OLED_Clear();
            OLED_Printf(0, 0, OLED_8X16, "VERIFY FAIL");
            OLED_Printf(0, 16, OLED_6X8, "Fallback to base");
            OLED_Update();
            osDelay(2000);
        }

        flash_update_state = FLASH_UPDATE_IDLE;
    }
}
