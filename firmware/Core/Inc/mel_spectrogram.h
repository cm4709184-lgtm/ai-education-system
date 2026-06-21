/**
 * @file    mel_spectrogram.h
 * @brief   Header for Mel spectrogram extraction: hand-written 512-FFT + 20 triangular Mel filter bank + ring buffer
 * @author  北京林业大学 理学院 无敌小车车
 * @advisor 张立
 */

#ifndef MEL_SPECTROGRAM_H
#define MEL_SPECTROGRAM_H

#include <stdint.h>
#include <stdbool.h>

#define MEL_FFT_SIZE            512
#define MEL_HOP_SIZE            256
#define MEL_NUM_FILTERS         20
#define MEL_NUM_BINS            (MEL_FFT_SIZE / 2 + 1)
#define MEL_MAX_FRAMES          32

#define MEL_SAMPLE_RATE         16000
#define MEL_LOW_FREQ            0
#define MEL_HIGH_FREQ           8000

void MelSpectrogram_Init(void);
void MelSpectrogram_ProcessBlock(const int16_t *pcm_block, uint32_t block_size);

extern float mel_frames[MEL_MAX_FRAMES][MEL_NUM_FILTERS];
extern volatile int mel_frame_count;
extern volatile int mel_frame_read_idx;

#endif
