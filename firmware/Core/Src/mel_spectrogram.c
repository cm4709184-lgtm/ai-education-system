/**
 * @file    mel_spectrogram.c
 * @brief   Mel频谱提取: 手写512-FFT + 20三角Mel滤波器 + 环形缓冲区
 * @author  北京林业大学 理学院 无敌小车车
 * @advisor 张立
 */

#include "mel_spectrogram.h"
#include "main.h"
#include <math.h>
#include <string.h>

#define PI                  3.141592653589793f
#define LOG_EPSILON         1e-12f
#define TWO_OVER_FFT_SIZE   (2.0f / MEL_FFT_SIZE)

float mel_frames[MEL_MAX_FRAMES][MEL_NUM_FILTERS];
volatile int mel_frame_count = 0;
volatile int mel_frame_read_idx = 0;

static float hann_window[MEL_FFT_SIZE];
static uint16_t bit_rev[MEL_FFT_SIZE];
static float filter_bank[MEL_NUM_FILTERS][MEL_NUM_BINS];
static float overlap_tail[MEL_HOP_SIZE];
static volatile int mel_write_idx = 0;

static float hz_to_mel(float hz)
{
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel)
{
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

static void init_hann_window(void)
{
    for (int i = 0; i < MEL_FFT_SIZE; i++)
        hann_window[i] = 0.5f * (1.0f - cosf(2.0f * PI * i / (MEL_FFT_SIZE - 1)));
}

static void init_bit_reversal(void)
{
    int bits = 0;
    int n = MEL_FFT_SIZE;
    while (n >>= 1) bits++;

    for (int i = 0; i < MEL_FFT_SIZE; i++)
    {
        int rev = 0;
        for (int j = 0; j < bits; j++)
            rev = (rev << 1) | ((i >> j) & 1);
        bit_rev[i] = rev;
    }
}

static void init_filter_bank(void)
{
    float mel_low = hz_to_mel(MEL_LOW_FREQ);
    float mel_high = hz_to_mel(MEL_HIGH_FREQ);
    float mel_step = (mel_high - mel_low) / (MEL_NUM_FILTERS + 1);

    memset(filter_bank, 0, sizeof(filter_bank));

    for (int m = 0; m < MEL_NUM_FILTERS; m++)
    {
        float mel_center = mel_low + (m + 1) * mel_step;

        float hz_center = mel_to_hz(mel_center);
        float hz_left = mel_to_hz(mel_center - mel_step);
        float hz_right = mel_to_hz(mel_center + mel_step);

        int bin_left = (int)(hz_left * MEL_FFT_SIZE / MEL_SAMPLE_RATE);
        int bin_center = (int)(hz_center * MEL_FFT_SIZE / MEL_SAMPLE_RATE);
        int bin_right = (int)(hz_right * MEL_FFT_SIZE / MEL_SAMPLE_RATE);

        if (bin_left < 0) bin_left = 0;
        if (bin_right > MEL_NUM_BINS - 1) bin_right = MEL_NUM_BINS - 1;

        float denom_left = (bin_center > bin_left) ? (float)(bin_center - bin_left) : 1.0f;
        float denom_right = (bin_right > bin_center) ? (float)(bin_right - bin_center) : 1.0f;

        for (int k = bin_left; k <= bin_center; k++)
            filter_bank[m][k] = (float)(k - bin_left) / denom_left;

        for (int k = bin_center + 1; k <= bin_right; k++)
            filter_bank[m][k] = (float)(bin_right - k) / denom_right;
    }
}

static void apply_fft(float* real, float* imag)
{
    for (int i = 0; i < MEL_FFT_SIZE; i++)
    {
        int j = bit_rev[i];
        if (i < j)
        {
            float t = real[i]; real[i] = real[j]; real[j] = t;
            t = imag[i]; imag[i] = imag[j]; imag[j] = t;
        }
    }

    for (int len = 2; len <= MEL_FFT_SIZE; len <<= 1)
    {
        float wlen_real = cosf(2.0f * PI / len);
        float wlen_imag = -sinf(2.0f * PI / len);

        for (int i = 0; i < MEL_FFT_SIZE; i += len)
        {
            float w_real = 1.0f;
            float w_imag = 0.0f;
            int half = len >> 1;

            for (int j = 0; j < half; j++)
            {
                float u_real = real[i + j];
                float u_imag = imag[i + j];
                float v_real = real[i + j + half] * w_real - imag[i + j + half] * w_imag;
                float v_imag = real[i + j + half] * w_imag + imag[i + j + half] * w_real;

                real[i + j] = u_real + v_real;
                imag[i + j] = u_imag + v_imag;
                real[i + j + half] = u_real - v_real;
                imag[i + j + half] = u_imag - v_imag;

                float t_real = w_real * wlen_real - w_imag * wlen_imag;
                w_imag = w_real * wlen_imag + w_imag * wlen_real;
                w_real = t_real;
            }
        }
    }
}

static void process_window(const float* windowed_input, float* out_energies)
{
    static float real[MEL_FFT_SIZE];
    static float imag[MEL_FFT_SIZE];
    static float power[MEL_NUM_BINS];

    for (int i = 0; i < MEL_FFT_SIZE; i++)
    {
        real[i] = windowed_input[i];
        imag[i] = 0.0f;
    }

    apply_fft(real, imag);

    for (int i = 0; i < MEL_NUM_BINS; i++)
        power[i] = (real[i] * real[i] + imag[i] * imag[i]) * TWO_OVER_FFT_SIZE;
    power[0] = 0.0f;  /* 排除 DC 分量 */
    power[1] = 0.0f;  /* 测试：确认代码生效 */

    for (int m = 0; m < MEL_NUM_FILTERS; m++)
    {
        float sum = 0.0f;
        for (int k = 0; k < MEL_NUM_BINS; k++)
            sum += power[k] * filter_bank[m][k];
        out_energies[m] = logf(sum + LOG_EPSILON);
    }
}

void MelSpectrogram_Init(void)
{
    init_hann_window();
    init_bit_reversal();
    init_filter_bank();
    memset(overlap_tail, 0, sizeof(overlap_tail));
    mel_write_idx = 0;
    mel_frame_count = 0;
    mel_frame_read_idx = 0;
}

void MelSpectrogram_ProcessBlock(const int16_t *pcm_block, uint32_t block_size)
{
    if (block_size < MEL_HOP_SIZE * 2)
        return;

    int base = mel_write_idx;
    float fft_buffer[MEL_FFT_SIZE];

    for (int i = 0; i < MEL_HOP_SIZE; i++)
        fft_buffer[i] = overlap_tail[i];
    for (int i = 0; i < MEL_HOP_SIZE; i++)
        fft_buffer[i + MEL_HOP_SIZE] = (float)pcm_block[i];
    for (int i = 0; i < MEL_FFT_SIZE; i++)
        fft_buffer[i] *= hann_window[i];
    process_window(fft_buffer, mel_frames[base]);
    base = (base + 1) & (MEL_MAX_FRAMES - 1);

    for (int i = 0; i < MEL_FFT_SIZE; i++)
        fft_buffer[i] = (float)pcm_block[i] * hann_window[i];
    process_window(fft_buffer, mel_frames[base]);
    base = (base + 1) & (MEL_MAX_FRAMES - 1);

    for (int i = 0; i < MEL_HOP_SIZE; i++)
        overlap_tail[i] = (float)pcm_block[i + MEL_HOP_SIZE];

    mel_write_idx = base;
    __disable_irq();
    mel_frame_count += 2;
    __enable_irq();
}
