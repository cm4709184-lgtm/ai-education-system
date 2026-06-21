/**
 * @file    inference.c
 * @brief   纯C手写DS-CNN推理引擎
 * @author  北京林业大学 理学院 无敌小车车
 * @advisor 张立
 */

#include "inference.h"
#include "model_loader.h"

#if QUANTIZED_INFERENCE
#include "model_weights_q.h"
#else
#include "model_weights.h"
#endif

#include <math.h>
#include <string.h>

#define BN_EPS     1e-5f
#define SOFTMAX_TEMP 1.0f

static float input_t [INF_MEL_FILTERS][INF_FRAME_WINDOW] __attribute__((section(".ccmram")));
static float conv1_out[CONV1_OUT][INF_FRAME_WINDOW] __attribute__((section(".ccmram")));
static float bn1_out  [CONV1_OUT][INF_FRAME_WINDOW] __attribute__((section(".ccmram")));
static float pool1_out[CONV1_OUT][POOL1_LEN] __attribute__((section(".ccmram")));
static float depth1_out[CONV1_OUT][POOL1_LEN] __attribute__((section(".ccmram")));
static float bn_d1_out[CONV1_OUT][POOL1_LEN] __attribute__((section(".ccmram")));
static float point1_out[CONV2_OUT][POOL1_LEN] __attribute__((section(".ccmram")));
static float bn_p1_out[CONV2_OUT][POOL1_LEN] __attribute__((section(".ccmram")));
static float pool2_out[CONV2_OUT][POOL2_LEN] __attribute__((section(".ccmram")));
static float depth2_out[CONV2_OUT][POOL2_LEN] __attribute__((section(".ccmram")));
static float bn_d2_out[CONV2_OUT][POOL2_LEN] __attribute__((section(".ccmram")));
static float point2_out[CONV3_OUT][POOL2_LEN] __attribute__((section(".ccmram")));
static float bn_p2_out[CONV3_OUT][POOL2_LEN] __attribute__((section(".ccmram")));
static float gap_out  [CONV3_OUT] __attribute__((section(".ccmram")));
static float fc_out   [MODEL_MAX_CLASSES] __attribute__((section(".ccmram")));
static float last_probs[MODEL_MAX_CLASSES];
#if HAS_SE
/* SE 隐层缓冲：需容纳 max(SE2_HIDDEN + CONV3_OUT) = 16 + 64 = 80 个 float
 * 实际申请 96 留余量。布局：s[0..hidden-1]=ReLU输出  s[hidden..hidden+channels-1]=sigmoid */
static float se_hidden_buf[96] __attribute__((section(".ccmram")));
#endif

void Inference_Init(void)
{
    memset(input_t,   0, sizeof(input_t));
    memset(conv1_out, 0, sizeof(conv1_out));
    memset(bn1_out,   0, sizeof(bn1_out));
    memset(pool1_out, 0, sizeof(pool1_out));
    memset(depth1_out,0, sizeof(depth1_out));
    memset(bn_d1_out, 0, sizeof(bn_d1_out));
    memset(point1_out,0, sizeof(point1_out));
    memset(bn_p1_out, 0, sizeof(bn_p1_out));
    memset(pool2_out, 0, sizeof(pool2_out));
    memset(depth2_out,0, sizeof(depth2_out));
    memset(bn_d2_out, 0, sizeof(bn_d2_out));
    memset(point2_out,0, sizeof(point2_out));
    memset(bn_p2_out, 0, sizeof(bn_p2_out));
    memset(gap_out,   0, sizeof(gap_out));
    memset(fc_out,    0, sizeof(fc_out));
}

static void conv1d_k5(int out_c, int in_c, int seq_len,
                       const float *weight, const float *bias,
                       const float (*input)[seq_len],
                       float (*output)[seq_len])
{
    for (int c = 0; c < out_c; c++)
    {
        for (int t = 0; t < seq_len; t++)
        {
            float sum = bias[c];
            for (int ic = 0; ic < in_c; ic++)
                for (int k = 0; k < 5; k++)
                {
                    int tt = t + k - 2;
                    float val = (tt >= 0 && tt < seq_len) ? input[ic][tt] : 0.0f;
                    sum += val * weight[c * (in_c * 5) + ic * 5 + k];
                }
            output[c][t] = sum;
        }
    }
}

static void batchnorm_relu(int channels, int seq_len,
                            const float *running_mean, const float *running_var,
                            const float *weight, const float *bias,
                            const float (*input)[seq_len],
                            float (*output)[seq_len])
{
    for (int c = 0; c < channels; c++)
    {
        float inv_std = 1.0f / sqrtf(running_var[c] + BN_EPS);
        for (int t = 0; t < seq_len; t++)
        {
            float n = (input[c][t] - running_mean[c]) * inv_std;
            n = n * weight[c] + bias[c];
            output[c][t] = n > 0.0f ? n : 0.0f;
        }
    }
}

static void maxpool_2(int channels, int in_len, int out_len,
                       const float (*input)[in_len],
                       float (*output)[out_len])
{
    for (int c = 0; c < channels; c++)
        for (int t = 0; t < out_len; t++)
        {
            float a = input[c][t * 2];
            float b = input[c][t * 2 + 1];
            output[c][t] = a > b ? a : b;
        }
}

static void gap1d(int channels, int seq_len,
                   const float (*input)[seq_len],
                   float *output)
{
    for (int c = 0; c < channels; c++)
    {
        float s = 0.0f;
        for (int t = 0; t < seq_len; t++)
            s += input[c][t];
        output[c] = s / (float)seq_len;
    }
}

/* Depthwise Conv1d k=3, groups=channels (float) */
static void depthwise_k3(int channels, int seq_len,
                          const float *weight, const float *bias,
                          const float (*input)[seq_len],
                          float (*output)[seq_len])
{
    for (int c = 0; c < channels; c++)
    {
        for (int t = 0; t < seq_len; t++)
        {
            float sum = bias[c];
            for (int k = 0; k < 3; k++)
            {
                int tt = t + k - 1;
                float val = (tt >= 0 && tt < seq_len) ? input[c][tt] : 0.0f;
                sum += val * weight[c * 3 + k];
            }
            output[c][t] = sum;
        }
    }
}

/* Pointwise Conv1d k=1 (float) */
static void pointwise_k1(int out_c, int in_c, int seq_len,
                          const float *weight, const float *bias,
                          const float (*input)[seq_len],
                          float (*output)[seq_len])
{
    for (int oc = 0; oc < out_c; oc++)
    {
        for (int t = 0; t < seq_len; t++)
        {
            float sum = bias[oc];
            for (int ic = 0; ic < in_c; ic++)
                sum += input[ic][t] * weight[oc * in_c + ic];
            output[oc][t] = sum;
        }
    }
}

static int softmax_with_reject(float *arr, int n, float *confidence)
{
    float max_val = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > max_val) max_val = arr[i];
    float exp_sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        arr[i] = expf((arr[i] - max_val) / SOFTMAX_TEMP);
        exp_sum += arr[i];
    }
    int   best = 0, second = 0;
    float best_prob = 0.0f, second_prob = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float prob = arr[i] / exp_sum;
        if (prob > best_prob)        { second = best; second_prob = best_prob; best = i; best_prob = prob; }
        else if (prob > second_prob) { second = i;    second_prob = prob; }
    }
    /* 拒识：top1 置信度不足或 top1-top2 差太小 */
    if (best_prob < g_active_model.conf_threshold ||
        (best_prob - second_prob) < g_active_model.margin_threshold)
    {
        if (g_active_model.noise_index < n) {
            best = g_active_model.noise_index;
        }
    }
    if (confidence) *confidence = best_prob;
    /* 保存全类 softmax 概率，供跃升检测使用 */
    for (int i = 0; i < n && i < MODEL_MAX_CLASSES; i++)
        last_probs[i] = arr[i] / exp_sum;
    return best;
}

#if HAS_SE
/* Squeeze-and-Excitation 通道注意力（float 权重）
 * 输入  input  : (channels, seq_len)
 * 输出  output : (channels, seq_len) = input * sigmoid(fc2(relu(fc1(GAP(input)))))
 *
 * 缓冲布局（se_hidden_buf[96]）：
 *   [0 .. channels-1]                          = GAP 输出
 *   [channels .. channels+hidden-1]            = fc1 + ReLU 输出
 *   [0 .. channels-1] (覆盖 GAP)                = fc2 + Sigmoid 输出（用于 Scale）
 * 最大占用：max(channels+hidden) = 64+16 = 80 ≤ 96
 */
static void se_block_f(int channels, int hidden, int seq_len,
                        const float *w1, const float *b1,
                        const float *w2, const float *b2,
                        const float (*input)[seq_len],
                        float (*output)[seq_len])
{
    int c, h, t;
    float *gap_buf  = se_hidden_buf;
    float *relu_buf = se_hidden_buf + channels;       /* 不与 gap_buf 重叠 */
    float *sig_buf  = se_hidden_buf;                 /* 复用 gap_buf 区域 */

    /* Squeeze: GAP → gap_buf[0..channels-1] */
    for (c = 0; c < channels; c++)
    {
        float sum = 0.0f;
        for (t = 0; t < seq_len; t++) sum += input[c][t];
        gap_buf[c] = sum / (float)seq_len;
    }
    /* Excitation 1: fc1 + ReLU → relu_buf[0..hidden-1]（独立区，不覆盖 gap） */
    for (h = 0; h < hidden; h++)
    {
        float acc = b1[h];
        for (c = 0; c < channels; c++) acc += gap_buf[c] * w1[h * channels + c];
        relu_buf[h] = acc > 0.0f ? acc : 0.0f;
    }
    /* Excitation 2: fc2 + Sigmoid → sig_buf[0..channels-1]（覆盖 gap_buf） */
    for (c = 0; c < channels; c++)
    {
        float acc = b2[c];
        for (h = 0; h < hidden; h++) acc += relu_buf[h] * w2[c * hidden + h];
        sig_buf[c] = 1.0f / (1.0f + expf(-acc));
    }
    /* Scale: output[c][t] = input[c][t] * sig_buf[c] */
    for (c = 0; c < channels; c++)
    {
        float scale = sig_buf[c];
        for (t = 0; t < seq_len; t++) output[c][t] = input[c][t] * scale;
    }
}
#endif

#if QUANTIZED_INFERENCE

/* Depthwise Conv1d k=3, groups=channels */
static void depthwise_k3_q(int channels, int seq_len,
                            const int8_t *weight_q, float weight_scale, const float *bias,
                            const float (*input)[seq_len],
                            float (*output)[seq_len])
{
    for (int c = 0; c < channels; c++)
    {
        for (int t = 0; t < seq_len; t++)
        {
            float sum = bias[c];
            for (int k = 0; k < 3; k++)
            {
                int tt = t + k - 1;
                float val = (tt >= 0 && tt < seq_len) ? input[c][tt] : 0.0f;
                sum += val * ((float)weight_q[c * 3 + k] * weight_scale);
            }
            output[c][t] = sum;
        }
    }
}

/* Pointwise Conv1d k=1 */
static void pointwise_k1_q(int out_c, int in_c, int seq_len,
                            const int8_t *weight_q, float weight_scale, const float *bias,
                            const float (*input)[seq_len],
                            float (*output)[seq_len])
{
    for (int oc = 0; oc < out_c; oc++)
    {
        for (int t = 0; t < seq_len; t++)
        {
            float sum = bias[oc];
            for (int ic = 0; ic < in_c; ic++)
                sum += input[ic][t] * ((float)weight_q[oc * in_c + ic] * weight_scale);
            output[oc][t] = sum;
        }
    }
}

int Inference_Run(const float *mel_window, float *confidence)
{
    int i, f;
    for (i = 0; i < INF_MEL_FILTERS * INF_FRAME_WINDOW; i++)
        input_t[i % INF_MEL_FILTERS][i / INF_MEL_FILTERS] = mel_window[i];
    for (i = 0; i < INF_FRAME_WINDOW; i++)
    {
        float mean = 0.0f, std = 0.0f;
        for (f = 0; f < INF_MEL_FILTERS; f++)
            mean += input_t[f][i];
        mean /= INF_MEL_FILTERS;
        for (f = 0; f < INF_MEL_FILTERS; f++)
        {
            float d = input_t[f][i] - mean;
            std += d * d;
        }
        std = sqrtf(std / INF_MEL_FILTERS) + 1e-6f;
        for (f = 0; f < INF_MEL_FILTERS; f++)
            input_t[f][i] = (input_t[f][i] - mean) / std;
    }

    conv1d_k5(CONV1_OUT, INF_MEL_FILTERS, INF_FRAME_WINDOW,
               g_active_model.conv1_weight, g_active_model.conv1_bias,
               (const float(*)[INF_FRAME_WINDOW])input_t,
               conv1_out);
    batchnorm_relu(CONV1_OUT, INF_FRAME_WINDOW,
                    g_active_model.bn1_rmean, g_active_model.bn1_rvar, g_active_model.bn1_weight, g_active_model.bn1_bias,
                    (const float(*)[INF_FRAME_WINDOW])conv1_out, bn1_out);
    maxpool_2(CONV1_OUT, INF_FRAME_WINDOW, POOL1_LEN,
               (const float(*)[INF_FRAME_WINDOW])bn1_out, pool1_out);

    depthwise_k3_q(CONV1_OUT, POOL1_LEN,
                    g_active_model.depth1_weight_q, g_active_model.depth1_weight_scale, g_active_model.depth1_bias,
                    (const float(*)[POOL1_LEN])pool1_out,
                    depth1_out);
    batchnorm_relu(CONV1_OUT, POOL1_LEN,
                    g_active_model.bn_d1_rmean, g_active_model.bn_d1_rvar, g_active_model.bn_d1_weight, g_active_model.bn_d1_bias,
                    (const float(*)[POOL1_LEN])depth1_out, bn_d1_out);

    pointwise_k1_q(CONV2_OUT, CONV1_OUT, POOL1_LEN,
                    g_active_model.point1_weight_q, g_active_model.point1_weight_scale, g_active_model.point1_bias,
                    (const float(*)[POOL1_LEN])bn_d1_out,
                    point1_out);
    batchnorm_relu(CONV2_OUT, POOL1_LEN,
                    g_active_model.bn_p1_rmean, g_active_model.bn_p1_rvar, g_active_model.bn_p1_weight, g_active_model.bn_p1_bias,
                    (const float(*)[POOL1_LEN])point1_out, bn_p1_out);
#if HAS_SE
    /* SE1: 在 pool2 之前对 bn_p1_out 做通道注意力（in-place 写回 bn_p1_out） */
    se_block_f(CONV2_OUT, SE1_HIDDEN, POOL1_LEN,
                g_active_model.se1_fc1_weight, g_active_model.se1_fc1_bias,
                g_active_model.se1_fc2_weight, g_active_model.se1_fc2_bias,
                (const float(*)[POOL1_LEN])bn_p1_out,
                bn_p1_out);
#endif
    maxpool_2(CONV2_OUT, POOL1_LEN, POOL2_LEN,
               (const float(*)[POOL1_LEN])bn_p1_out, pool2_out);

    depthwise_k3_q(CONV2_OUT, POOL2_LEN,
                    g_active_model.depth2_weight_q, g_active_model.depth2_weight_scale, g_active_model.depth2_bias,
                    (const float(*)[POOL2_LEN])pool2_out,
                    depth2_out);
    batchnorm_relu(CONV2_OUT, POOL2_LEN,
                    g_active_model.bn_d2_rmean, g_active_model.bn_d2_rvar, g_active_model.bn_d2_weight, g_active_model.bn_d2_bias,
                    (const float(*)[POOL2_LEN])depth2_out, bn_d2_out);

    pointwise_k1_q(CONV3_OUT, CONV2_OUT, POOL2_LEN,
                    g_active_model.point2_weight_q, g_active_model.point2_weight_scale, g_active_model.point2_bias,
                    (const float(*)[POOL2_LEN])bn_d2_out,
                    point2_out);
    batchnorm_relu(CONV3_OUT, POOL2_LEN,
                    g_active_model.bn_p2_rmean, g_active_model.bn_p2_rvar, g_active_model.bn_p2_weight, g_active_model.bn_p2_bias,
                    (const float(*)[POOL2_LEN])point2_out, bn_p2_out);
#if HAS_SE
    /* SE2: 在 GAP 之前对 bn_p2_out 做通道注意力 */
    se_block_f(CONV3_OUT, SE2_HIDDEN, POOL2_LEN,
                g_active_model.se2_fc1_weight, g_active_model.se2_fc1_bias,
                g_active_model.se2_fc2_weight, g_active_model.se2_fc2_bias,
                (const float(*)[POOL2_LEN])bn_p2_out,
                bn_p2_out);
#endif

    gap1d(CONV3_OUT, POOL2_LEN,
           (const float(*)[POOL2_LEN])bn_p2_out, gap_out);

    uint8_t nc = g_active_model.num_classes;

    for (i = 0; i < nc; i++)
    {
        float s = g_active_model.fc_bias[i];
        for (int k = 0; k < CONV3_OUT; k++)
            s += gap_out[k] * ((float)g_active_model.fc_weight_q[i * CONV3_OUT + k] * g_active_model.fc_weight_scale);
        fc_out[i] = s;
    }

    return softmax_with_reject(fc_out, nc, confidence);
}

#else

int Inference_Run(const float *mel_window, float *confidence)
{
    int i, f;
    for (i = 0; i < INF_MEL_FILTERS * INF_FRAME_WINDOW; i++)
        input_t[i % INF_MEL_FILTERS][i / INF_MEL_FILTERS] = mel_window[i];
    for (i = 0; i < INF_FRAME_WINDOW; i++)
    {
        float mean = 0.0f, std = 0.0f;
        for (f = 0; f < INF_MEL_FILTERS; f++)
            mean += input_t[f][i];
        mean /= INF_MEL_FILTERS;
        for (f = 0; f < INF_MEL_FILTERS; f++)
        {
            float d = input_t[f][i] - mean;
            std += d * d;
        }
        std = sqrtf(std / INF_MEL_FILTERS) + 1e-6f;
        for (f = 0; f < INF_MEL_FILTERS; f++)
            input_t[f][i] = (input_t[f][i] - mean) / std;
    }

    conv1d_k5(CONV1_OUT, INF_MEL_FILTERS, INF_FRAME_WINDOW,
               g_active_model.conv1_weight, g_active_model.conv1_bias,
               (const float(*)[INF_FRAME_WINDOW])input_t,
               conv1_out);
    batchnorm_relu(CONV1_OUT, INF_FRAME_WINDOW,
                    g_active_model.bn1_rmean, g_active_model.bn1_rvar, g_active_model.bn1_weight, g_active_model.bn1_bias,
                    (const float(*)[INF_FRAME_WINDOW])conv1_out, bn1_out);
    maxpool_2(CONV1_OUT, INF_FRAME_WINDOW, POOL1_LEN,
               (const float(*)[INF_FRAME_WINDOW])bn1_out, pool1_out);

    depthwise_k3(CONV1_OUT, POOL1_LEN,
                  g_active_model.depth1_weight, g_active_model.depth1_bias,
                  (const float(*)[POOL1_LEN])pool1_out,
                  depth1_out);
    batchnorm_relu(CONV1_OUT, POOL1_LEN,
                    g_active_model.bn_d1_rmean, g_active_model.bn_d1_rvar, g_active_model.bn_d1_weight, g_active_model.bn_d1_bias,
                    (const float(*)[POOL1_LEN])depth1_out, bn_d1_out);

    pointwise_k1(CONV2_OUT, CONV1_OUT, POOL1_LEN,
                  g_active_model.point1_weight, g_active_model.point1_bias,
                  (const float(*)[POOL1_LEN])bn_d1_out,
                  point1_out);
    batchnorm_relu(CONV2_OUT, POOL1_LEN,
                    g_active_model.bn_p1_rmean, g_active_model.bn_p1_rvar, g_active_model.bn_p1_weight, g_active_model.bn_p1_bias,
                    (const float(*)[POOL1_LEN])point1_out, bn_p1_out);
#if HAS_SE
    /* SE1: 通道注意力（in-place 写回 bn_p1_out） */
    se_block_f(CONV2_OUT, SE1_HIDDEN, POOL1_LEN,
                g_active_model.se1_fc1_weight, g_active_model.se1_fc1_bias,
                g_active_model.se1_fc2_weight, g_active_model.se1_fc2_bias,
                (const float(*)[POOL1_LEN])bn_p1_out,
                bn_p1_out);
#endif
    maxpool_2(CONV2_OUT, POOL1_LEN, POOL2_LEN,
               (const float(*)[POOL1_LEN])bn_p1_out, pool2_out);

    depthwise_k3(CONV2_OUT, POOL2_LEN,
                  g_active_model.depth2_weight, g_active_model.depth2_bias,
                  (const float(*)[POOL2_LEN])pool2_out,
                  depth2_out);
    batchnorm_relu(CONV2_OUT, POOL2_LEN,
                    g_active_model.bn_d2_rmean, g_active_model.bn_d2_rvar, g_active_model.bn_d2_weight, g_active_model.bn_d2_bias,
                    (const float(*)[POOL2_LEN])depth2_out, bn_d2_out);

    pointwise_k1(CONV3_OUT, CONV2_OUT, POOL2_LEN,
                  g_active_model.point2_weight, g_active_model.point2_bias,
                  (const float(*)[POOL2_LEN])bn_d2_out,
                  point2_out);
    batchnorm_relu(CONV3_OUT, POOL2_LEN,
                    g_active_model.bn_p2_rmean, g_active_model.bn_p2_rvar, g_active_model.bn_p2_weight, g_active_model.bn_p2_bias,
                    (const float(*)[POOL2_LEN])point2_out, bn_p2_out);
#if HAS_SE
    /* SE2: 通道注意力 */
    se_block_f(CONV3_OUT, SE2_HIDDEN, POOL2_LEN,
                g_active_model.se2_fc1_weight, g_active_model.se2_fc1_bias,
                g_active_model.se2_fc2_weight, g_active_model.se2_fc2_bias,
                (const float(*)[POOL2_LEN])bn_p2_out,
                bn_p2_out);
#endif

    gap1d(CONV3_OUT, POOL2_LEN,
           (const float(*)[POOL2_LEN])bn_p2_out, gap_out);

    uint8_t nc = g_active_model.num_classes;

    for (i = 0; i < nc; i++)
    {
        float s = g_active_model.fc_bias[i];
        for (int k = 0; k < CONV3_OUT; k++)
            s += gap_out[k] * g_active_model.fc_weights[i * CONV3_OUT + k];
        fc_out[i] = s;
    }

    return softmax_with_reject(fc_out, nc, confidence);
}

#endif

uint8_t Inference_GetNumClasses(void)
{
    return g_active_model.num_classes;
}

void Inference_GetProbs(float *buf, int max_classes)
{
    int n = g_active_model.num_classes;
    if (n > max_classes) n = max_classes;
    for (int i = 0; i < n; i++)
        buf[i] = last_probs[i];
}
