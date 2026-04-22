#include "common.cuh"


static __device__ __forceinline__ void dequantize_q2_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q2_0 * x = (const block_q2_0 *) vx;

    const float d = x[ib].d;

    const int q = x[ib].qs[iqs & 0x7];
    if (iqs < 8) {
        v.x = (q >> 0) & 0x03;
        v.y = (q >> 2) & 0x03;
    } else {
        v.x = (q >> 4) & 0x03;
        v.y = (q >> 6) & 0x03;
    }

    v.x = (v.x - 2.0f) * d;
    v.y = (v.y - 2.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_0_head(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0_head * x = (const block_q4_0_head *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q2_0_head(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q2_0_head * x = (const block_q2_0_head *) vx;

    const float d = x[ib].d;

    const int q = x[ib].qs[iqs & 0x1f];
    if (iqs < 32) {
        v.x = (q >> 0) & 0x03;
        v.y = (q >> 2) & 0x03;
    } else {
        v.x = (q >> 4) & 0x03;
        v.y = (q >> 6) & 0x03;
    }

    v.x = (v.x - 2.0f) * d;
    v.y = (v.y - 2.0f) * d;
}

static __device__ __forceinline__ int dequantize_q3_0_head_get_code(const block_q3_0_head * x, const int ir) {
    const uint8_t q = x->qs[ir & 0x1f];
    const uint8_t low = (ir < 32 ? (q >> 0) : (ir < 64 ? (q >> 4) : (ir < 96 ? (q >> 2) : (q >> 6)))) & 0x03;
    const uint8_t high = (x->qh[ir >> 3] >> (ir & 0x07)) & 0x01;
    return low | (high << 2);
}

static __device__ __forceinline__ void dequantize_q3_0_head(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q3_0_head * x = (const block_q3_0_head *) vx;

    const float d = x[ib].d;

    v.x = dequantize_q3_0_head_get_code(&x[ib], iqs + 0);
    v.y = dequantize_q3_0_head_get_code(&x[ib], iqs + QK3_0_HEAD/2);

    v.x = (v.x - 4.0f) * d;
    v.y = (v.y - 4.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_0_q2_0_head(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0_q2_0_head * x = (const block_q4_0_q2_0_head *) vx;

    const int iq4_block = iqs / QK4_0;
    const int ir4 = iqs % QK4_0;
    const uint8_t q4 = x[ib].q4[iq4_block].qs[ir4 & 0x0F];
    const float d4 = x[ib].q4[iq4_block].d;
    v.x = ((ir4 < QK4_0/2 ? (q4 & 0x0F) : (q4 >> 4)) - 8.0f) * d4;

    const int iq2_block = iqs / QK2_0;
    const int ir2 = iqs % QK2_0;
    const uint8_t q2 = x[ib].q2[iq2_block].qs[ir2 & 0x7];
    const float d2 = x[ib].q2[iq2_block].d;
    switch (ir2 >> 3) {
        case 0: v.y = (float)((q2 >> 0) & 0x03); break;
        case 1: v.y = (float)((q2 >> 4) & 0x03); break;
        case 2: v.y = (float)((q2 >> 2) & 0x03); break;
        default: v.y = (float)((q2 >> 6) & 0x03); break;
    }
    v.y = (v.y - 2.0f) * d2;
}

static __device__ __forceinline__ void dequantize_q2_0_q4_0_head(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q2_0_q4_0_head * x = (const block_q2_0_q4_0_head *) vx;

    const int iq2_block = iqs / QK2_0;
    const int ir2 = iqs % QK2_0;
    const uint8_t q2 = x[ib].q2[iq2_block].qs[ir2 & 0x7];
    const float d2 = x[ib].q2[iq2_block].d;
    switch (ir2 >> 3) {
        case 0: v.x = (float)((q2 >> 0) & 0x03); break;
        case 1: v.x = (float)((q2 >> 4) & 0x03); break;
        case 2: v.x = (float)((q2 >> 2) & 0x03); break;
        default: v.x = (float)((q2 >> 6) & 0x03); break;
    }
    v.x = (v.x - 2.0f) * d2;

    const int iq4_block = iqs / QK4_0;
    const int ir4 = iqs % QK4_0;
    const uint8_t q4 = x[ib].q4[iq4_block].qs[ir4 & 0x0F];
    const float d4 = x[ib].q4[iq4_block].d;
    v.y = ((ir4 < QK4_0/2 ? (q4 & 0x0F) : (q4 >> 4)) - 8.0f) * d4;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

static __device__ __forceinline__ void dequantize_q8_0_head(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0_head * x = (const block_q8_0_head *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}
