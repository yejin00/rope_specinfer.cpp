#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef uchar uint8_t;

#define QK4_0 32

//------------------------------------------------------------------------------
// block_q4_0
//------------------------------------------------------------------------------
struct block_q4_0
{
    half d;
    uint8_t qs[QK4_0 / 2];
};

// v = { mp, L, d }
inline uint fastdiv(uint n, uint4 v) {
    uint msbs;
    msbs = mul_hi(n, v.s0);
    return (msbs + n) >> v.s1;
}
inline uint fastmod(uint n, uint4 v) {
    uint q = fastdiv(n, v);
    return n - q * v.s2;
}

//------------------------------------------------------------------------------
// quantize_q4_0 - Optimized Q4_0 quantization (CUDA-inspired)
//------------------------------------------------------------------------------
// Key optimizations:
// 1. Reduced memory writes by packing in register
// 2. Better loop unrolling
//------------------------------------------------------------------------------
void quantize_q4_0(global const float * src, global struct block_q4_0 * dst) {
    float amax = 0.0f;
    float max = 0.0f;

    // Find max value
    #pragma unroll 8
    for (int j = 0; j < QK4_0; j++) {
        const float v = src[j];
        const float abs_v = fabs(v);
        if (amax < abs_v) {
            amax = abs_v;
            max = v;
        }
    }

    const float d = max / -8.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    dst->d = (half)d;

    // Quantize and pack
    #pragma unroll 8
    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = src[j] * id;
        const float x1 = src[j + QK4_0/2] * id;

        const uint8_t xi0 = min((uint8_t)15, (uint8_t)(x0 + 8.5f));
        const uint8_t xi1 = min((uint8_t)15, (uint8_t)(x1 + 8.5f));

        dst->qs[j] = xi0 | (xi1 << 4);
    }
}

kernel void kernel_set_rows_f32_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global float * dst_row = (global float *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = (float)src_row[ind];
    }
}

kernel void kernel_set_rows_f16_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global half  * dst_row = (global half  *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = src_row[ind];
    }
}

kernel void kernel_set_rows_f32_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global float * dst_row = (global float *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = (float)src_row[ind];
    }
}

kernel void kernel_set_rows_f16_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global half  * dst_row = (global half  *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = src_row[ind];
    }
}

//------------------------------------------------------------------------------
// block_q4_0_head (block size 128, per-head KV cache quantization)
//------------------------------------------------------------------------------
#define QK4_0_HEAD 128

struct block_q4_0_head
{
    half d;
    uint8_t qs[QK4_0_HEAD / 2];
};

void quantize_q4_0_head(global const float * src, global struct block_q4_0_head * dst) {
    float amax = 0.0f;
    float max = 0.0f;

    // Find max value
    for (int j = 0; j < QK4_0_HEAD; j++) {
        const float v = src[j];
        const float abs_v = fabs(v);
        if (amax < abs_v) {
            amax = abs_v;
            max = v;
        }
    }

    const float d = max / -8.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    dst->d = (half)d;

    // Quantize and pack
    for (int j = 0; j < QK4_0_HEAD/2; ++j) {
        const float x0 = src[j] * id;
        const float x1 = src[j + QK4_0_HEAD/2] * id;

        const uint8_t xi0 = min((uint8_t)15, (uint8_t)(x0 + 8.5f));
        const uint8_t xi1 = min((uint8_t)15, (uint8_t)(x1 + 8.5f));

        dst->qs[j] = xi0 | (xi1 << 4);
    }
}

kernel void kernel_set_rows_q4_0_head_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global struct block_q4_0_head * dst_row = (global struct block_q4_0_head *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float                  * src_row = (global float                  *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    // Each work item processes one Q4_0_HEAD block (128 floats)
    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        quantize_q4_0_head(src_row + ind*QK4_0_HEAD, dst_row + ind);
    }
}

kernel void kernel_set_rows_q4_0_head_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global struct block_q4_0_head * dst_row = (global struct block_q4_0_head *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float                  * src_row = (global float                  *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    // Each work item processes one Q4_0_HEAD block (128 floats)
    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        quantize_q4_0_head(src_row + ind*QK4_0_HEAD, dst_row + ind);
    }
}

kernel void kernel_set_rows_q4_0_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global struct block_q4_0 * dst_row = (global struct block_q4_0 *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float             * src_row = (global float             *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    // Each work item processes one Q4_0 block (32 floats)
    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        quantize_q4_0(src_row + ind*QK4_0, dst_row + ind);
    }
}

kernel void kernel_set_rows_q4_0_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global struct block_q4_0 * dst_row = (global struct block_q4_0 *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float             * src_row = (global float             *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    // Each work item processes one Q4_0 block (32 floats)
    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        quantize_q4_0(src_row + ind*QK4_0, dst_row + ind);
    }
}