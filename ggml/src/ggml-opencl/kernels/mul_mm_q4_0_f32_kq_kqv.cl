#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define REQD_SUBGROUP_SIZE __attribute__((intel_reqd_sub_group_size(16)))
#define GGML_CL_SG_SIZE 16
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE __attribute__((qcom_reqd_sub_group_size("half")))
#define GGML_CL_SG_SIZE 64
#else
#define REQD_SUBGROUP_SIZE
#define GGML_CL_SG_SIZE 32
#endif

#define QK4_0 32
#define QK4_0_BYTES (QK4_0 / 2)
#define QK4_0_SIZE_BYTES (sizeof(half) + QK4_0_BYTES)
#define N_DST 4

inline float block_q4_0_dot_y_half(
        global const uchar * block_ptr,
        float sumy,
        float16 yl,
        int il) {
    const float d = vload_half(0, (global const half *) block_ptr);
    global const ushort * qs = (global const ushort *) (block_ptr + sizeof(half) + il/2);
    float acc = 0.0f;

    acc += yl.s0 * (qs[0] & 0x000F);
    acc += yl.s1 * (qs[0] & 0x0F00);
    acc += yl.s8 * (qs[0] & 0x00F0);
    acc += yl.s9 * (qs[0] & 0xF000);

    acc += yl.s2 * (qs[1] & 0x000F);
    acc += yl.s3 * (qs[1] & 0x0F00);
    acc += yl.sa * (qs[1] & 0x00F0);
    acc += yl.sb * (qs[1] & 0xF000);

    acc += yl.s4 * (qs[2] & 0x000F);
    acc += yl.s5 * (qs[2] & 0x0F00);
    acc += yl.sc * (qs[2] & 0x00F0);
    acc += yl.sd * (qs[2] & 0xF000);

    acc += yl.s6 * (qs[3] & 0x000F);
    acc += yl.s7 * (qs[3] & 0x0F00);
    acc += yl.se * (qs[3] & 0x00F0);
    acc += yl.sf * (qs[3] & 0xF000);

    return d * (sumy * -8.0f + acc);
}

inline float q4_0_scalar(
        global const uchar * block_ptr,
        int elem_idx) {
    const float d = vload_half(0, (global const half *) block_ptr);
    global const uchar * qs = block_ptr + sizeof(half);
    const uchar packed = qs[elem_idx & 0xF];
    const int q = elem_idx < 16 ? (packed & 0x0F) : (packed >> 4);
    return d * (float) (q - 8);
}

REQD_SUBGROUP_SIZE
kernel void mul_mm_q4_0_f32_kq(
        global const uchar * src0,
        ulong offset0,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        global const float * src1,
        ulong offset1,
        ulong s11,
        ulong s12,
        ulong s13,
        global float * dst,
        ulong offsetd,
        ulong sd1,
        ulong sd2,
        ulong sd3,
        int ne00,
        int ne01,
        int ne12,
        int r2,
        int r3) {
    (void) nb00;

    const int lid = get_sub_group_local_id();
    const int row = get_group_id(0) * N_DST;
    const int col = get_group_id(1);
    const int im  = get_group_id(2);

    const int i12 = im % ne12;
    const int i13 = im / ne12;
    const int i02 = i12 / r2;
    const int i03 = i13 / r3;

    const int nb = ne00 / QK4_0;
    const int ib0 = lid >> 1;
    const int il  = (lid & 1) * 8;

    global const float * y = src1 + offset1 + (ulong) col * s11 + (ulong) i12 * s12 + (ulong) i13 * s13;
    const ulong row_off0 = offset0 + (ulong) (row + 0) * nb01 + (ulong) i02 * nb02 + (ulong) i03 * nb03;
    const ulong row_off1 = offset0 + (ulong) (row + 1) * nb01 + (ulong) i02 * nb02 + (ulong) i03 * nb03;
    const ulong row_off2 = offset0 + (ulong) (row + 2) * nb01 + (ulong) i02 * nb02 + (ulong) i03 * nb03;
    const ulong row_off3 = offset0 + (ulong) (row + 3) * nb01 + (ulong) i02 * nb02 + (ulong) i03 * nb03;

    float4 sumf = 0.0f;

    for (int ib = ib0; ib < nb; ib += GGML_CL_SG_SIZE/2) {
        float16 yl;
        float sumy = 0.0f;
        global const float * yb = y + (ulong) ib * QK4_0 + il;

        sumy += yb[0];
        sumy += yb[1];
        sumy += yb[2];
        sumy += yb[3];
        sumy += yb[4];
        sumy += yb[5];
        sumy += yb[6];
        sumy += yb[7];

        sumy += yb[16];
        sumy += yb[17];
        sumy += yb[18];
        sumy += yb[19];
        sumy += yb[20];
        sumy += yb[21];
        sumy += yb[22];
        sumy += yb[23];

        yl.s0 = yb[0];
        yl.s1 = yb[1]/256.0f;

        yl.s2 = yb[2];
        yl.s3 = yb[3]/256.0f;

        yl.s4 = yb[4];
        yl.s5 = yb[5]/256.0f;

        yl.s6 = yb[6];
        yl.s7 = yb[7]/256.0f;

        yl.s8 = yb[16]/16.0f;
        yl.s9 = yb[17]/4096.0f;

        yl.sa = yb[18]/16.0f;
        yl.sb = yb[19]/4096.0f;

        yl.sc = yb[20]/16.0f;
        yl.sd = yb[21]/4096.0f;

        yl.se = yb[22]/16.0f;
        yl.sf = yb[23]/4096.0f;

        if (row + 0 < ne01) {
            sumf.s0 += block_q4_0_dot_y_half(src0 + row_off0 + (ulong) ib * QK4_0_SIZE_BYTES, sumy, yl, il);
        }
        if (row + 1 < ne01) {
            sumf.s1 += block_q4_0_dot_y_half(src0 + row_off1 + (ulong) ib * QK4_0_SIZE_BYTES, sumy, yl, il);
        }
        if (row + 2 < ne01) {
            sumf.s2 += block_q4_0_dot_y_half(src0 + row_off2 + (ulong) ib * QK4_0_SIZE_BYTES, sumy, yl, il);
        }
        if (row + 3 < ne01) {
            sumf.s3 += block_q4_0_dot_y_half(src0 + row_off3 + (ulong) ib * QK4_0_SIZE_BYTES, sumy, yl, il);
        }
    }

    const float4 total = (float4) (
            sub_group_reduce_add(sumf.s0),
            sub_group_reduce_add(sumf.s1),
            sub_group_reduce_add(sumf.s2),
            sub_group_reduce_add(sumf.s3));

    if (lid == 0) {
        global float * out = dst + offsetd + (ulong) col * sd1 + (ulong) i12 * sd2 + (ulong) i13 * sd3;

        if (row + 0 < ne01) {
            out[row + 0] = total.s0;
        }
        if (row + 1 < ne01) {
            out[row + 1] = total.s1;
        }
        if (row + 2 < ne01) {
            out[row + 2] = total.s2;
        }
        if (row + 3 < ne01) {
            out[row + 3] = total.s3;
        }
    }
}

REQD_SUBGROUP_SIZE
kernel void mul_mm_q4_0_f32_kqv(
        global const uchar * src0,
        ulong offset0,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        global const float * src1,
        ulong offset1,
        ulong s11,
        ulong s12,
        ulong s13,
        global float * dst,
        ulong offsetd,
        ulong sd1,
        ulong sd2,
        ulong sd3,
        int ne00,
        int ne01,
        int ne12,
        int r2,
        int r3) {
    (void) nb01;

    const int lid = get_sub_group_local_id();
    const int row = get_group_id(0) * N_DST;
    const int col = get_group_id(1);
    const int im  = get_group_id(2);

    const int i12 = im % ne12;
    const int i13 = im / ne12;
    const int i02 = i12 / r2;
    const int i03 = i13 / r3;

    const int row_block0 = (row + 0) / QK4_0;
    const int row_block1 = (row + 1) / QK4_0;
    const int row_block2 = (row + 2) / QK4_0;
    const int row_block3 = (row + 3) / QK4_0;

    const int row_elem0 = (row + 0) % QK4_0;
    const int row_elem1 = (row + 1) % QK4_0;
    const int row_elem2 = (row + 2) % QK4_0;
    const int row_elem3 = (row + 3) % QK4_0;

    global const float * y = src1 + offset1 + (ulong) col * s11 + (ulong) i12 * s12 + (ulong) i13 * s13;

    float4 sumf = 0.0f;

    for (int k = lid; k < ne00; k += GGML_CL_SG_SIZE) {
        const float yk = y[k];
        const ulong base_off = offset0 + (ulong) k * nb00 + (ulong) i02 * nb02 + (ulong) i03 * nb03;

        if (row + 0 < ne01) {
            sumf.s0 += q4_0_scalar(src0 + base_off + (ulong) row_block0 * QK4_0_SIZE_BYTES, row_elem0) * yk;
        }
        if (row + 1 < ne01) {
            sumf.s1 += q4_0_scalar(src0 + base_off + (ulong) row_block1 * QK4_0_SIZE_BYTES, row_elem1) * yk;
        }
        if (row + 2 < ne01) {
            sumf.s2 += q4_0_scalar(src0 + base_off + (ulong) row_block2 * QK4_0_SIZE_BYTES, row_elem2) * yk;
        }
        if (row + 3 < ne01) {
            sumf.s3 += q4_0_scalar(src0 + base_off + (ulong) row_block3 * QK4_0_SIZE_BYTES, row_elem3) * yk;
        }
    }

    const float4 total = (float4) (
            sub_group_reduce_add(sumf.s0),
            sub_group_reduce_add(sumf.s1),
            sub_group_reduce_add(sumf.s2),
            sub_group_reduce_add(sumf.s3));

    if (lid == 0) {
        global float * out = dst + offsetd + (ulong) col * sd1 + (ulong) i12 * sd2 + (ulong) i13 * sd3;

        if (row + 0 < ne01) {
            out[row + 0] = total.s0;
        }
        if (row + 1 < ne01) {
            out[row + 1] = total.s1;
        }
        if (row + 2 < ne01) {
            out[row + 2] = total.s2;
        }
        if (row + 3 < ne01) {
            out[row + 3] = total.s3;
        }
    }
}
