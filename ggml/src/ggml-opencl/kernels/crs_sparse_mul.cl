#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_crs_sparse_mul_f32_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * src2,
        ulong         offset2,
        global char * dst,
        ulong         offsetd,
        int           ne0,
        int           ne1,
        int           ne2,
        int           ne3,
        int           top_k,
        ulong         nb00,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        ulong         nb10,
        ulong         nb11,
        ulong         nb20,
        ulong         nb21,
        ulong         nb0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 += offset0;
    src1 += offset1;
    src2 += offset2;
    dst  += offsetd;

    const int d  = get_global_id(0);
    const int h  = get_global_id(1);
    const int it = get_global_id(2);

    if (d >= ne0 || h >= ne1 || it >= ne2 * ne3) {
        return;
    }

    const int i2 = it % ne2;
    const int i3 = it / ne2;

    float total_scale = 1.0f;
    for (int k = 0; k < top_k; ++k) {
        const int idx = ((global int *) (src1 + k * nb10 + h * nb11))[0];
        if (idx != d) {
            continue;
        }

        const float scale = ((global float *) (src2 + k * nb20 + h * nb21))[0];
        total_scale *= scale;
    }

    global float * src_ptr = (global float *) (src0 + d * nb00 + h * nb01 + i2 * nb02 + i3 * nb03);
    global float * dst_ptr = (global float *) (dst  + d * nb0  + h * nb1  + i2 * nb2  + i3 * nb3);
    dst_ptr[0] = src_ptr[0] * total_scale;
}

kernel void kernel_crs_sparse_mul_index_f32_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * src2,
        ulong         offset2,
        global char * dst,
        ulong         offsetd,
        int           ne0,
        int           ne1,
        int           ne2,
        int           ne3,
        int           top_k,
        ulong         nb00,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        ulong         nb10,
        ulong         nb11,
        ulong         nb20,
        ulong         nb21,
        ulong         nb0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 += offset0;
    src1 += offset1;
    src2 += offset2;
    dst  += offsetd;

    const int k  = get_global_id(0);
    const int h  = get_global_id(1);
    const int it = get_global_id(2);

    if (k >= top_k || h >= ne1 || it >= ne2 * ne3) {
        return;
    }

    const int i2 = it % ne2;
    const int i3 = it / ne2;

    const int idx = ((global int *) (src1 + k * nb10 + h * nb11))[0];
    if (idx < 0 || idx >= ne0) {
        return;
    }

    const float scale = ((global float *) (src2 + k * nb20 + h * nb21))[0];
    global float * src_ptr = (global float *) (src0 + idx * nb00 + h * nb01 + i2 * nb02 + i3 * nb03);
    global float * dst_ptr = (global float *) (dst  + idx * nb0  + h * nb1  + i2 * nb2  + i3 * nb3);
    dst_ptr[0] = src_ptr[0] * scale;
}

kernel void kernel_crs_sparse_mul_f16_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * src2,
        ulong         offset2,
        global char * dst,
        ulong         offsetd,
        int           ne0,
        int           ne1,
        int           ne2,
        int           ne3,
        int           top_k,
        ulong         nb00,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        ulong         nb10,
        ulong         nb11,
        ulong         nb20,
        ulong         nb21,
        ulong         nb0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 += offset0;
    src1 += offset1;
    src2 += offset2;
    dst  += offsetd;

    const int d  = get_global_id(0);
    const int h  = get_global_id(1);
    const int it = get_global_id(2);

    if (d >= ne0 || h >= ne1 || it >= ne2 * ne3) {
        return;
    }

    const int i2 = it % ne2;
    const int i3 = it / ne2;

    float total_scale = 1.0f;
    for (int k = 0; k < top_k; ++k) {
        const int idx = ((global int *) (src1 + k * nb10 + h * nb11))[0];
        if (idx != d) {
            continue;
        }

        const float scale = ((global float *) (src2 + k * nb20 + h * nb21))[0];
        total_scale *= scale;
    }

    global half * src_ptr = (global half *) (src0 + d * nb00 + h * nb01 + i2 * nb02 + i3 * nb03);
    global half * dst_ptr = (global half *) (dst  + d * nb0  + h * nb1  + i2 * nb2  + i3 * nb3);
    dst_ptr[0] = (half) (convert_float(src_ptr[0]) * total_scale);
}

kernel void kernel_crs_sparse_mul_index_f16_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * src2,
        ulong         offset2,
        global char * dst,
        ulong         offsetd,
        int           ne0,
        int           ne1,
        int           ne2,
        int           ne3,
        int           top_k,
        ulong         nb00,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        ulong         nb10,
        ulong         nb11,
        ulong         nb20,
        ulong         nb21,
        ulong         nb0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 += offset0;
    src1 += offset1;
    src2 += offset2;
    dst  += offsetd;

    const int k  = get_global_id(0);
    const int h  = get_global_id(1);
    const int it = get_global_id(2);

    if (k >= top_k || h >= ne1 || it >= ne2 * ne3) {
        return;
    }

    const int i2 = it % ne2;
    const int i3 = it / ne2;

    const int idx = ((global int *) (src1 + k * nb10 + h * nb11))[0];
    if (idx < 0 || idx >= ne0) {
        return;
    }

    const float scale = ((global float *) (src2 + k * nb20 + h * nb21))[0];
    global half * src_ptr = (global half *) (src0 + idx * nb00 + h * nb01 + i2 * nb02 + i3 * nb03);
    global half * dst_ptr = (global half *) (dst  + idx * nb0  + h * nb1  + i2 * nb2  + i3 * nb3);
    dst_ptr[0] = (half) (convert_float(src_ptr[0]) * scale);
}
