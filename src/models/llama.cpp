#include "models.h"
#include <cstring>
#include <cstdlib>
#include <vector>

// ============================================================================
// RoPE distribution custom ops for collecting pre/post-RoPE K values
// ============================================================================
struct rope_dist_params {
    int layer;
    int64_t n_head;
    int64_t head_dim;
};

static int llama_crs_path_mode() {
    const char * env = getenv("LLAMA_CRS_INDEX_PATH");
    return env != nullptr && env[0] != '\0' ? atoi(env) : 2;
}

static void rope_dist_pre_op(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * userdata) {
    (void) dst;
    (void) nth;

    rope_dist_params * params = (rope_dist_params *) userdata;
    if (ith != 0) return;

    const int64_t head_dim = src->ne[0];
    const int64_t n_head = src->ne[1];
    const int64_t n_tokens = src->ne[2];

    if (src->type == GGML_TYPE_F32) {
        rope_dist_update_pre(params->layer, (const float *) src->data, n_head, head_dim, n_tokens);
    } else if (src->type == GGML_TYPE_F16) {
        std::vector<float> f32_data(ggml_nelements(src));
        for (int64_t i = 0; i < ggml_nelements(src); i++) {
            f32_data[i] = ggml_fp16_to_fp32(((const ggml_fp16_t *) src->data)[i]);
        }
        rope_dist_update_pre(params->layer, f32_data.data(), n_head, head_dim, n_tokens);
    }

    if (params->layer == 0) {
        rope_dist_advance_tokens(n_tokens);
    }
}

static void rope_dist_post_op(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * userdata) {
    (void) dst;
    (void) nth;

    rope_dist_params * params = (rope_dist_params *) userdata;
    if (ith != 0) return;

    const int64_t head_dim = src->ne[0];
    const int64_t n_head = src->ne[1];
    const int64_t n_tokens = src->ne[2];

    if (src->type == GGML_TYPE_F32) {
        rope_dist_update_post(params->layer, (const float *) src->data, n_head, head_dim, n_tokens);
    } else if (src->type == GGML_TYPE_F16) {
        std::vector<float> f32_data(ggml_nelements(src));
        for (int64_t i = 0; i < ggml_nelements(src); i++) {
            f32_data[i] = ggml_fp16_to_fp32(((const ggml_fp16_t *) src->data)[i]);
        }
        rope_dist_update_post(params->layer, f32_data.data(), n_head, head_dim, n_tokens);
    }
}

static void qk_dist_q_op(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * userdata) {
    (void) dst;
    (void) nth;

    rope_dist_params * params = (rope_dist_params *) userdata;
    if (ith != 0) return;

    const int64_t head_dim = src->ne[0];
    const int64_t n_head = src->ne[1];
    const int64_t n_tokens = src->ne[2];

    if (src->type == GGML_TYPE_F32) {
        qk_dist_q_update(params->layer, (const float *) src->data, n_head, head_dim, n_tokens);
    } else if (src->type == GGML_TYPE_F16) {
        std::vector<float> f32_data(ggml_nelements(src));
        for (int64_t i = 0; i < ggml_nelements(src); i++) {
            f32_data[i] = ggml_fp16_to_fp32(((const ggml_fp16_t *) src->data)[i]);
        }
        qk_dist_q_update(params->layer, f32_data.data(), n_head, head_dim, n_tokens);
    }
}

static void qk_dist_k_op(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * userdata) {
    (void) dst;
    (void) nth;

    rope_dist_params * params = (rope_dist_params *) userdata;
    if (ith != 0) return;

    const int64_t head_dim = src->ne[0];
    const int64_t n_head = src->ne[1];
    const int64_t n_tokens = src->ne[2];

    if (src->type == GGML_TYPE_F32) {
        qk_dist_k_update(params->layer, (const float *) src->data, n_head, head_dim, n_tokens);
    } else if (src->type == GGML_TYPE_F16) {
        std::vector<float> f32_data(ggml_nelements(src));
        for (int64_t i = 0; i < ggml_nelements(src); i++) {
            f32_data[i] = ggml_fp16_to_fp32(((const ggml_fp16_t *) src->data)[i]);
        }
        qk_dist_k_update(params->layer, f32_data.data(), n_head, head_dim, n_tokens);
    }
}

llm_build_llama::llm_build_llama(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    // Initialize rope distribution tracking if enabled via env var
    rope_dist_init_if_needed(n_layer, n_head_kv, n_embd_head);
    qk_dist_q_init_if_needed(n_layer, n_head,    n_embd_head);
    qk_dist_k_init_if_needed(n_layer, n_head_kv, n_embd_head);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    const int crs_path_mode = llama_crs_path_mode();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // rope freq factors for llama3; may return nullptr for llama2 and other models
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            if (model.layers[il].bq) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);
            }
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);
            if (model.layers[il].bk) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);
            }
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);
            if (model.layers[il].bv) {
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);
            }
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // Pre-RoPE K value collection if enabled
            if (getenv("ROPE_DIST_VALUES_PATH")) {
                rope_dist_params * pre_params = new rope_dist_params{il, n_head_kv, n_embd_head};
                Kcur = ggml_map_custom1_inplace(ctx0, Kcur, rope_dist_pre_op, 1, pre_params);
                ggml_format_name(Kcur, "Kcur_pre_rope_L%d", il);
            }

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            // Pre-RoPE: skip RoPE on K here; it will be applied at attention time
            if (!cparams.pre_rope) {
                Kcur = ggml_rope_ext(
                        ctx0, Kcur, inp_pos, rope_factors,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                        );
            }

            // Post-RoPE K value collection if enabled
            if (getenv("ROPE_DIST_VALUES_PATH") && !cparams.pre_rope) {
                rope_dist_params * post_params = new rope_dist_params{il, n_head_kv, n_embd_head};
                Kcur = ggml_map_custom1_inplace(ctx0, Kcur, rope_dist_post_op, 1, post_params);
                ggml_format_name(Kcur, "Kcur_post_rope_L%d", il);
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            
            // FireQ-style late CRS is defined post-RoPE. When pre_rope mode is
            // enabled, K is rotated later at attention time, so applying the
            // reciprocal Q/K scaling here would be misplaced.
            if (!cparams.pre_rope) {
                auto apply_crs_gather_mul_scatter = [&](ggml_tensor * cur,
                                                        ggml_tensor * indices,
                                                        ggml_tensor * scales,
                                                        const char * layout_name,
                                                        const char * gather_name,
                                                        const char * mul_name,
                                                        const char * scatter_name,
                                                        const char * out_name) -> ggml_tensor * {
                    ggml_tensor * cur4 = ggml_reshape_4d(ctx0, cur, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
                    // Reorder [dim, head, token, 1] -> [token, dim, head, 1] so
                    // ggml_get_rows/ggml_set_rows gather-scatter along the dim axis.
                    ggml_tensor * token_major = ggml_cont(ctx0, ggml_permute(ctx0, cur4, 1, 2, 0, 3));
                    cb(token_major, layout_name, il);
                    ggml_tensor * gathered = ggml_get_rows(ctx0, token_major, indices);
                    cb(gathered, gather_name, il);

                    ggml_tensor * scales4 = ggml_reshape_4d(ctx0, scales, 1, scales->ne[0], scales->ne[1], 1);
                    ggml_tensor * scales_rep = ggml_repeat(ctx0, scales4, gathered);
                    ggml_tensor * scaled = ggml_mul(ctx0, gathered, scales_rep);
                    cb(scaled, mul_name, il);

                    ggml_tensor * scattered = ggml_set_rows(ctx0, token_major, scaled, indices);
                    cb(scattered, scatter_name, il);
                    ggml_tensor * restored = ggml_cont(ctx0, ggml_permute(ctx0, scattered, 2, 0, 1, 3));
                    restored = ggml_reshape_3d(ctx0, restored, cur->ne[0], cur->ne[1], cur->ne[2]);
                    cb(restored, out_name, il);
                    return restored;
                };

                if (il < (int) model.crs_late_q_indices.size() &&
                        model.crs_late_q_indices[il] != nullptr &&
                        model.crs_late_q_scales[il] != nullptr) {
                    if (crs_path_mode == 2) {
                        Qcur = apply_crs_gather_mul_scatter(Qcur,
                                model.crs_late_q_indices[il],
                                model.crs_late_q_scales[il],
                                "Qcur_crs_layout",
                                "Qcur_crs_gather",
                                "Qcur_crs_mul",
                                "Qcur_crs_scatter",
                                "Qcur_scaled");
                    } else {
                        Qcur = ggml_crs_sparse_mul_inplace(ctx0, Qcur,
                                model.crs_late_q_indices[il],
                                model.crs_late_q_scales[il]);
                        cb(Qcur, "Qcur_scaled", il);
                    }
                }

                if (il < (int) model.crs_late_k_indices.size() &&
                        model.crs_late_k_indices[il] != nullptr &&
                        model.crs_late_k_scales[il] != nullptr) {
                    if (crs_path_mode == 2) {
                        Kcur = apply_crs_gather_mul_scatter(Kcur,
                                model.crs_late_k_indices[il],
                                model.crs_late_k_scales[il],
                                "Kcur_crs_layout",
                                "Kcur_crs_gather",
                                "Kcur_crs_mul",
                                "Kcur_crs_scatter",
                                "Kcur_scaled");
                    } else {
                        Kcur = ggml_crs_sparse_mul_inplace(ctx0, Kcur,
                                model.crs_late_k_indices[il],
                                model.crs_late_k_scales[il]);
                        cb(Kcur, "Kcur_scaled", il);
                    }
                }
            }

            cb(Vcur, "Vcur", il);

            if (hparams.use_kq_norm) {
                // Llama4TextL2Norm
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
                cb(Qcur, "Qcur_normed", il);
                cb(Kcur, "Kcur_normed", il);
            }

            // Collect final Q/K tensors exactly as they enter attention.
            if (getenv("QK_DIST_Q_PATH")) {
                rope_dist_params * q_params = new rope_dist_params{il, n_head, n_embd_head};
                Qcur = ggml_map_custom1_inplace(ctx0, Qcur, qk_dist_q_op, 1, q_params);
                ggml_format_name(Qcur, "Qcur_final_qkdist_L%d", il);
            }
            if (getenv("QK_DIST_K_PATH")) {
                rope_dist_params * k_params = new rope_dist_params{il, n_head_kv, n_embd_head};
                Kcur = ggml_map_custom1_inplace(ctx0, Kcur, qk_dist_k_op, 1, k_params);
                ggml_format_name(Kcur, "Kcur_final_qkdist_L%d", il);
            }

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il, rope_factors);
            cb(cur, "attn_out", il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network (non-MoE)
        if (model.layers[il].ffn_gate_inp == nullptr) {

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
            cb(cur, "ffn_moe_out", il);
        }
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
