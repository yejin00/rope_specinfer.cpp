#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <string>

struct llama_kv_sensitivity_state {
    llama_kv_sensitivity_state(
            int32_t     n_layer,
            int32_t     layer,
            ggml_type   baseline_type,
            ggml_type   probe_type,
            ggml_type   baseline_k_type,
            ggml_type   baseline_v_type,
            ggml_type   probe_k_type,
            ggml_type   probe_v_type,
      const std::string & dump_path);

    ~llama_kv_sensitivity_state();

    void add_score_metrics(
            double diff_sq_baseline,
            double diff_sq_probe,
            double ref_sq,
            int64_t n_elements);

    void add_output_metrics(
            double diff_sq_baseline,
            double diff_sq_probe,
            double ref_sq,
            int64_t n_elements);

    void dump_json() const;

    int32_t layer;
    int32_t n_layer;
    ggml_type baseline_type;
    ggml_type probe_type;
    ggml_type baseline_k_type;
    ggml_type baseline_v_type;
    ggml_type probe_k_type;
    ggml_type probe_v_type;
    std::string dump_path;

    double score_diff_sq_baseline = 0.0;
    double score_diff_sq_probe    = 0.0;
    double score_ref_sq           = 0.0;
    double output_diff_sq_baseline = 0.0;
    double output_diff_sq_probe    = 0.0;
    double output_ref_sq           = 0.0;
    int64_t score_elements        = 0;
    int64_t output_elements       = 0;
    int32_t score_captures        = 0;
    int32_t output_captures       = 0;
    bool dumped                   = false;
};

struct llama_kv_sensitivity_capture_params {
    llama_kv_sensitivity_state * state;
};

void llama_kv_sensitivity_capture_score_op(
        ggml_tensor * dst,
  const ggml_tensor * ref,
  const ggml_tensor * baseline,
  const ggml_tensor * probe,
                 int   ith,
                 int   nth,
               void * userdata);

void llama_kv_sensitivity_capture_output_op(
        ggml_tensor * dst,
  const ggml_tensor * ref,
  const ggml_tensor * baseline,
  const ggml_tensor * probe,
                 int   ith,
                 int   nth,
               void * userdata);
