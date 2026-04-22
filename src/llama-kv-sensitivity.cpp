#include "llama-kv-sensitivity.h"

#include "llama-impl.h"

#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

std::mutex g_kv_sensitivity_mutex;

double kv_sensitivity_rel_l2(double diff_sq, double ref_sq) {
    static constexpr double eps = 1e-20;
    return std::sqrt(diff_sq / std::max(ref_sq, eps));
}

void kv_sensitivity_copy_identity(ggml_tensor * dst, const ggml_tensor * src) {
    if (dst->data != src->data) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
    }
}

float kv_sensitivity_read_f32(const ggml_tensor * tensor, int64_t i) {
    switch (tensor->type) {
        case GGML_TYPE_F32:
            return ((const float *) tensor->data)[i];
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(((const ggml_fp16_t *) tensor->data)[i]);
        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(((const ggml_bf16_t *) tensor->data)[i]);
        default:
            throw std::runtime_error(format(
                    "KV sensitivity capture expects F32/F16/BF16 tensors, got %s",
                    ggml_type_name(tensor->type)));
    }
}

void kv_sensitivity_capture_common(
        ggml_tensor * dst,
  const ggml_tensor * ref,
  const ggml_tensor * baseline,
  const ggml_tensor * probe,
                 int   ith,
                 int   nth,
               void * userdata,
                bool   is_output) {
    GGML_UNUSED(nth);

    if (ith != 0) {
        return;
    }

    auto * params = static_cast<llama_kv_sensitivity_capture_params *>(userdata);
    if (params == nullptr || params->state == nullptr) {
        return;
    }

    kv_sensitivity_copy_identity(dst, ref);

    const int64_t n = ggml_nelements(ref);
    double ref_sq = 0.0;
    double diff_sq_baseline = 0.0;
    double diff_sq_probe = 0.0;

    for (int64_t i = 0; i < n; ++i) {
        const double x_ref = kv_sensitivity_read_f32(ref, i);
        const double x_baseline = kv_sensitivity_read_f32(baseline, i);
        const double x_probe = kv_sensitivity_read_f32(probe, i);

        const double d_baseline = x_baseline - x_ref;
        const double d_probe = x_probe - x_ref;

        ref_sq += x_ref * x_ref;
        diff_sq_baseline += d_baseline * d_baseline;
        diff_sq_probe += d_probe * d_probe;
    }

    if (is_output) {
        params->state->add_output_metrics(diff_sq_baseline, diff_sq_probe, ref_sq, n);
    } else {
        params->state->add_score_metrics(diff_sq_baseline, diff_sq_probe, ref_sq, n);
    }
}

} // namespace

llama_kv_sensitivity_state::llama_kv_sensitivity_state(
        int32_t     n_layer_,
        int32_t     layer_,
        ggml_type   baseline_type_,
        ggml_type   probe_type_,
        ggml_type   baseline_k_type_,
        ggml_type   baseline_v_type_,
        ggml_type   probe_k_type_,
        ggml_type   probe_v_type_,
  const std::string & dump_path_) :
    layer(layer_),
    n_layer(n_layer_),
    baseline_type(baseline_type_),
    probe_type(probe_type_),
    baseline_k_type(baseline_k_type_),
    baseline_v_type(baseline_v_type_),
    probe_k_type(probe_k_type_),
    probe_v_type(probe_v_type_),
    dump_path(dump_path_) {
}

llama_kv_sensitivity_state::~llama_kv_sensitivity_state() {
    try {
        dump_json();
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: failed to dump KV sensitivity metrics: %s\n", __func__, e.what());
    }
}

void llama_kv_sensitivity_state::add_score_metrics(
        double diff_sq_baseline,
        double diff_sq_probe,
        double ref_sq,
        int64_t n_elements) {
    std::lock_guard<std::mutex> lock(g_kv_sensitivity_mutex);
    score_diff_sq_baseline += diff_sq_baseline;
    score_diff_sq_probe += diff_sq_probe;
    score_ref_sq += ref_sq;
    score_elements += n_elements;
    score_captures++;
}

void llama_kv_sensitivity_state::add_output_metrics(
        double diff_sq_baseline,
        double diff_sq_probe,
        double ref_sq,
        int64_t n_elements) {
    std::lock_guard<std::mutex> lock(g_kv_sensitivity_mutex);
    output_diff_sq_baseline += diff_sq_baseline;
    output_diff_sq_probe += diff_sq_probe;
    output_ref_sq += ref_sq;
    output_elements += n_elements;
    output_captures++;
}

void llama_kv_sensitivity_state::dump_json() const {
    std::lock_guard<std::mutex> lock(g_kv_sensitivity_mutex);

    if (dumped || dump_path.empty()) {
        return;
    }

    std::filesystem::path out_path(dump_path);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    json out = {
        {"layer", layer},
        {"n_layer", n_layer},
        {"baseline_type", baseline_k_type == baseline_v_type ? ggml_type_name(baseline_type) : "mixed"},
        {"probe_type", probe_k_type == probe_v_type ? ggml_type_name(probe_type) : "mixed"},
        {"baseline_k_type", ggml_type_name(baseline_k_type)},
        {"baseline_v_type", ggml_type_name(baseline_v_type)},
        {"probe_k_type", ggml_type_name(probe_k_type)},
        {"probe_v_type", ggml_type_name(probe_v_type)},
        {"baseline_ea", kv_sensitivity_rel_l2(score_diff_sq_baseline, score_ref_sq)},
        {"probe_ea", kv_sensitivity_rel_l2(score_diff_sq_probe, score_ref_sq)},
        {"delta_ea", kv_sensitivity_rel_l2(score_diff_sq_probe, score_ref_sq) -
                     kv_sensitivity_rel_l2(score_diff_sq_baseline, score_ref_sq)},
        {"baseline_eo", kv_sensitivity_rel_l2(output_diff_sq_baseline, output_ref_sq)},
        {"probe_eo", kv_sensitivity_rel_l2(output_diff_sq_probe, output_ref_sq)},
        {"delta_eo", kv_sensitivity_rel_l2(output_diff_sq_probe, output_ref_sq) -
                     kv_sensitivity_rel_l2(output_diff_sq_baseline, output_ref_sq)},
        {"score_elements", score_elements},
        {"output_elements", output_elements},
        {"score_captures", score_captures},
        {"output_captures", output_captures},
    };

    if (baseline_k_type == probe_k_type) {
        out["k_fixed_type"] = ggml_type_name(baseline_k_type);
    }

    std::ofstream fout(out_path);
    if (!fout) {
        throw std::runtime_error(format("failed to open %s", dump_path.c_str()));
    }

    fout << out.dump(2) << '\n';
    fout.close();

    LLAMA_LOG_INFO("%s: wrote KV sensitivity metrics to %s\n", __func__, dump_path.c_str());
    const_cast<llama_kv_sensitivity_state *>(this)->dumped = true;
}

void llama_kv_sensitivity_capture_score_op(
        ggml_tensor * dst,
  const ggml_tensor * ref,
  const ggml_tensor * baseline,
  const ggml_tensor * probe,
                 int   ith,
                 int   nth,
               void * userdata) {
    kv_sensitivity_capture_common(dst, ref, baseline, probe, ith, nth, userdata, false);
}

void llama_kv_sensitivity_capture_output_op(
        ggml_tensor * dst,
  const ggml_tensor * ref,
  const ggml_tensor * baseline,
  const ggml_tensor * probe,
                 int   ith,
                 int   nth,
               void * userdata) {
    kv_sensitivity_capture_common(dst, ref, baseline, probe, ith, nth, userdata, true);
}
