#include "arg.h"
#include "common.h"

#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

int main(void) {
    common_params params;

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.args) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg);
                        exit(1);
                    }
                }
                // check for env var duplications
                if (opt.env) {
                    if (seen_env_vars.find(opt.env) == seen_env_vars.end()) {
                        seen_env_vars.insert(opt.env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", opt.env);
                        exit(1);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));

    // empty layer-wise KV cache spec
    argv = {"binary_name", "--kv-layer-k-types", ""};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    argv = {"binary_name", "--hadamard-granularity", "bad"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--draft", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.n_max == 123);

    params = common_params();
    argv = {"binary_name", "--kv-layer-types", "0-3:q4_0_head,4-7:q2_0_q4_0_head"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kv_layer_k_types == "0-3:q4_0_head,4-7:q2_0_q4_0_head");
    assert(params.kv_layer_v_types == "0-3:q4_0_head,4-7:q2_0_q4_0_head");

    params = common_params();
    argv = {"binary_name", "-ctv", "q2_0_head"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cache_type_v == GGML_TYPE_Q2_0_HEAD);

    params = common_params();
    argv = {"binary_name", "-ctk", "q3_0_head"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cache_type_k == GGML_TYPE_Q3_0_HEAD);

    params = common_params();
    argv = {"binary_name", "-ctv", "q8_0_head"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cache_type_v == GGML_TYPE_Q8_0_HEAD);

    params = common_params();
    argv = {"binary_name", "--hadamard", "--hadamard-seed", "42", "--hadamard-granularity", "layer"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.hadamard == true);
    assert(params.hadamard_seed == 42);
    assert(params.hadamard_granularity == LLAMA_HADAMARD_GRANULARITY_LAYER);

    params = common_params();
    argv = {
        "binary_name",
        "--measure-kv-sensitivity",
        "--sensitivity-layer", "13",
        "--sensitivity-baseline-type", "q4_0_head",
        "--sensitivity-probe-type", "q2_0_head",
        "--dump-attn-error", "attn_error.json",
    };
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.measure_kv_sensitivity == true);
    assert(params.sensitivity_layer == 13);
    assert(params.sensitivity_baseline_type == GGML_TYPE_Q4_0_HEAD);
    assert(params.sensitivity_probe_type == GGML_TYPE_Q2_0_HEAD);
    assert(params.sensitivity_baseline_k_type == GGML_TYPE_Q4_0_HEAD);
    assert(params.sensitivity_baseline_v_type == GGML_TYPE_Q4_0_HEAD);
    assert(params.sensitivity_probe_k_type == GGML_TYPE_Q2_0_HEAD);
    assert(params.sensitivity_probe_v_type == GGML_TYPE_Q2_0_HEAD);
    assert(params.dump_attn_error == "attn_error.json");

    params = common_params();
    argv = {
        "binary_name",
        "--measure-kv-sensitivity",
        "--sensitivity-baseline-k-type", "q4_0_head",
        "--sensitivity-baseline-v-type", "q4_0_head",
        "--sensitivity-probe-k-type", "q4_0_head",
        "--sensitivity-probe-v-type", "q2_0_head",
    };
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.sensitivity_baseline_k_type == GGML_TYPE_Q4_0_HEAD);
    assert(params.sensitivity_baseline_v_type == GGML_TYPE_Q4_0_HEAD);
    assert(params.sensitivity_probe_k_type == GGML_TYPE_Q4_0_HEAD);
    assert(params.sensitivity_probe_v_type == GGML_TYPE_Q2_0_HEAD);

    params = common_params();
    argv = {"binary_name", "--kv-layer-types", "0-3:q4_0_head", "--kv-layer-v-types", "0,2,5:q4_0"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.kv_layer_k_types == "0-3:q4_0_head");
    assert(params.kv_layer_v_types == "0,2,5:q4_0");

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);


    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    printf("test-arg-parser: test curl-related functions\n\n");
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    {
        printf("test-arg-parser: test good URL\n\n");
        auto res = common_remote_get_content(GOOD_URL, {});
        assert(res.first == 200);
        assert(res.second.size() > 0);
        std::string str(res.second.data(), res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }

    printf("test-arg-parser: all tests OK\n\n");
}
