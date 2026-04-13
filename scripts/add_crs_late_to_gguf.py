import gguf
import numpy as np
import argparse
import struct
import shutil
import os

def read_scales(filepath):
    with open(filepath, 'rb') as f:
        header_fmt = '<6I'
        header_data = f.read(struct.calcsize(header_fmt))
        if not header_data:
            return None
        magic, version, n_layers, n_heads, n_dims, top_k = struct.unpack(header_fmt, header_data)
        
        all_data = []
        for l in range(n_layers):
            layer_data = []
            for h in range(n_heads):
                idx_data = f.read(top_k * 4)
                indices = struct.unpack(f'<{top_k}i', idx_data)
                
                scale_data = f.read(top_k * 4)
                scales = struct.unpack(f'<{top_k}f', scale_data)
                
                layer_data.append({'indices': indices, 'scales': scales})
            all_data.append(layer_data)
                
        return {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_dims': n_dims,
            'top_k': top_k,
            'data': all_data
        }

def add_crs_tensors(input_gguf, output_gguf, scales_file):
    scales_info = read_scales(scales_file)
    if not scales_info:
        print("Failed to read scales file.")
        return

    n_layers = scales_info['n_layers']
    n_dims = scales_info['n_dims']
    
    print(f"Adding CRS late tensors for {n_layers} layers, dim={n_dims}")
    
    reader = gguf.GGUFReader(input_gguf)
    
    arch = "llama"
    for field in reader.fields.values():
        if field.name == "general.architecture":
            arch = str(field.parts[field.data[-1]][0], 'utf-8')
            break
            
    writer = gguf.GGUFWriter(output_gguf, arch)

    for kv in reader.fields.values():
        if kv.name == "general.architecture":
            continue
            
        if kv.types[0] == gguf.GGUFValueType.ARRAY:
            data = kv.parts[kv.data[-1]]
            if len(data) > 0:
                writer.add_array(kv.name, list(data))
            continue
            
        val = kv.parts[kv.data[-1]][0]
        if kv.types[0] == gguf.GGUFValueType.UINT32:
            writer.add_uint32(kv.name, val)
        elif kv.types[0] == gguf.GGUFValueType.FLOAT32:
            writer.add_float32(kv.name, val)
        elif kv.types[0] == gguf.GGUFValueType.BOOL:
            writer.add_bool(kv.name, val)
        elif kv.types[0] == gguf.GGUFValueType.STRING:
            writer.add_string(kv.name, str(val, 'utf-8') if isinstance(val, bytes) else val)
        elif kv.types[0] == gguf.GGUFValueType.UINT64:
            writer.add_uint64(kv.name, val)
        elif kv.types[0] == gguf.GGUFValueType.INT32:
            writer.add_int32(kv.name, val)
        elif kv.types[0] == gguf.GGUFValueType.FLOAT64:
            writer.add_float64(kv.name, val)
            
    for layer in range(n_layers):
        s_q = np.ones(n_dims, dtype=np.float32)
        s_k_inv = np.ones(n_dims, dtype=np.float32)
        
        is_outlier = np.zeros(n_dims, dtype=bool)
        channel_scales = np.ones(n_dims, dtype=np.float32)
        
        for head in range(scales_info['n_heads']):
            head_data = scales_info['data'][layer][head]
            for i, idx in enumerate(head_data['indices']):
                if 0 <= idx < n_dims:
                    scale = head_data['scales'][i]
                    if not is_outlier[idx] or scale > channel_scales[idx]:
                        channel_scales[idx] = scale
                        is_outlier[idx] = True
                        
        for d in range(n_dims):
            if is_outlier[d] and channel_scales[d] > 0:
                s_q[d] = channel_scales[d]
                # Avoid extreme values or div by zero
                if channel_scales[d] < 1e-6:
                    s_k_inv[d] = 1.0
                else:
                    s_k_inv[d] = 1.0 / channel_scales[d]
                
        writer.add_tensor(f"blk.{layer}.attn_crs_late_s_q", s_q)
        writer.add_tensor(f"blk.{layer}.attn_crs_late_s_k_inv", s_k_inv)
        
    for tensor in reader.tensors:
        # Check if the tensor data is a supported type or memory-mapped view
        # The issue might be memory mapped quantized tensors
        
        # Determine raw type from tensor info
        # We need to correctly pass raw dtype for quantized tensors
        try:
            # We use gguf.GGMLQuantizationType to check if it's quantized
            is_quantized = tensor.tensor_type not in [
                gguf.GGMLQuantizationType.F32, 
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.I8,
                gguf.GGMLQuantizationType.I16,
                gguf.GGMLQuantizationType.I32,
                gguf.GGMLQuantizationType.I64,
                gguf.GGMLQuantizationType.F64
            ]
            
            if is_quantized:
                # For quantized tensors, we need to pass the raw dtype as uint8 byte array
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
            else:
                writer.add_tensor(tensor.name, tensor.data)
        except Exception as e:
            print(f"Error adding tensor {tensor.name}: {e}")
            raise
        
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Successfully wrote {output_gguf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scales", required=True)
    args = parser.parse_args()
    add_crs_tensors(args.input, args.output, args.scales)
