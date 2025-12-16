# Recipe Format Proposal

## Problem

SGLang changes across versions. Users need to track:
- Models
- Container versions
- Environment variables 
- SGLang CLI flags
- Benchmark configurations

Currently, these are scattered across scripts, docs, and tribal knowledge.

## Solution

Single YAML format that captures everything needed to reproduce a configuration.

## Format

To see a full spec please see `spec/v1/schema.yaml`

```yaml
name: "recipe-name"

# Versioning
sglang_version: "v0.5.5.post2"  # Container/SGLang version used

model:
  path: "deepseek-ai/DeepSeek-R1"
  precision: "fp8"  # fp4, fp8, fp16, bf16

resources:
  gpu_type: "gb200"  # gb200, gb300, h100

  # Disaggregated (separate prefill/decode)
  prefill_nodes: 1
  decode_nodes: 2
  prefill_workers: 1
  decode_workers: 2

  # OR Aggregated (combined)
  # nodes: 4
  # workers: 4

# Environment variables (version-specific, perf-critical)
environment:
  prefill:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
    PYTHONUNBUFFERED: "1"
    SGLANG_ENABLE_FLASHINFER_GEMM: "1"
    # ... GB200-specific vars

  decode:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
    PYTHONUNBUFFERED: "1"
    # ... version-specific tunables

# SGLang CLI flags (passed directly)
sglang_config:
  prefill:
    disaggregation-mode: "prefill"
    served-model-name: "deepseek-ai/DeepSeek-R1"
    model-path: "/model/"
    trust-remote-code: true
    kv-cache-dtype: "fp8_e4m3"
    tensor-parallel-size: 4
    max-total-tokens: 32768
    chunked-prefill-size: 8192
    # ... all flags version-specific

  decode:
    disaggregation-mode: "decode"
    served-model-name: "deepseek-ai/DeepSeek-R1"
    model-path: "/model/"
    trust-remote-code: true
    kv-cache-dtype: "fp8_e4m3"
    tensor-parallel-size: 4
    # ... version-specific flags

# Benchmark config
benchmark:
  type: "sa-bench"  # or manual, mmlu, gpqa
  isl: 1024
  osl: 1024
  concurrencies: "4x8x16x32x64"
  req_rate: "inf"
```

## Benefits

1. **Version Control**: Each recipe explicitly states SGLang version
2. **Reproducibility**: Complete config in one file
3. **Portability**: Works across SLURM, K8s, local, production
4. **Data Driven**: Any tool can parse and share relevant pieces of the spec
5. **Backward Compatibility**: Old recipes remain valid references

