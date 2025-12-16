#!/usr/bin/env python3
"""
Convert YAML configuration to bash commands with environment variables.

This module provides templating functionality to convert structured YAML configs
into executable bash commands with proper flag formatting and environment variable handling.
"""

from typing import Any


def expand_template(template: Any, values: dict[str, Any]) -> Any:
    """Recursively expand template strings with values.

    Args:
        template: Template object (dict, list, str, or other)
        values: Dictionary of parameter values to substitute

    Returns:
        Expanded template with {param} placeholders replaced

    Example:
        >>> template = {"command": "python train.py --model {model}"}
        >>> values = {"model": "gpt-3"}
        >>> expand_template(template, values)
        {'command': 'python train.py --model gpt-3'}
    """
    if isinstance(template, dict):
        return {k: expand_template(v, values) for k, v in template.items()}
    elif isinstance(template, list):
        return [expand_template(item, values) for item in template]
    elif isinstance(template, str):
        result = template
        for key, value in values.items():
            placeholder = f"{{{key}}}"
            # Handle list values specially - convert to comma-separated string or keep as list
            if isinstance(value, list):
                # If the entire string is just the placeholder, replace with the list
                if placeholder in result and result == placeholder:
                    return value
                else:
                    # If it's embedded in a string, convert to comma-separated
                    result = result.replace(placeholder, ",".join(str(v) for v in value))
            else:
                result = result.replace(placeholder, str(value))
        return result
    else:
        return template


def config_to_flags(config: dict) -> list[str]:
    """Convert config dict to CLI flags.

    Args:
        config: Configuration dict with keys as flag names

    Returns:
        List of flag strings with backslash continuations

    Example:
        >>> config = {"model-path": "/models/gpt", "port": 8080, "verbose": True}
        >>> config_to_flags(config)
        ['    --model-path /models/gpt \\', '    --port 8080 \\', '    --verbose \\']
    """
    lines = []

    for key, value in sorted(config.items()):
        # Convert underscores to hyphens for CLI flags
        flag_name = key.replace("_", "-")

        if isinstance(value, bool):
            if value:
                lines.append(f"    --{flag_name} \\")
        elif isinstance(value, list):
            values_str = " ".join(str(v) for v in value)
            lines.append(f"    --{flag_name} {values_str} \\")
        else:
            lines.append(f"    --{flag_name} {value} \\")

    return lines


def render_command(
    base_command: str,
    config: dict,
    environment: dict[str, str] = None
) -> str:
    """Render full command with environment variables and flags.

    Args:
        base_command: Base command to execute (e.g., "python -m server")
        config: Configuration dict to convert to flags
        environment: Optional environment variables to prepend

    Returns:
        Multi-line bash command string with env vars and flags

    Example:
        >>> render_command(
        ...     "python train.py",
        ...     {"epochs": 10, "lr": 0.001},
        ...     {"CUDA_VISIBLE_DEVICES": "0,1"}
        ... )
        'CUDA_VISIBLE_DEVICES=0,1 \\\\npython train.py \\\\n    --epochs 10 \\\\n    --lr 0.001 \\\\'
    """
    lines = []

    # Environment variables
    if environment:
        for key, val in environment.items():
            lines.append(f"{key}={val} \\")

    # Base command
    lines.append(f"{base_command} \\")

    # Flags from config
    flag_lines = config_to_flags(config)
    lines.extend(flag_lines)

    # Remove trailing backslash from last line
    if lines and lines[-1].endswith(" \\"):
        lines[-1] = lines[-1][:-2]

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage with the actual spec format
    import yaml

    # Example recipe YAML matching the spec format
    recipe_yaml = """
    name: "deepseek-r1-disagg-test"
    sglang_version: "v0.5.5.post2"

    model:
      path: "deepseek-ai/DeepSeek-R1"
      precision: "fp8"

    resources:
      gpu_type: "gb200"
      prefill_nodes: 1
      decode_nodes: 2
      prefill_workers: 1
      decode_workers: 2

    environment:
      prefill:
        TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
        PYTHONUNBUFFERED: "1"
        SGLANG_ENABLE_FLASHINFER_GEMM: "1"
      decode:
        TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
        PYTHONUNBUFFERED: "1"

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
      decode:
        disaggregation-mode: "decode"
        served-model-name: "deepseek-ai/DeepSeek-R1"
        model-path: "/model/"
        trust-remote-code: true
        kv-cache-dtype: "fp8_e4m3"
        tensor-parallel-size: 4

    benchmark:
      type: "sa-bench"
      isl: 1024
      osl: 1024
      concurrencies: "4x8x16x32x64"
      req_rate: "inf"
    """

    recipe = yaml.safe_load(recipe_yaml)

    # Show the input YAML
    print("=" * 60)
    print("DISAGGREGATED MODE RECIPE (YAML)")
    print("=" * 60)
    print(recipe_yaml.strip())
    print()

    # Generate prefill worker command
    print("=" * 60)
    print("PREFILL WORKER COMMAND")
    print("=" * 60)
    prefill_command = render_command(
        base_command="python -m sglang.launch_server",
        config=recipe["sglang_config"]["prefill"],
        environment=recipe["environment"]["prefill"]
    )
    print(prefill_command)
    print()

    # Generate decode worker command
    print("=" * 60)
    print("DECODE WORKER COMMAND")
    print("=" * 60)
    decode_command = render_command(
        base_command="python -m sglang.launch_server",
        config=recipe["sglang_config"]["decode"],
        environment=recipe["environment"]["decode"]
    )
    print(decode_command)
    print()

    # Aggregated mode example
    print("=" * 60)
    print("AGGREGATED MODE RECIPE (YAML)")
    print("=" * 60)

    agg_recipe_yaml = """
    name: "example-agg-job"

    model:
      path: "deepseek-ai/DeepSeek-V3.2"
      container: "lmsysorg/sglang:v0.5.5.post2"

    resources:
      gpu_type: "l40s"
      agg_nodes: 1
      agg_workers: 1
      gpus_per_node: 8

    backend:
      aggregated_environment:
        TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
        PYTHONUNBUFFERED: "1"
        SGLANG_ENABLE_FLASHINFER_GEMM: "1"
      sglang_config:
        aggregated:
          served-model-name: "deepseek-ai/DeepSeek-R1"
          model-path: "/model/"
          trust-remote-code: true
          kv-cache-dtype: "fp8_e4m3"
          tensor-parallel-size: 8
          quantization: fp8
          disaggregation-mode: "prefill"

    benchmark:
      type: "sa-bench"
      isl: 1024
      osl: 1024
      concurrencies: [256, 512]
      req_rate: "inf"
    """

    print(agg_recipe_yaml.strip())
    print()

    print("=" * 60)
    print("AGGREGATED WORKER COMMAND")
    print("=" * 60)

    agg_recipe = yaml.safe_load(agg_recipe_yaml)

    agg_command = render_command(
        base_command="python -m sglang.launch_server",
        config=agg_recipe["backend"]["sglang_config"]["aggregated"],
        environment=agg_recipe["backend"]["aggregated_environment"]
    )
    print(agg_command)
    print("=" * 60)
