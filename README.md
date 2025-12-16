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

To see a full spec for agg and disagg please see `/spec/v1`

## Benefits

1. **Version Control**: Each recipe explicitly states SGLang version
2. **Reproducibility**: Complete config in one file
3. **Portability**: Works across SLURM, K8s, local, production
4. **Data Driven**: Any tool can parse and share relevant pieces of the spec
5. **Backward Compatibility**: Old recipes remain valid references

