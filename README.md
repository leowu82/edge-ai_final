# Edge AI Final Project

This repository contains the final project for edge AI, focusing on deploying machine learning models on edge devices.

## Description

This project demonstrates the deployment of a machine learning model on edge devices using TensorFlow Lite. It includes:
- Model training (if applicable)
- Model conversion to TensorFlow Lite format
- Inference on edge devices
- Example use cases for real-time processing

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Model Structure
```bash
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 3072)
    (layers): ModuleList(
      (0-27): 28 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=3072, out_features=3072, bias=False)
          (k_proj): Linear(in_features=3072, out_features=1024, bias=False)
          (v_proj): Linear(in_features=3072, out_features=1024, bias=False)
          (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=3072, out_features=8192, bias=False)
          (up_proj): Linear(in_features=3072, out_features=8192, bias=False)
          (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((3072,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((3072,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=3072, out_features=128256, bias=False)
)
```
## Principles
Every token you generate incurs three broad costs:

Compute (FLOPs) to do the attention + feed-forward.

Memory movement for weights, activations, and KV cache.

CPU↔GPU overhead for driver launches, Python loops, and data transfers.

$$Throughput (tokens/sec) =

\frac{\text{# tokens generated}}{\text{Compute time} + \text{Memory time} + \text{Overhead}}$$

To maximize throughput you must:

- Minimize compute time (e.g. by quantizing to lower bits, using faster kernels)

- Minimize memory time (e.g. by coalescing, using Tensor Cores, reducing model size)

- Minimize overhead (e.g. by fusing GPU launches, reducing Python overhead)