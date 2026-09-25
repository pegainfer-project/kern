"""Attention backends that compute nothing.

vLLM derives its whole memory plan from the layers' backends: the kernel
block size, the KV layout, CUDA graph support. These declare what the
manifest's kernels were written against and hand the per-step metadata
through untouched; the model reads it and drives kern.
"""
from typing import ClassVar

import torch

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

KERNEL_BLOCK = 64


class KernBuilder(AttentionMetadataBuilder[CommonAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    reorder_batch_threshold: int | None = 1

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False, **_):
        return common_attn_metadata


class KernImpl(AttentionImpl):
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError("kern runs the whole model; no layer forward is called")


class KernAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list] = ["auto", "bfloat16"]
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "KERN"

    @staticmethod
    def get_impl_cls():
        return KernImpl

    @staticmethod
    def get_builder_cls():
        return KernBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [KERNEL_BLOCK]

    @classmethod
    def supported_kv_cache_layouts(cls):
        return (KVCacheLayout.LBNHC,)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []


class KernStateBackend(KernAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "KERN_STATE"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return []

    @classmethod
    def supported_kv_cache_layouts(cls):
        return None

    @classmethod
    def is_ssm(cls) -> bool:
        return True
