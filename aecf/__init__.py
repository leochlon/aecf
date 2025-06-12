"""AECF – Adaptive Ensemble CLIP Fusion.

Import conveniences::

    from aecf import AECFLayer
"""

# Import the modular AECF components
from .proper_aecf_core import (
    CurriculumMasking,
    MultimodalAttentionPool,
    multimodal_attention_pool,
    create_fusion_pool
)

__all__ = [
    'CurriculumMasking',
    'MultimodalAttentionPool',
    'multimodal_attention_pool',
    'create_fusion_pool'
]
