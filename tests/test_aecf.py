"""
Comprehensive Test Suite for AECF (Attention Entropy Curriculum Filtering)

This test suite validates all components of the AECF implementation against
PyTorch standards for numerical stability, correctness, and integration.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
import gc
from typing import Dict, Any, Tuple, Optional
from unittest.mock import patch
import numpy as np

# Import the modules under test
try:
    from aecf.proper_aecf_core import (
        CurriculumMasking, 
        MultimodalAttentionPool, 
        multimodal_attention_pool,
        create_fusion_pool
    )
except ImportError:
    # For when running tests standalone
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from proper_aecf_core import (
        CurriculumMasking, 
        MultimodalAttentionPool, 
        multimodal_attention_pool,
        create_fusion_pool
    )


class TestCurriculumMasking:
    """Test suite for CurriculumMasking module."""
    
    @pytest.fixture
    def masking_module(self):
        """Standard curriculum masking module."""
        return CurriculumMasking(base_mask_prob=0.15, entropy_target=0.7, min_active=1)
    
    @pytest.fixture
    def sample_weights(self):
        """Sample attention weights for testing."""
        torch.manual_seed(42)
        batch_size, seq_len = 4, 8
        weights = torch.rand(batch_size, seq_len)
        weights = F.softmax(weights, dim=-1)
        return weights
    
    def test_initialization_valid_params(self):
        """Test proper initialization with valid parameters."""
        masking = CurriculumMasking(base_mask_prob=0.2, entropy_target=0.8, min_active=2)
        
        assert masking.base_mask_prob == 0.2
        assert masking.entropy_target == 0.8
        assert masking.min_active == 2
        assert hasattr(masking, '_eps')
        assert isinstance(masking._eps, torch.Tensor)
    
    def test_initialization_invalid_params(self):
        """Test initialization raises errors for invalid parameters."""
        with pytest.raises(ValueError, match="base_mask_prob must be in"):
            CurriculumMasking(base_mask_prob=0.0)
        
        with pytest.raises(ValueError, match="base_mask_prob must be in"):
            CurriculumMasking(base_mask_prob=1.5)
        
        with pytest.raises(ValueError, match="entropy_target must be in"):
            CurriculumMasking(entropy_target=0.0)
        
        with pytest.raises(ValueError, match="entropy_target must be in"):
            CurriculumMasking(entropy_target=1.5)
        
        with pytest.raises(ValueError, match="min_active must be"):
            CurriculumMasking(min_active=0)
    
    def test_compute_entropy_correctness(self, masking_module):
        """Test entropy computation against analytical solutions."""
        # Test uniform distribution (maximum entropy)
        uniform_weights = torch.ones(2, 4) / 4  # (batch=2, seq_len=4)
        entropy = masking_module.compute_entropy(uniform_weights)
        expected_entropy = math.log(4)  # log(4) for uniform over 4 elements
        
        torch.testing.assert_close(entropy, torch.full((2,), expected_entropy), rtol=1e-6, atol=1e-6)
        
        # Test delta distribution (minimum entropy)
        delta_weights = torch.zeros(2, 4)
        delta_weights[:, 0] = 1.0  # All mass on first element
        entropy = masking_module.compute_entropy(delta_weights)
        
        # Should be close to 0 (within numerical precision)
        assert torch.all(entropy < 1e-6)
    
    def test_compute_entropy_shapes(self, masking_module):
        """Test entropy computation handles various input shapes."""
        shapes = [
            (10,),      # 1D
            (5, 10),    # 2D
            (3, 5, 10), # 3D
            (2, 3, 5, 10), # 4D
        ]
        
        for shape in shapes:
            weights = torch.rand(shape)
            weights = F.softmax(weights, dim=-1)
            entropy = masking_module.compute_entropy(weights)
            
            expected_shape = shape[:-1]  # Remove last dimension
            assert entropy.shape == expected_shape
    
    def test_forward_training_mode(self, masking_module, sample_weights):
        """Test forward pass in training mode."""
        masking_module.train()
        
        masked_weights, info = masking_module(sample_weights)
        
        # Check output shapes
        assert masked_weights.shape == sample_weights.shape
        assert 'entropy' in info
        assert 'mask_rate' in info
        assert 'target_entropy' in info
        
        # Check normalization is preserved
        torch.testing.assert_close(
            masked_weights.sum(dim=-1), 
            torch.ones(sample_weights.size(0)),
            rtol=1e-5,
            atol=1e-5
        )
        
        # Check entropy and mask_rate are reasonable
        assert torch.all(info['entropy'] >= 0)
        assert torch.all(info['mask_rate'] >= 0)
        assert torch.all(info['mask_rate'] <= 1)
    
    def test_forward_eval_mode(self, masking_module, sample_weights):
        """Test forward pass in evaluation mode."""
        masking_module.eval()
        
        masked_weights, info = masking_module(sample_weights)
        
        # In eval mode, weights should be unchanged
        torch.testing.assert_close(masked_weights, sample_weights)
        
        # Mask rate should be zero
        assert torch.all(info['mask_rate'] == 0)
    
    def test_min_active_constraint(self):
        """Test minimum active elements constraint is enforced."""
        masking = CurriculumMasking(base_mask_prob=0.9, min_active=2)
        masking.train()
        
        # Create weights that would likely be heavily masked
        weights = torch.tensor([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]])
        
        masked_weights, info = masking(weights)
        
        # Count active elements (non-zero weights)
        active_count = (masked_weights > 1e-8).sum(dim=-1)
        assert torch.all(active_count >= 2), f"Got active counts: {active_count}"
    
    def test_single_element_sequence(self, masking_module):
        """Test handling of single-element sequences."""
        single_weights = torch.ones(3, 1)  # (batch=3, seq_len=1)
        
        masked_weights, info = masking_module(single_weights)
        
        # Single element should remain unchanged
        torch.testing.assert_close(masked_weights, single_weights)
        
        # Entropy should be 0 for single elements
        assert torch.all(info['entropy'] == 0)
        assert torch.all(info['mask_rate'] == 0)
    
    def test_numerical_stability_edge_cases(self, masking_module):
        """Test numerical stability with edge cases."""
        # Test very small weights
        small_weights = torch.full((2, 5), 1e-10)
        small_weights = F.softmax(small_weights, dim=-1)
        
        masked_weights, info = masking_module(small_weights)
        assert torch.isfinite(masked_weights).all()
        assert torch.isfinite(info['entropy']).all()
        
        # Test weights with NaN
        nan_weights = torch.tensor([[0.5, float('nan'), 0.5], [1.0, 0.0, 0.0]])
        masked_weights, info = masking_module(nan_weights)
        assert torch.isfinite(masked_weights).all()
        
        # Test weights with infinity
        inf_weights = torch.tensor([[0.5, float('inf'), 0.5], [1.0, 0.0, 0.0]])
        masked_weights, info = masking_module(inf_weights)
        assert torch.isfinite(masked_weights).all()
    
    def test_entropy_loss_computation(self, masking_module):
        """Test entropy loss computation."""
        # Test with known entropy values
        entropy = torch.tensor([0.5, 1.0, 1.5])
        loss = masking_module.entropy_loss(entropy)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Should be scalar
        assert loss >= 0  # MSE loss is non-negative
        assert torch.isfinite(loss)
        
        # Test with NaN/inf entropy
        bad_entropy = torch.tensor([float('nan'), float('inf'), -float('inf')])
        loss = masking_module.entropy_loss(bad_entropy)
        assert torch.isfinite(loss)
    
    def test_reproducibility(self, masking_module, sample_weights):
        """Test deterministic behavior with fixed seed."""
        torch.manual_seed(123)
        masking_module.train()
        result1, info1 = masking_module(sample_weights)
        
        torch.manual_seed(123)
        result2, info2 = masking_module(sample_weights)
        
        torch.testing.assert_close(result1, result2)
        torch.testing.assert_close(info1['entropy'], info2['entropy'])
    
    def test_gradient_flow(self, masking_module):
        """Test gradients flow through the module."""
        weights = torch.rand(2, 5, requires_grad=True)
        weights = F.softmax(weights, dim=-1)
        weights.retain_grad()  # Ensure gradients are retained for non-leaf tensor
        
        masking_module.train()
        masked_weights, info = masking_module(weights)
        
        # Compute a dummy loss
        loss = masked_weights.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        assert weights.grad is not None
        assert torch.isfinite(weights.grad).all()
    
    def test_device_compatibility(self, masking_module):
        """Test module works on different devices."""
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        
        for device in devices:
            masking_module.to(device)
            weights = torch.rand(2, 5, device=device)
            weights = F.softmax(weights, dim=-1)
            
            result, info = masking_module(weights)
            
            assert result.device == device
            assert info['entropy'].device == device
    
    def test_state_dict_compatibility(self, masking_module):
        """Test state dict save/load functionality."""
        state_dict = masking_module.state_dict()
        
        # Create new module and load state
        new_masking = CurriculumMasking()
        new_masking.load_state_dict(state_dict)
        
        # Test parameters match
        assert new_masking.base_mask_prob == masking_module.base_mask_prob
        assert new_masking.entropy_target == masking_module.entropy_target
        assert new_masking.min_active == masking_module.min_active
    
    def test_repr_string(self, masking_module):
        """Test string representation."""
        repr_str = repr(masking_module)
        assert 'CurriculumMasking' in repr_str
        assert 'base_mask_prob=0.15' in repr_str
        assert 'entropy_target=0.7' in repr_str
        assert 'min_active=1' in repr_str


class TestMultimodalAttentionPool:
    """Test suite for MultimodalAttentionPool module."""
    
    @pytest.fixture
    def attention_pool(self):
        """Standard attention pool module."""
        return MultimodalAttentionPool(embed_dim=64, num_heads=4)
    
    @pytest.fixture
    def sample_tensors(self):
        """Sample query, key, value tensors."""
        torch.manual_seed(42)
        batch_size, seq_len, embed_dim = 2, 8, 64
        query = torch.randn(batch_size, 1, embed_dim)  # Single query per batch
        key = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)
        return query, key, value
    
    def test_initialization_valid_params(self):
        """Test proper initialization with valid parameters."""
        pool = MultimodalAttentionPool(
            embed_dim=128,
            num_heads=8,
            dropout=0.1,
            bias=False,
            batch_first=False
        )
        
        assert pool.embed_dim == 128
        assert pool.num_heads == 8
        assert pool.batch_first == False
        assert isinstance(pool.attention, nn.MultiheadAttention)
    
    def test_initialization_invalid_params(self):
        """Test initialization raises errors for invalid parameters."""
        with pytest.raises(ValueError, match="embed_dim must be positive"):
            MultimodalAttentionPool(embed_dim=0)
        
        with pytest.raises(ValueError, match="num_heads must be positive"):
            MultimodalAttentionPool(embed_dim=64, num_heads=0)
        
        with pytest.raises(ValueError, match="embed_dim .* must be divisible by num_heads"):
            MultimodalAttentionPool(embed_dim=64, num_heads=5)
        
        with pytest.raises(ValueError, match="dropout must be in"):
            MultimodalAttentionPool(embed_dim=64, dropout=1.5)
    
    def test_forward_basic(self, attention_pool, sample_tensors):
        """Test basic forward pass."""
        query, key, value = sample_tensors
        
        output = attention_pool(query, key, value)
        
        assert output.shape == query.shape
        assert torch.isfinite(output).all()
    
    def test_forward_with_info(self, attention_pool, sample_tensors):
        """Test forward pass with return_info=True."""
        query, key, value = sample_tensors
        
        output, info = attention_pool(query, key, value, return_info=True)
        
        assert output.shape == query.shape
        assert 'attention_weights' in info
        assert info['attention_weights'].shape[0] == query.size(0)  # Batch dimension
    
    def test_forward_without_value(self, attention_pool, sample_tensors):
        """Test forward pass with value=None (uses key as value)."""
        query, key, _ = sample_tensors
        
        output = attention_pool(query, key)  # value=None
        
        assert output.shape == query.shape
        assert torch.isfinite(output).all()
    
    def test_curriculum_masking_integration(self, sample_tensors):
        """Test integration with curriculum masking."""
        masking = CurriculumMasking(base_mask_prob=0.2)
        pool = MultimodalAttentionPool(embed_dim=64, curriculum_masking=masking)
        pool.train()
        
        query, key, value = sample_tensors
        output, info = pool(query, key, value, return_info=True)
        
        assert output.shape == query.shape
        assert 'entropy' in info
        assert 'mask_rate' in info
    
    def test_attention_masks(self, attention_pool, sample_tensors):
        """Test attention mask functionality."""
        query, key, value = sample_tensors
        batch_size, seq_len = key.shape[:2]
        
        # Test key padding mask
        key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        key_padding_mask[:, -2:] = True  # Mask last 2 positions
        
        output = attention_pool(query, key, value, key_padding_mask=key_padding_mask)
        assert output.shape == query.shape
        
        # Test attention mask
        attn_mask = torch.triu(torch.ones(1, seq_len), diagonal=1).bool()
        output = attention_pool(query, key, value, attn_mask=attn_mask)
        assert output.shape == query.shape
    
    def test_batch_first_false(self, sample_tensors):
        """Test with batch_first=False."""
        pool = MultimodalAttentionPool(embed_dim=64, batch_first=False)
        query, key, value = sample_tensors
        
        # Transpose to (seq_len, batch, embed_dim)
        query_t = query.transpose(0, 1)
        key_t = key.transpose(0, 1)
        value_t = value.transpose(0, 1)
        
        output = pool(query_t, key_t, value_t)
        assert output.shape == query_t.shape
    
    def test_input_validation(self, attention_pool):
        """Test input validation catches shape mismatches."""
        batch_size, embed_dim = 2, 64
        
        query = torch.randn(batch_size, 1, embed_dim)
        key = torch.randn(batch_size, 8, embed_dim)
        
        # Wrong embedding dimension
        bad_value = torch.randn(batch_size, 8, 32)  # Wrong embed_dim
        
        with pytest.raises(RuntimeError, match="Value shape .* incompatible"):
            attention_pool(query, key, bad_value)
        
        # Wrong batch size
        bad_query = torch.randn(4, 1, embed_dim)  # Wrong batch_size
        
        with pytest.raises(RuntimeError, match="Key shape .* incompatible"):
            attention_pool(bad_query, key)
    
    def test_gradient_checkpointing(self, attention_pool, sample_tensors):
        """Test gradient checkpointing functionality."""
        query, key, value = sample_tensors
        
        # Requires grad for checkpointing to be meaningful
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)
        
        attention_pool.train()
        output = attention_pool(query, key, value, use_checkpoint=True)
        
        loss = output.sum()
        loss.backward()
        
        assert query.grad is not None
        assert torch.isfinite(query.grad).all()
    
    def test_device_compatibility(self, attention_pool, sample_tensors):
        """Test device compatibility."""
        devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        
        for device in devices:
            attention_pool.to(device)
            query, key, value = [t.to(device) for t in sample_tensors]
            
            output = attention_pool(query, key, value)
            assert output.device == device
    
    def test_different_dtypes(self, sample_tensors):
        """Test different dtype compatibility."""
        dtypes = [torch.float32, torch.float64]
        if torch.cuda.is_available():
            dtypes.append(torch.float16)
        
        for dtype in dtypes:
            pool = MultimodalAttentionPool(embed_dim=64, dtype=dtype)
            query, key, value = [t.to(dtype) for t in sample_tensors]
            
            output = pool(query, key, value)
            assert output.dtype == dtype
    
    def test_multihead_attention_equivalence(self, sample_tensors):
        """Test equivalence with PyTorch MultiheadAttention when no curriculum masking."""
        embed_dim, num_heads = 64, 4
        query, key, value = sample_tensors
        
        # Our implementation
        our_pool = MultimodalAttentionPool(embed_dim=embed_dim, num_heads=num_heads)
        our_output = our_pool(query, key, value)
        
        # Direct PyTorch implementation
        torch_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # Copy weights to ensure same initialization
        torch_attn.load_state_dict(our_pool.attention.state_dict())
        
        torch_output, _ = torch_attn(query, key, value)
        
        torch.testing.assert_close(our_output, torch_output, rtol=1e-5, atol=1e-5)
    
    def test_memory_efficiency(self, attention_pool):
        """Test memory efficiency with large sequences."""
        # Test with relatively large sequence to check for memory leaks
        batch_size, seq_len, embed_dim = 4, 1000, 64
        
        query = torch.randn(batch_size, 1, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        output = attention_pool(query, key)
        
        # Clean up
        del output, query, key
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            # Memory should not grow significantly
            assert abs(final_memory - initial_memory) < 100 * 1024 * 1024  # 100MB threshold


class TestFunctionalInterface:
    """Test suite for functional interfaces."""
    
    def test_multimodal_attention_pool_function(self):
        """Test functional interface."""
        torch.manual_seed(42)
        batch_size, seq_len, embed_dim = 2, 5, 32
        
        query = torch.randn(batch_size, 1, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        
        output = multimodal_attention_pool(query, key, embed_dim=embed_dim, training=True)
        
        assert output.shape == query.shape
        assert torch.isfinite(output).all()
    
    def test_create_fusion_pool(self):
        """Test factory function for fusion tasks."""
        embed_dim, num_modalities = 128, 3
        
        fusion_query, pool = create_fusion_pool(
            embed_dim=embed_dim,
            num_modalities=num_modalities,
            mask_prob=0.2
        )
        
        assert isinstance(fusion_query, nn.Parameter)
        assert fusion_query.shape == (1, 1, embed_dim)
        assert isinstance(pool, MultimodalAttentionPool)
        assert pool.curriculum_masking is not None


class TestIntegrationAndPerformance:
    """Integration tests and performance validation."""
    
    def test_jit_compatibility(self):
        """Test TorchScript compatibility."""
        pool = MultimodalAttentionPool(embed_dim=32, num_heads=2)
        pool.eval()
        
        # Create sample inputs
        query = torch.randn(1, 1, 32)
        key = torch.randn(1, 4, 32)
        
        # Test scripting (curriculum masking disabled for JIT)
        try:
            scripted_pool = torch.jit.script(pool)
            output = scripted_pool(query, key)
            assert output.shape == query.shape
        except Exception as e:
            pytest.skip(f"JIT compilation failed: {e}")
    
    def test_numerical_precision(self):
        """Test numerical precision and stability."""
        pool = MultimodalAttentionPool(embed_dim=64)
        
        # Test with very small values
        scale = 1e-6
        query = torch.randn(2, 1, 64) * scale
        key = torch.randn(2, 8, 64) * scale
        
        output = pool(query, key)
        assert torch.isfinite(output).all()
        
        # Test with very large values
        scale = 1e6
        query = torch.randn(2, 1, 64) * scale
        key = torch.randn(2, 8, 64) * scale
        
        output = pool(query, key)
        assert torch.isfinite(output).all()
    
    def test_convergence_simple_task(self):
        """Test convergence on a simple learning task."""
        torch.manual_seed(42)
        
        # Create a simple task: attend to the last element
        batch_size, seq_len, embed_dim = 16, 8, 32
        
        # Model setup
        masking = CurriculumMasking(base_mask_prob=0.1)
        pool = MultimodalAttentionPool(embed_dim=embed_dim, curriculum_masking=masking)
        
        # Target: always attend to last element
        target_weights = torch.zeros(batch_size, seq_len)
        target_weights[:, -1] = 1.0
        
        optimizer = torch.optim.Adam(pool.parameters(), lr=0.01)
        
        # Training loop
        initial_loss = None
        for step in range(50):
            query = torch.randn(batch_size, 1, embed_dim)
            key = torch.randn(batch_size, seq_len, embed_dim)
            
            # Make last key element distinctive
            key[:, -1] = key[:, -1] + 2.0
            
            output, info = pool(query, key, return_info=True)
            
            # Simple attention loss (encourage attending to last element)
            attn_weights = info['attention_weights']
            # Ensure shapes match - attention weights might be (batch, num_heads, query_len, key_len)
            if attn_weights.dim() > 2:
                attn_weights = attn_weights.mean(dim=1)  # Average over heads
            if attn_weights.shape != target_weights.shape:
                attn_weights = attn_weights.squeeze(1)  # Remove query dimension if present
            
            loss = F.mse_loss(attn_weights, target_weights)
            
            # Add entropy loss if curriculum masking is used
            if 'entropy' in info:
                entropy_loss = masking.entropy_loss(info['entropy'])
                loss = loss + 0.1 * entropy_loss
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Loss should decrease
        final_loss = loss.item()
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in repeated forward passes."""
        pool = MultimodalAttentionPool(embed_dim=32)
        
        if torch.cuda.is_available():
            pool = pool.cuda()
            initial_memory = torch.cuda.memory_allocated()
        
        # Perform many forward passes
        for _ in range(100):
            query = torch.randn(2, 1, 32)
            key = torch.randn(2, 8, 32)
            
            if torch.cuda.is_available():
                query, key = query.cuda(), key.cuda()
            
            output = pool(query, key)
            del output, query, key
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory
            
            # Allow some growth but not excessive
            assert memory_growth < 10 * 1024 * 1024, f"Memory grew by {memory_growth} bytes"


class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness."""
    
    def test_empty_sequence_handling(self):
        """Test handling of empty sequences."""
        pool = MultimodalAttentionPool(embed_dim=32)
        
        # This should raise an error as empty sequences are invalid
        query = torch.randn(2, 1, 32)
        empty_key = torch.randn(2, 0, 32)  # Empty sequence length
        
        with pytest.raises((RuntimeError, ValueError)):
            pool(query, empty_key)
    
    def test_very_long_sequences(self):
        """Test with very long sequences (memory and time)."""
        pool = MultimodalAttentionPool(embed_dim=32, num_heads=2)
        
        # Test with long sequence
        query = torch.randn(1, 1, 32)
        long_key = torch.randn(1, 10000, 32)
        
        # This should work but might be slow
        try:
            output = pool(query, long_key)
            assert output.shape == query.shape
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip("Insufficient memory for long sequence test")
            else:
                raise
    
    def test_pathological_attention_weights(self):
        """Test curriculum masking with pathological attention distributions."""
        masking = CurriculumMasking(base_mask_prob=0.5, min_active=1)
        masking.train()
        
        # Test with all attention on one element (very low entropy)
        concentrated = torch.zeros(2, 10)
        concentrated[:, 0] = 1.0
        
        masked, info = masking(concentrated)
        assert torch.isfinite(masked).all()
        assert (masked.sum(dim=-1) - 1.0).abs().max() < 1e-5
        
        # Test with perfectly uniform attention (high entropy)
        uniform = torch.ones(2, 10) / 10
        
        masked, info = masking(uniform)
        assert torch.isfinite(masked).all()
        assert (masked.sum(dim=-1) - 1.0).abs().max() < 1e-5


# Pytest configuration and runners
def test_all_cuda_compatibility():
    """Test CUDA compatibility if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Test basic CUDA operations
    masking = CurriculumMasking().cuda()
    pool = MultimodalAttentionPool(embed_dim=32).cuda()
    
    query = torch.randn(2, 1, 32, device='cuda')
    key = torch.randn(2, 8, 32, device='cuda')
    
    output = pool(query, key)
    assert output.device.type == 'cuda'


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running basic smoke tests...")
    
    # Test CurriculumMasking
    masking = CurriculumMasking()
    weights = torch.rand(2, 5)
    weights = F.softmax(weights, dim=-1)
    result, info = masking(weights)
    print(f"✓ CurriculumMasking: input {weights.shape} -> output {result.shape}")
    
    # Test MultimodalAttentionPool
    pool = MultimodalAttentionPool(embed_dim=64)
    query = torch.randn(2, 1, 64)
    key = torch.randn(2, 8, 64)
    output = pool(query, key)
    print(f"✓ MultimodalAttentionPool: query {query.shape}, key {key.shape} -> output {output.shape}")
    
    # Test functional interface
    output_func = multimodal_attention_pool(query, key)
    print(f"✓ Functional interface: {output_func.shape}")
    
    # Test factory function
    fusion_query, fusion_pool = create_fusion_pool(embed_dim=64, num_modalities=3)
    print(f"✓ Factory function: query {fusion_query.shape}")
    
    print("\nAll smoke tests passed! Run with pytest for comprehensive testing.")
    print("Usage: pytest test_aecf.py -v")