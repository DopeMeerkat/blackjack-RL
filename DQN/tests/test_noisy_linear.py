"""Tests for NoisyLinear and BlackjackNet."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# NoisyLinear tests
# ---------------------------------------------------------------------------

class TestNoisyLinear:
    def _make(self, p=8, q=16, sigma0=0.5):
        from agent.noisy_linear import NoisyLinear
        return NoisyLinear(p, q, sigma0)

    def test_output_shape(self):
        layer = self._make(8, 16)
        x = torch.randn(4, 8)
        y = layer(x)
        assert y.shape == (4, 16)

    def test_parameter_shapes(self):
        p, q = 8, 16
        layer = self._make(p, q)
        assert layer.mu_w.shape    == (q, p)
        assert layer.sigma_w.shape == (q, p)
        assert layer.mu_b.shape    == (q,)
        assert layer.sigma_b.shape == (q,)

    def test_mu_init_in_range(self):
        p = 64
        layer = self._make(p, 128)
        bound = 1.0 / math.sqrt(p)
        assert layer.mu_w.abs().max().item() <= bound + 1e-6
        assert layer.mu_b.abs().max().item() <= bound + 1e-6

    def test_sigma_init_value(self):
        p, sigma0 = 64, 0.5
        layer = self._make(p, 128, sigma0)
        expected = sigma0 / math.sqrt(p)
        assert torch.allclose(layer.sigma_w, torch.full_like(layer.sigma_w, expected), atol=1e-6)
        assert torch.allclose(layer.sigma_b, torch.full_like(layer.sigma_b, expected), atol=1e-6)

    def test_noise_buffers_shape(self):
        p, q = 8, 16
        layer = self._make(p, q)
        assert layer.eps_w.shape == (q, p)
        assert layer.eps_b.shape == (q,)

    def test_reset_noise_changes_buffers(self):
        layer = self._make(32, 64)
        eps_w_before = layer.eps_w.clone()
        # Resample multiple times until noise changes (probabilistic, but almost
        # certainly different after a second sample).
        changed = False
        for _ in range(10):
            layer.reset_noise()
            if not torch.allclose(layer.eps_w, eps_w_before):
                changed = True
                break
        assert changed, "reset_noise() should produce different noise"

    def test_stochastic_vs_deterministic_output(self):
        """Stochastic output should differ from deterministic (with non-zero sigma)."""
        layer = self._make(32, 64)
        x = torch.randn(4, 32)

        layer.reset_noise()
        out_stochastic = layer(x).detach()

        layer.set_deterministic(True)
        out_deterministic = layer(x).detach()
        layer.set_deterministic(False)

        # They should differ (sigma > 0 means noise adds variation).
        # Theoretically could be equal with probability 0 — run a few times.
        differs = not torch.allclose(out_stochastic, out_deterministic, atol=1e-6)
        for _ in range(5):
            if differs:
                break
            layer.reset_noise()
            out_stochastic = layer(x).detach()
            differs = not torch.allclose(out_stochastic, out_deterministic, atol=1e-6)
        assert differs, "Stochastic and deterministic outputs should differ"

    def test_deterministic_is_pure_mu(self):
        """Deterministic mode must equal exactly mu_w @ x + mu_b."""
        layer = self._make(16, 8)
        x = torch.randn(2, 16)
        layer.set_deterministic(True)
        out = layer(x)
        expected = torch.nn.functional.linear(x, layer.mu_w, layer.mu_b)
        assert torch.allclose(out, expected, atol=1e-6)
        layer.set_deterministic(False)

    def test_noise_is_parameters_are_learned(self):
        """mu_w, sigma_w, mu_b, sigma_b must be leaf Parameters (have grad)."""
        layer = self._make()
        param_names = {n for n, _ in layer.named_parameters()}
        for name in ("mu_w", "sigma_w", "mu_b", "sigma_b"):
            assert name in param_names, f"{name} not in parameters"

    def test_eps_not_parameters(self):
        """eps_w and eps_b must be buffers, not learnable parameters."""
        layer = self._make()
        param_names = {n for n, _ in layer.named_parameters()}
        assert "eps_w" not in param_names
        assert "eps_b" not in param_names

    def test_factorized_noise_structure(self):
        """eps_w must be an outer product: eps_out ⊗ eps_in."""
        layer = self._make(4, 8)
        layer.reset_noise()
        # Check that eps_w rows are proportional (outer-product structure).
        # Row i of eps_w = eps_out[i] * eps_in.
        # So eps_w[i] / eps_w[j] should be a scalar for all i, j where eps_out[j] ≠ 0.
        eps_w = layer.eps_w.detach()
        # Find first non-zero row.
        for i in range(eps_w.shape[0]):
            if eps_w[i].abs().max() > 1e-8:
                for j in range(eps_w.shape[0]):
                    if eps_w[j].abs().max() > 1e-8:
                        ratio = eps_w[i] / eps_w[j]
                        # ratio should be constant across all columns.
                        assert ratio.std().item() < 1e-5, (
                            "eps_w rows not proportional — factorized structure broken"
                        )
                break

    def test_gradient_flows_through_layer(self):
        """Loss backward should produce non-zero gradients for mu_w, sigma_w."""
        layer = self._make(8, 16)
        layer.reset_noise()
        x = torch.randn(4, 8)
        loss = layer(x).sum()
        loss.backward()
        assert layer.mu_w.grad is not None
        assert layer.sigma_w.grad is not None
        assert layer.mu_w.grad.abs().sum().item() > 0

    def test_device_consistency_cpu(self):
        """Buffers and parameters should all be on CPU."""
        layer = self._make()
        for name, buf in layer.named_buffers():
            assert buf.device.type == "cpu", f"Buffer {name} not on CPU"
        for name, param in layer.named_parameters():
            assert param.device.type == "cpu", f"Param {name} not on CPU"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_reset_noise_on_gpu(self):
        layer = self._make().cuda()
        layer.reset_noise()
        assert layer.eps_w.device.type == "cuda"
        x = torch.randn(2, 8).cuda()
        out = layer(x)
        assert out.device.type == "cuda"


# ---------------------------------------------------------------------------
# BlackjackNet tests
# ---------------------------------------------------------------------------

class TestBlackjackNet:
    def _cfg(self):
        return {
            "obs_dim": 28,
            "trunk_hidden": 64,   # small for speed
            "head_hidden": 64,
            "playing_actions": 4,
            "bet_actions": 5,
            "noisy_sigma0": 0.5,
        }

    def _make(self):
        from agent.network import BlackjackNet
        return BlackjackNet(self._cfg())

    def test_output_shapes(self):
        net = self._make()
        x = torch.randn(8, 28)
        play_q, bet_q = net(x)
        assert play_q.shape == (8, 4)
        assert bet_q.shape  == (8, 5)

    def test_reset_noise_affects_output(self):
        net = self._make()
        x = torch.randn(4, 28)
        net.reset_noise()
        out1 = net(x)[0].detach()
        # After another reset, output should differ (almost certainly).
        changed = False
        for _ in range(10):
            net.reset_noise()
            out2 = net(x)[0].detach()
            if not torch.allclose(out1, out2, atol=1e-6):
                changed = True
                break
        assert changed

    def test_set_deterministic_propagates(self):
        from agent.noisy_linear import NoisyLinear
        net = self._make()
        net.set_deterministic(True)
        for m in net.modules():
            if isinstance(m, NoisyLinear):
                assert m.deterministic is True
        net.set_deterministic(False)
        for m in net.modules():
            if isinstance(m, NoisyLinear):
                assert m.deterministic is False

    def test_deterministic_output_reproducible(self):
        """Same obs → same output when deterministic."""
        net = self._make()
        net.set_deterministic(True)
        x = torch.randn(4, 28)
        out1 = net(x)[0].detach()
        out2 = net(x)[0].detach()
        assert torch.allclose(out1, out2)
        net.set_deterministic(False)

    def test_select_play_actions_respects_mask(self):
        """Masked (illegal) actions must never be selected."""
        net = self._make()
        net.set_deterministic(True)
        x = torch.randn(16, 28)
        # Mask out actions 2 and 3 (double and split) for all envs.
        mask = torch.ones(16, 4, dtype=torch.bool)
        mask[:, 2] = False
        mask[:, 3] = False
        actions = net.select_play_actions(x, mask)
        assert actions.shape == (16,)
        assert (actions < 2).all(), "Should only select hit(0) or stand(1)"

    def test_trunk_uses_standard_linear(self):
        """Trunk layers must NOT be NoisyLinear."""
        from agent.noisy_linear import NoisyLinear
        net = self._make()
        for layer in net.trunk:
            assert not isinstance(layer, NoisyLinear), \
                "Trunk should use standard Linear, not NoisyLinear"

    def test_heads_use_noisy_linear(self):
        """Each head must contain NoisyLinear layers (4 per dueling head: 2 per stream)."""
        from agent.noisy_linear import NoisyLinear
        net = self._make()
        expected = 4 if net.dueling else 2
        for head in (net.play_head, net.bet_head):
            noisy_count = sum(1 for m in head.modules() if isinstance(m, NoisyLinear))
            assert noisy_count == expected, (
                f"Expected {expected} NoisyLinear in head, got {noisy_count}"
            )

    def test_parameter_count_reasonable(self):
        """Full-size net should have ~400K parameters (within an order of magnitude)."""
        from agent.network import BlackjackNet
        cfg = {
            "obs_dim": 28, "trunk_hidden": 256, "head_hidden": 256,
            "playing_actions": 4, "bet_actions": 5, "noisy_sigma0": 0.5,
        }
        net = BlackjackNet(cfg)
        n_params = sum(p.numel() for p in net.parameters())
        assert 100_000 < n_params < 2_000_000, f"Unexpected parameter count: {n_params:,}"

    def test_gradient_flows_to_trunk(self):
        """Loss from play head should produce gradients in trunk parameters."""
        net = self._make()
        net.reset_noise()
        x = torch.randn(4, 28, requires_grad=False)
        play_q, _ = net(x)
        loss = play_q.sum()
        loss.backward()
        # First trunk Linear should have grad
        first_linear = net.trunk[0]
        assert first_linear.weight.grad is not None
        assert first_linear.weight.grad.abs().sum().item() > 0
