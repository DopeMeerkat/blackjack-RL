"""
Hand Encoder (Set Transformer)
================================
Encodes the player's current hand as an *unordered set* of cards.

Design rationale
----------------
A standard MLP or RNN would impose an ordering on cards that is
semantically meaningless in blackjack — {7, 8} is the same hand as {8, 7}.
We use a Set Transformer (Lee et al., 2019) which is permutation-equivariant:
it applies self-attention over the (small) set of hand cards, then pools to
a fixed-size representation.

NaN safety
----------
During the BETTING phase the player hand is empty (all-zero tokens).
Two previous attempts to guard this with padding_idx + safe_mask still
produced NaNs because:
  - padding_idx=0 forces the pad embedding to exactly zero, making
    LayerNorm divide-by-zero when every token in a row is the pad token.
  - Unmasking one "dummy" pad token exposes that zero vector to LayerNorm
    and produces NaN activations.

The correct fix is:
  1. Remove padding_idx from the embedding so every token (including 0)
     learns a proper non-zero vector.
  2. Short-circuit the entire attention block for samples whose hand is
     completely empty, writing zeros into those rows AFTER the block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.card_utils import VOCAB_SIZE


class HandEncoder(nn.Module):
    """
    Set Transformer for encoding a blackjack hand.

    Args:
        vocab_size : card vocabulary size (default: VOCAB_SIZE = 14)
        d_hand     : output embedding dimension (default: 32)
        n_heads    : attention heads (default: 4)
        d_ff       : feed-forward inner dimension (default: 128)
        dropout    : dropout probability (default: 0.1)
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_hand: int = 32,
        n_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_hand = d_hand

        # NO padding_idx — token 0 must learn a real embedding so that
        # LayerNorm never sees an all-zero sequence.
        self.embed = nn.Embedding(vocab_size, d_hand)

        # Single Set-Attention Block: multi-head self-attention + FFN
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_hand,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_hand)
        self.ffn = nn.Sequential(
            nn.Linear(d_hand, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_hand),
        )
        self.ffn_norm = nn.LayerNorm(d_hand)

        # Projection: cat(hand_repr, dealer_repr) → d_hand
        self.fusion = nn.Linear(2 * d_hand, d_hand)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        hand: torch.Tensor,           # (B, MAX_HAND) long — 0 = empty slot
        hand_len: torch.Tensor,       # (B,)          long — real card count
        dealer_upcard: torch.Tensor,  # (B,)          long — 0 if BETTING
    ) -> torch.Tensor:
        """
        Returns hand_state : FloatTensor (B, d_hand).
        Guaranteed NaN-free even when hand_len == 0 for every sample.
        """
        B, S = hand.shape
        device = hand.device

        # Which samples have at least one real card?
        has_cards = (hand_len > 0)          # (B,) bool
        empty_rows = ~has_cards             # samples with no hand yet

        # ── Embed all tokens (including pad token 0) ──────────────────────
        x = self.embed(hand)               # (B, S, d_hand)

        # ── Build key-padding mask only for rows that HAVE cards ──────────
        # For empty rows we will skip attention entirely (handled below).
        # For rows with cards, mask the true pad positions.
        pad_mask = (hand == 0)             # (B, S) True = pad

        # ── Run attention only on rows with at least one real card ────────
        if has_cards.any():
            idx = has_cards.nonzero(as_tuple=True)[0]   # indices of non-empty rows
            x_sub      = x[idx]                          # (M, S, d_hand)
            pmask_sub  = pad_mask[idx]                   # (M, S)

            # Pre-LN self-attention
            res = x_sub
            x_sub_norm = self.attn_norm(x_sub)
            attn_out, _ = self.self_attn(
                x_sub_norm, x_sub_norm, x_sub_norm,
                key_padding_mask=pmask_sub,
            )
            x_sub = res + attn_out

            # Pre-LN FFN
            x_sub = x_sub + self.ffn(self.ffn_norm(x_sub))

            # Write back
            x = x.clone()
            x[idx] = x_sub

        # ── Masked mean-pool ─────────────────────────────────────────────
        real_mask = (~pad_mask).float().unsqueeze(-1)      # (B, S, 1)
        hand_repr = (x * real_mask).sum(dim=1)             # (B, d_hand)
        n_cards   = hand_len.float().unsqueeze(-1).clamp(min=1.0)
        hand_repr = hand_repr / n_cards                    # (B, d_hand)

        # Zero out empty-hand rows so their garbage embeddings don't pollute
        hand_repr = hand_repr * has_cards.float().unsqueeze(-1)

        # ── Dealer upcard embedding ───────────────────────────────────────
        dealer_repr = self.embed(dealer_upcard)            # (B, d_hand)

        # ── Fuse hand + dealer ────────────────────────────────────────────
        combined   = torch.cat([hand_repr, dealer_repr], dim=-1)  # (B, 2*d_hand)
        hand_state = F.gelu(self.fusion(combined))                # (B, d_hand)

        return hand_state
