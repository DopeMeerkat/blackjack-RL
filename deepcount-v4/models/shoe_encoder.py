"""
Shoe Encoder
============
A small Transformer that reads the sliding window of recently-seen cards
and produces a fixed-size "shoe state" vector summarising remaining
deck composition.

Design rationale
----------------
- Full (non-causal) self-attention over the window: the window is already
  a chronological prefix, so no future leakage to mask.
- Positional encodings intentionally OMITTED: Hi-Lo count (and any richer
  statistic) depends only on card frequencies, not their order within the
  window. The model can discover whether order matters.
- Mean-pool over real (non-pad) tokens → permutation-invariant aggregation.
- Auxiliary head predicts the Hi-Lo true count, giving a supervised signal
  that jump-starts useful representations.

NaN safety
----------
The same LayerNorm / all-pad NaN that affects the hand encoder also
affects the Transformer when every position in the window is the UNSEEN
pad token (i.e. at the very start of a shoe before any cards are visible).
Fix: short-circuit the Transformer for all-pad rows and write zeros there.
We also remove norm_first=True (Pre-LN) because PyTorch's fused
TransformerEncoder path doesn't support it cleanly and emits a warning.
Post-LN is numerically identical given our small depth.
"""

import math
import torch
import torch.nn as nn

from env.card_utils import VOCAB_SIZE


class ShoeEncoder(nn.Module):
    """
    Transformer encoder over a window of recently-seen card ranks.

    Args:
        vocab_size : number of unique tokens (ranks 0-13; 0 = UNSEEN pad)
        d_model    : token embedding / hidden dimension (default 64)
        n_heads    : number of attention heads (default 4)
        n_layers   : number of Transformer encoder layers (default 2)
        d_ff       : feed-forward inner dimension (default 256)
        dropout    : dropout probability (default 0.1)
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # No padding_idx: token 0 must learn a real embedding so that
        # LayerNorm never sees an all-zero sequence inside the transformer.
        self.embed = nn.Embedding(vocab_size, d_model)

        # Post-LN encoder layers (avoids the norm_first NaN / warning)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,   # Post-LN: stable and warning-free
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

        # Auxiliary head: predict Hi-Lo true count from shoe_state
        self.aux_count_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

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
        shoe_history: torch.Tensor,   # (B, WINDOW) long
        return_aux: bool = False,
    ):
        """
        Args:
            shoe_history : LongTensor (B, WINDOW) — card rank indices,
                           0 = UNSEEN padding token.
            return_aux   : if True, also return predicted true-count (B, 1).

        Returns:
            shoe_state        : FloatTensor (B, d_model)
            (optional) count_pred : FloatTensor (B, 1)
        """
        B, W = shoe_history.shape
        device = shoe_history.device

        pad_mask  = (shoe_history == 0)           # (B, W) True = pad
        has_cards = ~pad_mask.all(dim=1)          # (B,) True = at least one real card

        # Embed all tokens (pad token 0 has a learned non-zero embedding)
        x = self.embed(shoe_history) * math.sqrt(self.d_model)   # (B, W, d_model)

        # Run Transformer only on rows that have at least one real card
        if has_cards.any():
            idx    = has_cards.nonzero(as_tuple=True)[0]
            x_sub  = x[idx]                          # (M, W, d_model)
            pm_sub = pad_mask[idx]                   # (M, W)

            x_sub = self.transformer(x_sub, src_key_padding_mask=pm_sub)

            x = x.clone()
            x[idx] = x_sub

        # Masked mean-pool over real tokens
        real_mask  = (~pad_mask).float().unsqueeze(-1)     # (B, W, 1)
        sum_repr   = (x * real_mask).sum(dim=1)            # (B, d_model)
        n_real     = real_mask.sum(dim=1).clamp(min=1.0)   # (B, 1)
        shoe_state = sum_repr / n_real                     # (B, d_model)

        # Zero out all-pad rows so garbage doesn't propagate
        shoe_state = shoe_state * has_cards.float().unsqueeze(-1)

        # LayerNorm only on rows with real content; others stay zero
        if has_cards.any():
            idx = has_cards.nonzero(as_tuple=True)[0]
            shoe_state = shoe_state.clone()
            shoe_state[idx] = self.out_norm(shoe_state[idx])

        if return_aux:
            count_pred = self.aux_count_head(shoe_state)
            return shoe_state, count_pred

        return shoe_state
