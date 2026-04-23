"""
Curriculum Manager
==================
Controls the three training stages.

Stage 1  (0 → stage1_end):
    flat_bet fixed, bet head frozen. Goal: learn basic strategy.

Stage 2  (stage1_end → stage2_end):
    flat_bet fixed, bet head frozen. Goal: count-aware play decisions.

Stage 3  (stage2_end → ∞):
    bet head unfrozen. Goal: count-dependent bet sizing.

Auxiliary loss annealing
-------------------------
λ is held at 1.0 for the entire stage 1 + stage 2 period, then annealed
to 0 linearly over `aux_anneal_steps` steps beginning at stage2_end.

Rationale: the shoe encoder needs a strong count-prediction signal
throughout stages 1 and 2. Annealing before the encoder has learned a
useful representation wastes the only dense supervised signal available.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CurriculumConfig:
    stage1_end:        int   = 200_000    # step at which stage 1 ends
    stage2_end:        int   = 1_000_000  # step at which stage 2 ends
    aux_anneal_steps:  int   = 500_000    # steps to anneal λ → 0 after stage2_end
    flat_bet:          float = 10.0       # chip bet used in stages 1 & 2
    use_bet_curriculum: bool = True       # if False: bet head trains freely from step 0


class CurriculumManager:
    def __init__(self, cfg: CurriculumConfig | None = None):
        self.cfg = cfg or CurriculumConfig()

    def get_stage(self, step: int) -> int:
        if step < self.cfg.stage1_end:
            return 1
        elif step < self.cfg.stage2_end:
            return 2
        return 3

    def bet_head_frozen(self, step: int) -> bool:
        if not self.cfg.use_bet_curriculum:
            return False
        return step < self.cfg.stage2_end

    def flat_bet(self, step: int) -> float | None:
        if not self.cfg.use_bet_curriculum:
            return None
        if step < self.cfg.stage2_end:
            return self.cfg.flat_bet
        return None

    def aux_loss_weight(self, step: int) -> float:
        """
        λ = 1.0 during stages 1 and 2 (full strength while encoder is learning).
        λ anneals 1.0 → 0.0 linearly over aux_anneal_steps after stage2_end.
        """
        anneal_start = self.cfg.stage2_end
        anneal_end   = anneal_start + self.cfg.aux_anneal_steps
        if step <= anneal_start:
            return 1.0
        if step >= anneal_end:
            return 0.0
        return 1.0 - (step - anneal_start) / self.cfg.aux_anneal_steps

    def describe(self, step: int) -> str:
        stage = self.get_stage(step)
        lam   = self.aux_loss_weight(step)
        flat  = self.flat_bet(step)
        return (
            f"Stage {stage} | "
            f"bet={'fixed' if flat else 'learned'} | "
            f"bet_head={'frozen' if self.bet_head_frozen(step) else 'live'} | "
            f"λ_aux={lam:.3f}"
        )
