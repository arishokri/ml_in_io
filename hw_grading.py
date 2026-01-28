### Used for calculating final hw grades.abs

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class HWScoreResult:
    avg: float  # homework average in [0, 1]
    dropped_index: Optional[int]  # which HW was "dropped" (None if drop=False)
    effective_weights: np.ndarray  # P_i for non-dropped; P'_d for dropped
    normalized_scores: np.ndarray  # s_i = earned/max


def homework_score(
    earned: Iterable[float],
    max_points: Iterable[float],
    rho: float = 0.75,
    alpha: float = 1.0,
    drop: bool = True,
    tie_break: str = "largest",  # "largest" or "smallest" among ties for lowest s_i
) -> HWScoreResult:
    """
    Compute a homework average with optional 'partial drop' of the lowest homework.

    Definitions:
      s_i = earned_i / max_i
      d   = argmin_i s_i   (lowest normalized score)
      r(P)= rho * ((P - Pmin)/(Pmax - Pmin))**alpha   if Pmax > Pmin, else 0
      P'_d = r(P_d) * P_d
      effective weights: w_i = P_i for i != d, and w_d = P'_d if drop=True

    Returns:
      HWScoreResult with avg in [0, 1] (not multiplied by any course weight).

    Notes:
      - If drop=False: this reduces to the standard points-weighted average.
      - If all max_points are identical, r(P)=0 for all, so dropping makes the
        dropped HW weight 0 (a normal "drop").
      - Ties for lowest s_i can be broken by choosing the largest-max HW ("largest")
        or smallest-max HW ("smallest") among the tied lowest.
    """
    earned = np.asarray(list(earned), dtype=float)
    max_points = np.asarray(list(max_points), dtype=float)

    if earned.shape != max_points.shape:
        raise ValueError("earned and max_points must have the same length.")
    if earned.ndim != 1:
        raise ValueError("earned and max_points must be 1D sequences.")
    if len(earned) == 0:
        raise ValueError("earned/max_points cannot be empty.")

    if not (0.0 <= rho <= 1.0):
        raise ValueError("rho must be in [0, 1].")
    if alpha <= 0:
        raise ValueError("alpha must be > 0.")
    if np.any(max_points <= 0):
        raise ValueError("All max_points must be > 0.")
    if np.any(earned < 0) or np.any(earned - max_points > 1e-12):
        raise ValueError("Earned points must satisfy 0 <= earned_i <= max_points_i.")

    s = earned / max_points

    # Base case: no drop => plain weighted average by max_points
    if not drop:
        avg = float(np.sum(max_points * s) / np.sum(max_points))
        return HWScoreResult(
            avg=avg,
            dropped_index=None,
            effective_weights=max_points.copy(),
            normalized_scores=s,
        )

    # Choose dropped homework (lowest normalized score; tie-breaker selectable)
    min_s = np.min(s)
    tied = np.where(np.isclose(s, min_s))[0]
    if len(tied) == 1:
        d = int(tied[0])
    else:
        if tie_break not in {"largest", "smallest"}:
            raise ValueError("tie_break must be 'largest' or 'smallest'.")
        if tie_break == "largest":
            d = int(tied[np.argmax(max_points[tied])])
        else:
            d = int(tied[np.argmin(max_points[tied])])

    Pmin = float(np.min(max_points))
    Pmax = float(np.max(max_points))

    if np.isclose(Pmax, Pmin):
        # All equal -> standard drop: residual is 0
        r_d = 0.0
    else:
        # r(P) for the dropped HW only
        scaled = (max_points[d] - Pmin) / (Pmax - Pmin)
        r_d = float(rho * (scaled**alpha))

    P_prime_d = r_d * max_points[d]

    w_eff = max_points.copy()
    w_eff[d] = P_prime_d

    # Weighted average using effective weights
    denom = float(np.sum(w_eff))
    if np.isclose(denom, 0.0):
        # Would only happen if all weights become zero; safeguard.
        avg = 0.0
    else:
        avg = float(np.sum(w_eff * s) / denom)

    return HWScoreResult(
        avg=avg, dropped_index=d, effective_weights=w_eff, normalized_scores=s
    )


# ---- Example usage (your two scenarios) ----
if __name__ == "__main__":
    max_pts = [100, 100, 120, 140]
    # The first three students have 3/4 homework scores in common.
    students = {
        # Student A: HW1 missed
        "earned_A": [0, 98, 0.82 * 120, 0.78 * 140],
        # Student B: HW3 missed
        "earned_B": [95, 98, 0, 0.78 * 140],
        # Student C: Average across the board
        "earned_C": [95, 98, 0.82 * 120, 0.78 * 140],
        # Student D: Great across the board
        "earned_D": [97, 98, 0.93 * 120, 0.91 * 140],
        # Student E: Poor across the board
        "earned_E": [88, 89, 0.71 * 120, 0.73 * 140],
    }

    for label, earned in students.items():
        res_drop = homework_score(earned, max_pts, rho=0.75, alpha=1.0, drop=True)
        res_nodrop = homework_score(earned, max_pts, drop=False)

        print(f"Student {label}:")
        print(
            f"  drop=True : avg={res_drop.avg:.4f}, dropped_index={res_drop.dropped_index}, "
            f"effective_weights={res_drop.effective_weights}"
        )
        # print(f"  drop=False: avg={res_nodrop.avg:.4f}")
