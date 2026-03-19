### Used for calculating final hw grades.abs

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

RHO = 0.0


@dataclass(frozen=True)
class HWScoreResult:
    avg: float  # homework average in [0, 1]
    dropped_index: Optional[int]  # which HW was "dropped" (None if drop=False)
    effective_weights: np.ndarray  # P_i for non-dropped; P'_d for dropped
    normalized_scores: np.ndarray  # s_i = earned/max


def score_homework(
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


def curve_scores(
    scores: Iterable[float], strategy: str = "linear", target_mean: float = None
) -> List[float]:
    """
    Applies a curving strategy to an array of scores.

    Parameters:
      scores: The raw scores on a scale of [0.0, 1.0]
      strategy:
        "linear" adds a constant to all scores shifting the max to 1.0.
        "target" scores are scaled proportionally to achieve a max target mean.

    Notes:
      - If using "target" strategy, you must specify a "target_mean"
    Retuns:
      A list of curved scores on a scale of [0.0, 1.0]
    """
    scores = np.asarray(list(scores), dtype=float)

    if strategy not in ["linear", "target"]:
        raise ValueError("strategy must be either of 'linear' or 'target'")
    if np.any(scores > 1.0) or np.any(scores < 0.0):
        raise ValueError("scores should be between 0.0 and 1.0")

    if strategy == "linear":
        max_score = scores.max()
        gap = 1.0 - max_score
        if gap > 0.0:
            curved = scores + gap
            return curved.tolist()

    if strategy == "target":
        if target_mean is None:
            raise KeyError("strategy=target requires a target_mean between 0 and 1")
        if not (0.0 < target_mean < 1.0):
            raise ValueError("target_mean must be a value between 0.0 and 1.0")
        avg = scores.mean()
        if target_mean <= avg:
            raise ValueError(
                f"The target_mean = {target_mean} provided is less than or equal to scores average = {avg:.3f}"
            )
        curved = scores + target_mean - avg
        if curved.max() > 1.0:
            raise ValueError(
                f"The target_mean = {target_mean} raises the max score to more than 1.0. Pick a lower target_mean."
            )
        return curved.tolist()


def get_final_score(
    hw_maxes: Iterable, hw_raw: Iterable, exam: float, attend: float, paper: float
) -> float:
    """Calculates the weighted final score for a student."""
    hw_avg = score_homework(hw_raw, hw_maxes, rho=RHO, alpha=1.0, drop=True)
    if not 0.0 <= exam <= 1.0:
        raise ValueError("exam value must be in [0.0, 1.0]")
    if not 0.0 <= attend <= 1.0:
        raise ValueError("part value must be in [0.0, 1.0]")
    if not 0.0 <= paper <= 1.0:
        raise ValueError("paper value must be in [0.0, 1.0]")
    final_score = 0.43 * hw_avg.avg + 0.2 * exam + 0.07 * attend + 0.3 * paper
    return final_score


# ---- Example usage (your two scenarios) ----
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--simulate_drop", action="store_true")
    parser.add_argument("--simulate_curving", action="store_true")
    args = parser.parse_args()

    exam_raw = {
        "Vivian": 0.57,
        "Eliz": 0.66,
        "Angela": 0.5767,
        "Mohammad": 0.5217,
        "Kenneth": 0.4717,
        "Kezia": 0.4117,
        "Matthew": 0.5083,
        "Shengqi": 0.70,
    }
    hw_max_pts = [100, 100, 120, 140]
    hw_raw = {
        "Vivian": [95.5, 98.5, 48, 132],
        "Eliz": [0, 97, 103, 99],
        "Angela": [94, 96, 98.5, 137],
        "Mohammad": [96.5, 98, 94, 135],
        "Kenneth": [98, 97, 113, 136],
        "Kezia": [91.5, 94.5, 108, 82],
        "Matthew": [97, 97, 111, 108],
        "Shengqi": [94.5, 93.5, 109.5, 136],
    }
    if args.simulate_drop:
        # The first three students have 3/4 homework scores in common.
        hw_raw = dict(sorted(hw_raw.items(), key=lambda item: item[1], reverse=True))
        for student, raw in hw_raw.items():
            hw_score_drop = score_homework(
                raw, hw_max_pts, rho=RHO, alpha=1.0, drop=True
            )
            hw_score_no_drop = score_homework(raw, hw_max_pts, drop=False)

            print(f"Student {student}:")
            print(
                f"  drop=True : avg={hw_score_drop.avg:.4f}, dropped_index={hw_score_drop.dropped_index}, "
                f"effective_weights={hw_score_drop.effective_weights}"
            )
            print(f"  drop=False: avg={hw_score_no_drop.avg:.4f}")

    elif args.simulate_curving:
        exam_raw = dict(
            sorted(exam_raw.items(), key=lambda item: item[1], reverse=True)
        )
        scores = list(exam_raw.values())
        mean_score = np.average(scores)
        print(f"Mean score before curving: {mean_score:.3f}")
        curved_scores = curve_scores(scores=scores, strategy="linear", target_mean=0.85)
        for i, (student, raw) in enumerate(exam_raw.items()):
            print(
                f"{student:<8} | raw score = {raw:.3f} | curved score = {curved_scores[i]:.4f}"
            )

    else:
        exam = {
            "Vivian": 0.90,
            "Eliz": 0.96,
            "Angela": 0.90,
            "Mohammad": 0.85,
            "Kenneth": 0.86,
            "Kezia": 0.79,
            "Matthew": 0.84,
            "Shengqi": 1.0,
        }
        attend = {
            "Vivian": 1.0,
            "Eliz": 0.9,
            "Angela": 1.0,
            "Mohammad": 0.9,
            "Kenneth": 1.0,
            "Kezia": 1.0,
            "Matthew": 1.0,
            "Shengqi": 0.9,
        }
        paper = {
            "Vivian": 0.975,
            "Eliz": 0.64,
            "Angela": 0.975,
            "Mohammad": 0.945,
            "Kenneth": 0.945,
            "Kezia": 0.74,
            "Matthew": 0.87,
            "Shengqi": 0.91,
        }
        for student in hw_raw.keys():
            final_score = get_final_score(
                hw_maxes=hw_max_pts,
                hw_raw=hw_raw.get(student),
                exam=exam.get(student),
                attend=attend.get(student),
                paper=paper.get(student),
            )
            print(f"{student:<8} | {final_score:.3f}")
