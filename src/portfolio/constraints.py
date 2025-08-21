# src/portfolio/constraints.py
# Copyright (c) GMF
# SPDX-License-Identifier: MIT
"""
Constraint layer for portfolio optimization.

This module provides:
- Hard exposure constraints (long-only / box bounds / net & gross exposure).
- Leverage control (gross cap) with optional rescaling.
- Turnover constraints (max absolute change per asset and portfolio-level).
- Group/sector caps (e.g., TSLA sector <= 25%).
- Utility functions to produce optimizer-ready bounds and validate feasibility.

Design principles:
- Pure, side-effect free (except logging).
- Deterministic and unit-test friendly.
- No solver dependency; works with PyPortfolioOpt or custom solvers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------
# Dataclass configurations
# ---------------------------


@dataclass(frozen=True)
class BoxBounds:
    """Per-asset lower/upper bounds."""

    lower: Mapping[str, float]  # ticker -> lb
    upper: Mapping[str, float]  # ticker -> ub

    def as_list(self, order: Sequence[str]) -> List[Tuple[float, float]]:
        """Return bounds in optimizer order."""
        lbs = []
        ubs = []
        for t in order:
            lb = self.lower.get(t, 0.0)
            ub = self.upper.get(t, 1.0)
            if lb > ub:
                raise ValueError(f"Invalid bounds for {t}: lb {lb} > ub {ub}")
            lbs.append(lb)
            ubs.append(ub)
        return list(zip(lbs, ubs))


@dataclass(frozen=True)
class GroupCaps:
    """
    Group caps, e.g. sector limits or thematic buckets.

    Example:
    -------
        groups = {
            "Tech": ["AAPL", "MSFT", "TSLA"],
            "Bonds": ["BND"]
        }
        caps = {"Tech": 0.35, "Bonds": 0.70}

    """

    groups: Mapping[str, Sequence[str]]
    caps: Mapping[str, float]  # group -> max weight (0..1)

    def group_matrix(self, order: Sequence[str]) -> pd.DataFrame:
        """
        Return a (G x N) binary membership matrix for groups vs tickers.
        Rows=groups, columns=order.
        """
        mat = pd.DataFrame(0.0, index=list(self.groups.keys()), columns=list(order))
        for g, tickers in self.groups.items():
            for t in tickers:
                if t in mat.columns:
                    mat.loc[g, t] = 1.0
        return mat


@dataclass(frozen=True)
class ConstraintConfig:
    """
    Master configuration for constraints.

    Attributes
    ----------
        long_only: if True, enforce non-negative weights by default.
        box: per-asset lower/upper bounds.
        gross_limit: maximum sum of absolute weights (leverage), e.g. 1.0 for unlevered long-only.
        net_limit: |sum(weights)| <= net_limit (optional).
        rescale_to_gross: if True, after clipping/caps apply a positive homothetic rescale to meet gross_limit.
        turnover_asset_cap: max |w_i - w_i_prev| per asset (optional).
        turnover_portfolio_cap: max sum_i |Δw_i| (optional).
        group_caps: group membership + caps.
        cash_ticker: optional cash sleeve symbol; if provided, cash absorbs residual to keep sum(weights)=1 within bounds.

    """

    long_only: bool = True
    box: Optional[BoxBounds] = None
    gross_limit: float = 1.0
    net_limit: Optional[float] = None
    rescale_to_gross: bool = True
    turnover_asset_cap: Optional[float] = None
    turnover_portfolio_cap: Optional[float] = None
    group_caps: Optional[GroupCaps] = None
    cash_ticker: Optional[str] = None


# ---------------------------
# Core utilities
# ---------------------------


def _ensure_order(weights: Mapping[str, float], order: Sequence[str]) -> np.ndarray:
    return np.array([weights.get(t, 0.0) for t in order], dtype=float)


def _to_dict(arr: np.ndarray, order: Sequence[str]) -> Dict[str, float]:
    return {t: float(w) for t, w in zip(order, arr)}


def _project_to_box(w: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    """Clip weights to per-asset box bounds."""
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    return np.minimum(np.maximum(w, lb), ub)


def _renormalize_with_cash(
    w: np.ndarray, bounds: List[Tuple[float, float]], cash_idx: Optional[int]
) -> np.ndarray:
    """
    Adjust weights to sum to 1 by allocating residual to cash (if provided) within its bounds.
    If no cash, renormalize proportionally for positive weights (long-only) or by net shift otherwise.
    """
    total = w.sum()
    if abs(total - 1.0) < 1e-12:
        return w

    if cash_idx is not None:
        residual = 1.0 - total
        new_cash = w[cash_idx] + residual
        lb, ub = bounds[cash_idx]
        w[cash_idx] = np.clip(new_cash, lb, ub)
        # If we clipped, recompute residual and spread proportionally if required
        total2 = w.sum()
        if abs(total2 - 1.0) < 1e-10:
            return w

    # Fallback: proportional scaling of non-cash longs (safe for long-only)
    positives = w > 0
    pos_sum = w[positives].sum()
    if pos_sum > 0:
        w[positives] *= 1.0 / (w.sum())  # exact normalize
    else:
        # As a last resort, set uniform feasible distribution within box
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        # Try to hit the middle of each bound range then normalize
        mid = 0.5 * (lb + ub)
        if mid.sum() == 0:
            # all-zero bounds; return zeros
            return w * 0.0
        w = mid / mid.sum()
    return w


def _apply_gross_cap(w: np.ndarray, gross_limit: float) -> np.ndarray:
    gross = np.abs(w).sum()
    if gross <= gross_limit + 1e-12:
        return w
    # Homothetic scaling to meet gross limit exactly
    scale = gross_limit / gross
    return w * scale


def _apply_net_cap(w: np.ndarray, net_limit: Optional[float]) -> np.ndarray:
    if net_limit is None:
        return w
    net = w.sum()
    if abs(net) <= net_limit + 1e-12:
        return w
    # Shift weights towards zero net without changing gross magnitude too much.
    shift = np.sign(net) * (abs(net) - net_limit)
    # Distribute shift across all assets proportionally to weight sign and magnitude
    denom = np.sum(np.sign(net) * np.sign(w) * np.maximum(np.abs(w), 1e-12))
    if denom == 0:
        return w
    w_adjust = (
        (shift / denom) * np.sign(net) * np.sign(w) * np.maximum(np.abs(w), 1e-12)
    )
    return w - w_adjust


def _apply_group_caps(
    w: np.ndarray, order: Sequence[str], group_caps: GroupCaps
) -> np.ndarray:
    """
    Simple projection heuristic: if a group's weight exceeds cap, scale weights
    inside the group down proportionally while preserving non-group weights.
    """
    mat = group_caps.group_matrix(order)  # G x N
    w_series = pd.Series(w, index=order)
    for g in mat.index:
        mask = mat.loc[g] > 0
        group_weight = float(w_series[mask].sum())
        cap = float(group_caps.caps.get(g, 1.0))
        if group_weight > cap + 1e-12 and group_weight > 0:
            scale = cap / group_weight
            w_series.loc[mask] *= scale
            logger.debug(
                "Group cap applied", extra={"group": g, "cap": cap, "scale": scale}
            )
    return w_series.values


def _apply_turnover_caps(
    w_prev: np.ndarray,
    w_target: np.ndarray,
    per_asset_cap: Optional[float],
    portfolio_cap: Optional[float],
) -> np.ndarray:
    """
    Apply turnover limits by clipping per-asset change and,
    if necessary, proportionally scaling total turnover to meet portfolio_cap.
    """
    delta = w_target - w_prev

    # Per-asset cap
    if per_asset_cap is not None:
        cap = float(per_asset_cap)
        delta = np.clip(delta, -cap, cap)

    # Portfolio-level cap
    if portfolio_cap is not None:
        tcost = np.sum(np.abs(delta))
        cap_tot = float(portfolio_cap)
        if tcost > cap_tot + 1e-12:
            scale = cap_tot / tcost
            delta *= scale

    return w_prev + delta


# ---------------------------
# Public API
# ---------------------------


class ConstraintSet:
    """
    Apply and validate portfolio constraints end-to-end.

    Typical flow:
        cs = ConstraintSet(config, universe=["TSLA","SPY","BND"])
        w_new, report = cs.enforce(target_weights, prev_weights)

    The result is guaranteed to be inside all box/group/gross/net/turnover bounds
    (up to numerical tolerance).
    """

    def __init__(self, config: ConstraintConfig, universe: Sequence[str]):
        self.cfg = config
        self.universe = list(universe)
        self._bounds = self._build_bounds()

        # cache cash index if provided
        self._cash_idx = (
            self.universe.index(self.cfg.cash_ticker)
            if self.cfg.cash_ticker in self.universe
            else None
        )

    # ---------- Build-time ----------

    def _build_bounds(self) -> List[Tuple[float, float]]:
        if self.cfg.box is not None:
            b = self.cfg.box.as_list(self.universe)
        elif self.cfg.long_only:
            b = [(0.0, 1.0) for _ in self.universe]
        else:
            # relaxed symmetric bounds by default for long/short
            b = [(-1.0, 1.0) for _ in self.universe]
        return b

    def optimizer_bounds(self) -> List[Tuple[float, float]]:
        """Bounds for optimizers (e.g., PyPortfolioOpt / cvxpy wrappers)."""
        return list(self._build_bounds())

    # ---------- Validation ----------

    def validate(self, w: Mapping[str, float]) -> None:
        """Raise ValueError if weights violate any constraints."""
        arr = _ensure_order(w, self.universe)

        # Box
        lb = np.array([b[0] for b in self._bounds])
        ub = np.array([b[1] for b in self._bounds])
        if np.any(arr < lb - 1e-10) or np.any(arr > ub + 1e-10):
            raise ValueError("Weights violate per-asset box bounds.")

        # Group caps
        if self.cfg.group_caps is not None:
            mat = self.cfg.group_caps.group_matrix(self.universe)
            gw = mat.values @ arr
            for i, g in enumerate(mat.index):
                cap = self.cfg.group_caps.caps.get(g, 1.0)
                if gw[i] > cap + 1e-10:
                    raise ValueError(f"Group {g} exceeds cap {cap}: {gw[i]:.4f}")

        # Gross & net
        gross = float(np.sum(np.abs(arr)))
        if gross > self.cfg.gross_limit + 1e-10:
            raise ValueError(
                f"Gross {gross:.4f} exceeds limit {self.cfg.gross_limit:.4f}"
            )
        if self.cfg.net_limit is not None:
            net = float(arr.sum())
            if abs(net) > self.cfg.net_limit + 1e-10:
                raise ValueError(
                    f"Net {net:.4f} exceeds limit {self.cfg.net_limit:.4f}"
                )

        # Sum to one (soft check if cash present)
        if abs(arr.sum() - 1.0) > 1e-8 and self.cfg.cash_ticker is None:
            raise ValueError(f"Weights must sum to 1. Current sum={arr.sum():.8f}.")

    # ---------- Enforcement ----------

    def enforce(
        self,
        target: Mapping[str, float],
        prev: Optional[Mapping[str, float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Project target weights onto the feasible set and apply turnover controls.

        Args:
        ----
            target: desired weights by ticker (can be outside bounds).
            prev: previous weights; if None, turnover caps are ignored for the first allocation.

        Returns:
        -------
            (feasible_weights_dict, report_dict)

        """
        w = _ensure_order(target, self.universe)

        # 1) Box bounds
        w = _project_to_box(w, self._bounds)

        # 2) Group caps
        if self.cfg.group_caps is not None:
            w = _apply_group_caps(w, self.universe, self.cfg.group_caps)

        # 3) Net exposure cap (before normalization)
        w = _apply_net_cap(w, self.cfg.net_limit)

        # 4) Sum to 1 with cash sleeve or proportional normalization
        w = _renormalize_with_cash(w, self._bounds, self._cash_idx)

        # 5) Gross leverage cap (homothetic scaling)
        if self.cfg.rescale_to_gross and self.cfg.gross_limit is not None:
            w = _apply_gross_cap(w, self.cfg.gross_limit)

        # 6) Turnover caps
        if prev is not None and (
            self.cfg.turnover_asset_cap is not None
            or self.cfg.turnover_portfolio_cap is not None
        ):
            w_prev = _ensure_order(prev, self.universe)
            w = _apply_turnover_caps(
                w_prev=w_prev,
                w_target=w,
                per_asset_cap=self.cfg.turnover_asset_cap,
                portfolio_cap=self.cfg.turnover_portfolio_cap,
            )

        report = {
            "gross": float(np.sum(np.abs(w))),
            "net": float(np.sum(w)),
            "sum": float(np.sum(w)),
        }
        return _to_dict(w, self.universe), report
