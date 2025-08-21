# src/portfolio/rebalancing.py
# Copyright (c) GMF
# SPDX-License-Identifier: MIT
"""
Rebalancing rules and schedulers:
- Time-based rebalancing (M/W/Q) with trading-day alignment.
- VIX-triggered rebalancing with hysteresis and cooldown windows.
- Composite policies (OR / AND).
- Helpers to generate rebalance dates over a price index.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------
# Time-based policy
# ---------------------------


class Frequency(str, Enum):
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"


@dataclass(frozen=True)
class TimeRebalancePolicy:
    """
    Time-based rebalancing.

    Attributes
    ----------
        freq: 'W' (weekly), 'M' (monthly), 'Q' (quarterly)
        nth: pick the nth trading day within the period (1-based). Example: nth=1 → first trading day.
        align: 'start' or 'end' within the period for resample anchor.

    """

    freq: Frequency = Frequency.MONTHLY
    nth: int = 1
    align: str = "end"  # 'start' or 'end'

    def dates(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Return trading dates that satisfy the time policy.
        """
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("index must be a pd.DatetimeIndex")

        # Resample to period endpoints
        if self.align == "start":
            per_ends = index.to_series().resample(self.freq.value).first().dropna()
        else:
            per_ends = index.to_series().resample(self.freq.value).last().dropna()

        # For each period, choose nth trading day relative to start/end
        selected = []
        for start, end in zip(
            per_ends.index, per_ends.index[1:].tolist() + [per_ends.index[-1]]
        ):
            # Get all trading days within period
            mask = (index >= start) & (index <= end)
            days = index[mask]
            if len(days) == 0:
                continue
            if self.align == "start":
                k = min(self.nth - 1, len(days) - 1)
                selected.append(days[k])
            else:
                k = min(self.nth - 1, len(days) - 1)
                selected.append(days[-(k + 1)])

        return pd.DatetimeIndex(sorted(set(selected)))


# ---------------------------
# VIX-triggered policy
# ---------------------------


@dataclass(frozen=True)
class VIXTriggerPolicy:
    """
    Rebalance when VIX crosses thresholds with hysteresis and cooldown.

    Attributes
    ----------
        upper: enter "high-vol" when VIX >= upper.
        lower: exit "high-vol" when VIX <= lower (must be < upper for hysteresis).
        cooldown_days: minimum trading days between consecutive triggers.
        min_days_high: minimum days to stay in high-vol regime before exiting.

    """

    upper: float = 30.0
    lower: float = 22.0
    cooldown_days: int = 5
    min_days_high: int = 3

    def trigger_dates(self, vix: pd.Series) -> pd.DatetimeIndex:
        """
        Return trading dates where a trigger occurs (enter or exit high-vol regime).
        """
        if not isinstance(vix.index, pd.DatetimeIndex):
            raise TypeError("vix series must have DatetimeIndex")

        if self.lower >= self.upper:
            raise ValueError("lower must be < upper for hysteresis")

        vix = vix.sort_index().dropna()
        state_high = False
        last_trigger: Optional[pd.Timestamp] = None
        high_since: Optional[pd.Timestamp] = None
        triggers: List[pd.Timestamp] = []

        for dt, x in vix.items():
            # Enforce cooldown
            if (
                last_trigger is not None
                and (dt - last_trigger).days < self.cooldown_days
            ):
                continue

            if not state_high:
                # Enter high-vol regime
                if x >= self.upper:
                    state_high = True
                    high_since = dt
                    triggers.append(dt)
                    last_trigger = dt
            else:
                # Potential exit after min_days_high and lower threshold
                if high_since is not None:
                    if (dt - high_since).days >= self.min_days_high and x <= self.lower:
                        state_high = False
                        high_since = None
                        triggers.append(dt)
                        last_trigger = dt

        return pd.DatetimeIndex(triggers)


# ---------------------------
# Composite policy
# ---------------------------


class LogicalMode(str, Enum):
    OR = "OR"
    AND = "AND"


@dataclass(frozen=True)
class CompositePolicy:
    """
    Combine time-based and VIX-triggered policies.

    mode=OR: rebalance if any policy fires on a date.
    mode=AND: rebalance only if both conditions are met (same date).
    """

    time_policy: Optional[TimeRebalancePolicy] = None
    vix_policy: Optional[VIXTriggerPolicy] = None
    mode: LogicalMode = LogicalMode.OR

    def dates(
        self, index: pd.DatetimeIndex, vix: Optional[pd.Series] = None
    ) -> pd.DatetimeIndex:
        dates_sets: List[pd.DatetimeIndex] = []

        if self.time_policy is not None:
            dates_sets.append(self.time_policy.dates(index))

        if self.vix_policy is not None:
            if vix is None:
                raise ValueError("vix series required when vix_policy is provided")
            dates_sets.append(self.vix_policy.trigger_dates(vix.reindex(index).ffill()))

        if not dates_sets:
            return pd.DatetimeIndex([])

        if self.mode == LogicalMode.OR:
            all_dates = sorted(set().union(*[set(d) for d in dates_sets]))
            return pd.DatetimeIndex(all_dates)
        else:
            # intersection
            common = set(dates_sets[0])
            for d in dates_sets[1:]:
                common = common.intersection(set(d))
            return pd.DatetimeIndex(sorted(common))


# ---------------------------
# Schedulers / Helpers
# ---------------------------


def generate_rebalance_dates(
    price_index: pd.DatetimeIndex,
    policy: CompositePolicy | TimeRebalancePolicy | VIXTriggerPolicy,
    vix: Optional[pd.Series] = None,
) -> pd.DatetimeIndex:
    """
    Convenience wrapper to compute rebalance dates given any policy.
    """
    if isinstance(policy, CompositePolicy):
        return policy.dates(price_index, vix=vix)
    elif isinstance(policy, TimeRebalancePolicy):
        return policy.dates(price_index)
    elif isinstance(policy, VIXTriggerPolicy):
        if vix is None:
            raise ValueError("vix series required for VIXTriggerPolicy")
        return policy.trigger_dates(vix.reindex(price_index).ffill())
    else:
        raise TypeError("Unknown policy type")


def next_rebalance_date(
    current_date: pd.Timestamp,
    price_index: pd.DatetimeIndex,
    dates: pd.DatetimeIndex,
) -> Optional[pd.Timestamp]:
    """
    Given precomputed rebalance dates, return the next date >= current_date.
    """
    after = dates[dates >= current_date]
    return None if len(after) == 0 else after[0]
