from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


EARTH_RADIUS_KM: float = 6371.0


def haversine_distances_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    radius_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """Vectorized Haversine distance.

    Computes distance (km) from a single point (lat1, lon1) to *many* points
    given by arrays `lats`, `lons`.

    Notes
    -----
    - Inputs are converted to radians before processing.
    - Uses numpy vectorization to support NVIDIA-accelerated architectures.
    """
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)

    dlat = lats_rad - lat1_rad
    dlon = lons_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lats_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return radius_km * c


def select_next_stop_index(
    *,
    source_lat: float,
    source_lon: float,
    candidate_lats: Sequence[float],
    candidate_lons: Sequence[float],
    demands: Sequence[float],
    priorities: Sequence[float],
    radius_km: float = EARTH_RADIUS_KM,
) -> int:
    """Select argmin index using route cost model.

    Implements:
        score_j = d_j + 0.5 * demand_j - 1.2 * priority_j

    Returns
    -------
    int
        Index j of the best (minimum) score among all candidates.
    """
    candidate_lats_np = np.asarray(candidate_lats, dtype=np.float64)
    candidate_lons_np = np.asarray(candidate_lons, dtype=np.float64)
    demands_np = np.asarray(demands, dtype=np.float64)
    priorities_np = np.asarray(priorities, dtype=np.float64)

    if not (
        candidate_lats_np.shape == candidate_lons_np.shape == demands_np.shape == priorities_np.shape
    ):
        raise ValueError(
            "candidate_lats, candidate_lons, demands, and priorities must have the same shape"
        )

    d_km = haversine_distances_km(
        lat1=np.asarray(source_lat, dtype=np.float64),
        lon1=np.asarray(source_lon, dtype=np.float64),
        lats=candidate_lats_np,
        lons=candidate_lons_np,
        radius_km=radius_km,
    )

    score = d_km + 0.5 * demands_np - 1.2 * priorities_np
    return int(np.argmin(score))


@dataclass
class RouteCostResult:
    index: int
    distances_km: np.ndarray
    scores: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "distances_km": self.distances_km.tolist(),
            "scores": self.scores.tolist(),
        }


def compute_route_cost_model(
    *,
    source_lat: float,
    source_lon: float,
    candidate_lats: Sequence[float],
    candidate_lons: Sequence[float],
    demands: Sequence[float],
    priorities: Sequence[float],
    radius_km: float = EARTH_RADIUS_KM,
) -> RouteCostResult:
    """Compute distances + score_j, and pick argmin index."""
    candidate_lats_np = np.asarray(candidate_lats, dtype=np.float64)
    candidate_lons_np = np.asarray(candidate_lons, dtype=np.float64)
    demands_np = np.asarray(demands, dtype=np.float64)
    priorities_np = np.asarray(priorities, dtype=np.float64)

    d_km = haversine_distances_km(
        lat1=np.asarray(source_lat, dtype=np.float64),
        lon1=np.asarray(source_lon, dtype=np.float64),
        lats=candidate_lats_np,
        lons=candidate_lons_np,
        radius_km=radius_km,
    )

    scores = d_km + 0.5 * demands_np - 1.2 * priorities_np
    best_idx = int(np.argmin(scores))

    return RouteCostResult(index=best_idx, distances_km=d_km, scores=scores)

