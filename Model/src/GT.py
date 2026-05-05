"""Ground-truth route and class counts used for evaluation and comparison."""

from __future__ import annotations

from copy import deepcopy
import pandas as pd


FLOW_ORDER = ["S->N", "N->S", "E->S", "E->N", "S->E", "N->E"]
CLASS_ORDER = ["car", "light truck", "heavy truck", "semi-trailer / combination vehicle", "bus", "motorcycle", "bicycle"]

GT_BY_VIDEO = {

    "Fv587_Haukeland_day_1h_rush.mp4": {
        "S->N": {"car": 523, "light truck": 16, "heavy truck": 14, "semi-trailer / combination vehicle": 15, "bus": 3, "motorcycle": 17, "bicycle": 1},
        "N->S": {"car": 354, "light truck": 11, "heavy truck": 11, "semi-trailer / combination vehicle": 14, "bus": 2, "motorcycle": 13, "bicycle": 2},
        "E->S": {"car": 18,  "light truck": 0,  "heavy truck": 0, "semi-trailer / combination vehicle": 0,   "bus": 0, "motorcycle": 0,  "bicycle": 0},
        "E->N": {"car": 16,  "light truck": 0,  "heavy truck": 0, "semi-trailer / combination vehicle": 0,   "bus": 0, "motorcycle": 0,  "bicycle": 1},
        "S->E": {"car": 18,  "light truck": 0,  "heavy truck": 0, "semi-trailer / combination vehicle": 0,   "bus": 0, "motorcycle": 1,  "bicycle": 4},
        "N->E": {"car": 10,  "light truck": 1,  "heavy truck": 0, "semi-trailer / combination vehicle": 0,   "bus": 2, "motorcycle": 0,  "bicycle": 0},
        "Gangfeltet": {"person": 7},
    },

    "Fv587_Haukeland_15min.mp4": {
        "S->N": {"car": 134, "light truck": 5, "heavy truck": 2, "semi-trailer / combination vehicle": 1, "bus": 0, "motorcycle": 1, "bicycle": 0},
        "N->S": {"car": 71, "light truck": 4, "heavy truck": 4, "semi-trailer / combination vehicle": 1, "bus": 1, "motorcycle": 3, "bicycle": 0},
        "E->S": {"car": 1, "light truck": 0, "heavy truck": 0, "semi-trailer / combination vehicle": 0, "bus": 0, "motorcycle": 0, "bicycle": 0},
        "E->N": {"car": 1, "light truck": 0, "heavy truck": 0, "semi-trailer / combination vehicle": 0, "bus": 0, "motorcycle": 0, "bicycle": 0},
        "S->E": {"car": 4, "light truck": 0, "heavy truck": 0, "semi-trailer / combination vehicle": 0, "bus": 0, "motorcycle": 1, "bicycle": 0},
        "N->E": {"car": 4, "light truck": 0, "heavy truck": 0, "semi-trailer / combination vehicle": 0, "bus": 0, "motorcycle": 0, "bicycle": 0},
        "Gangfeltet": {"person": 3},
    },
    
    "Fv587_Haukeland_night_1h.mp4": {
        "S->N": {"car": 60, "light truck": 0, "heavy truck": 1, "semi-trailer / combination vehicle": 6, "bus": 1, "motorcycle": 0, "bicycle": 0},
        "N->S": {"car": 54, "light truck": 1, "heavy truck": 1, "semi-trailer / combination vehicle": 1, "bus": 1, "motorcycle": 2, "bicycle": 0},
        "E->S": {"car": 2, "light truck": 0, "heavy truck": 0, "semi-trailer / combination vehicle": 0, "bus": 0, "motorcycle": 0, "bicycle": 0},
        "E->N": {"car": 2, "light truck": 0, "heavy truck": 0, "semi-trailer / combination vehicle": 0, "bus": 0, "motorcycle": 0, "bicycle": 0},
        "S->E": {"car": 2, "light truck": 0, "heavy truck": 0, "semi-trailer / combination vehicle": 0, "bus": 0, "motorcycle": 0, "bicycle": 0},
        "N->E": {"car": 0, "light truck": 0, "heavy truck": 0, "semi-trailer / combination vehicle": 0, "bus": 0, "motorcycle": 0, "bicycle": 0},
        },

    "Fv587_Haukeland_10min_transition.mp4": {
        "S->N": {"car": 16, "light truck": 0, "heavy truck": 0, "semi-trailer / combination vehicle": 0, "bus": 0, "motorcycle": 0, "bicycle": 0},
        "N->S": {"car": 10, "light truck": 0, "heavy truck": 0, "semi-trailer / combination vehicle": 0, "bus": 1, "motorcycle": 0, "bicycle": 1},
        "E->S": {"car": 0, "light truck": 0, "heavy truck": 0, "semi-trailer / combination vehicle": 0, "bus": 0, "motorcycle": 0, "bicycle": 0},
        "E->N": {"car": 1, "light truck": 0, "heavy truck": 0, "semi-trailer / combination vehicle": 0, "bus": 0, "motorcycle": 0, "bicycle": 0},
        "S->E": {"car": 1, "light truck": 0, "heavy truck": 0, "semi-trailer / combination vehicle": 0, "bus": 0, "motorcycle": 0, "bicycle": 0},
        "N->E": {"car": 0, "light truck": 0, "heavy truck": 0, "semi-trailer / combination vehicle": 0, "bus": 0, "motorcycle": 0, "bicycle": 0},
        },
}

def get_gt(video_name: str) -> dict:
    try:
        return deepcopy(GT_BY_VIDEO[video_name])
    except KeyError as exc:
        available = ", ".join(sorted(GT_BY_VIDEO))
        raise KeyError(f"Unknown video '{video_name}'. Available ground truths: {available}") from exc


def load_gt_vehicles(
    video_name: str,
    flow_order: list[str] = None,
    class_order: list[str] = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Load ground truth vehicle data for a video and return both dict and DataFrame.
    
    Args:
        video_name: Name of the video
        flow_order: Order of flow directions (default: FLOW_ORDER)
        class_order: Order of vehicle classes (default: CLASS_ORDER)
    
    Returns:
        Tuple of (gt_dict, gt_dataframe) where:
        - gt_dict: Raw ground truth dict with only vehicle routes (not Gangfeltet)
        - gt_dataframe: pandas DataFrame indexed by flow_order with class columns and Sum
    """
    if flow_order is None:
        flow_order = FLOW_ORDER
    if class_order is None:
        class_order = CLASS_ORDER
    
    gt_dict = get_gt(video_name)
    
    # Filter to only vehicle routes (exclude Gangfeltet)
    gt_vehicles = {k: v for k, v in gt_dict.items() if k in flow_order}
    
    # Create DataFrame
    gt_df = (
        pd.DataFrame.from_dict(gt_vehicles, orient="index")
        .reindex(flow_order)
        .fillna(0)
        .astype(int)
    )
    gt_df = gt_df[class_order]
    gt_df["Sum"] = gt_df[class_order].sum(axis=1)
    
    return gt_vehicles, gt_df


def load_gt_persons(
    video_name: str,
    zone_name: str = "Gangfeltet",
) -> tuple[dict, int]:
    """
    Load ground truth person data for a specific zone in a video.
    
    Args:
        video_name: Name of the video
        zone_name: Name of the zone (default: "Gangfeltet")
    
    Returns:
        Tuple of (zone_dict, person_count) where:
        - zone_dict: Zone data dict (e.g., {"person": 3})
        - person_count: Number of persons in the zone
    """
    gt_dict = get_gt(video_name)
    zone_dict = gt_dict.get(zone_name, {"person": 0})
    person_count = zone_dict.get("person", 0)
    return zone_dict, person_count


__all__ = [
    "FLOW_ORDER",
    "CLASS_ORDER",
    "GT_BY_VIDEO",
    "get_gt",
    "load_gt_vehicles",
    "load_gt_persons",
]

