"""
split_id_fallback.py
Post-processing for fragmented track ID recovery.

When a vehicle track is fragmented into multiple identities, the fragments
can be recovered by matching temporal continuity, spatial proximity, velocity
prediction, and class consistency. This is useful for tracker switches, short
occlusions, and other brief tracking interruptions.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

from .geometry import tracking_point_xyxy, point_in_polygon


class SplitIDRecovery:
    """
    Recover fragmented track identities across brief tracking interruptions.
    Matches fragments based on spatial distance, velocity, class, and time gap.
    """
    
    def __init__(
        self,
        model,
        gate_zones: Dict,
        fps: float = 30.0,
        enable: bool = True,
        max_dist_raw: float = 180.0,
        max_dist_pred: float = 160.0,
        class_penalty: float = 40.0,
        max_gap_seconds: float = 3.0,
    ):
        """
        Initialize fragmented track recovery.
        
        Args:
            model: YOLO model for class names
            gate_zones: Dict of counting zones
            fps: Video frame rate (for temporal scaling)
            enable: Enable fallback processing
            max_dist_raw: Max raw euclidean distance
            max_dist_pred: Max distance with velocity prediction
            class_penalty: Penalty for class mismatch
            max_gap_seconds: Max temporal gap between fragments
        """
        self.model = model
        self.gate_zones = gate_zones
        self.fps = fps
        self.enable = enable
        self.max_dist_raw = max_dist_raw
        self.max_dist_pred = max_dist_pred
        self.class_penalty = class_penalty
        self.max_gap = int(fps * max_gap_seconds)
    
    def _euclid(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Euclidean distance."""
        return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))
    
    def _zone_of_point(self, pt: Tuple[float, float]) -> str:
        """Get zone name for a point."""
        for zname, zinfo in self.gate_zones.items():
            poly = zinfo.get("polygon", [])
            if len(poly) < 3:
                continue
            if point_in_polygon((int(round(pt[0])), int(round(pt[1]))), poly):
                return zinfo.get("dir", zname)
        return None
    
    def _main_class(self, clsid_list: List[int]) -> int:
        """Get most common class from list."""
        if not clsid_list:
            return -1
        vals, counts = np.unique(np.array(clsid_list, dtype=int), return_counts=True)
        return int(vals[np.argmax(counts)])
    
    def _build_fragments(self, all_tracks_by_frame: Dict) -> List[Dict]:
        """
        Extract track fragments from detections.
        
        Returns list of fragments with structure:
        {
            "id": track_id,
            "start_f": start_frame,
            "end_f": end_frame,
            "start_p": (x, y),
            "end_p": (x, y),
            "start_zone": zone_name,
            "end_zone": zone_name,
            "main_clsid": class_id,
            "vel": (vx, vy),
        }
        """
        by_tid = defaultdict(lambda: {"frames": [], "points": [], "clsids": []})
        
        for fidx, dets in all_tracks_by_frame.items():
            for d in dets:
                if len(d) < 4:
                    continue
                tid, box_full_frame, _, clsid = d[0], d[1], d[2], d[3]
                tid = int(tid)
                p = tracking_point_xyxy(box_full_frame)
                
                by_tid[tid]["frames"].append(int(fidx))
                by_tid[tid]["points"].append((float(p[0]), float(p[1])))
                by_tid[tid]["clsids"].append(int(clsid) if clsid is not None else -1)
        
        frags = []
        for tid, d in by_tid.items():
            if len(d["frames"]) < 2:
                continue
            
            order = np.argsort(d["frames"])
            frames = [d["frames"][i] for i in order]
            points = [d["points"][i] for i in order]
            clsids = [d["clsids"][i] for i in order]
            
            dt = max(1, frames[-1] - frames[0])
            vx = (points[-1][0] - points[0][0]) / dt
            vy = (points[-1][1] - points[0][1]) / dt
            
            frags.append({
                "id": tid,
                "start_f": frames[0],
                "end_f": frames[-1],
                "start_p": points[0],
                "end_p": points[-1],
                "start_zone": self._zone_of_point(points[0]),
                "end_zone": self._zone_of_point(points[-1]),
                "main_clsid": self._main_class(clsids),
                "vel": (vx, vy),
            })
        
        return frags
    
    def apply(
        self,
        all_tracks_by_frame: Dict,
        od_events: List[Dict],
        od_counts: Dict,
    ) -> Dict[str, object]:
        """
        Apply fragmented-track recovery to O--D events and counts.
        
        Args:
            all_tracks_by_frame: {frame_idx: [(tid, box, conf, clsid, ...)]}
            od_events: Existing O-D events.
            od_counts: Existing O-D counts (will be modified).
        
        Returns:
            Dictionary with aggregate counts and recovered fragment matches.
        """
        if not self.enable or not self.gate_zones or not all_tracks_by_frame:
            return {
                "added_same_id": 0,
                "added_split_id": 0,
                "same_id_track_ids": [],
                "split_id_matches": [],
            }
        
        frags = self._build_fragments(all_tracks_by_frame)
        if not frags:
            return {
                "added_same_id": 0,
                "added_split_id": 0,
                "same_id_track_ids": [],
                "split_id_matches": [],
            }
        
        counted_ids = set(int(e.get("track_id")) for e in od_events if "track_id" in e)
        
        # A) Same-ID missing trip (track never left origin zone before switch)
        added_same = 0
        same_id_track_ids = []
        for f in frags:
            tid = int(f["id"])
            if tid in counted_ids:
                continue
            if f["start_zone"] is None or f["end_zone"] is None:
                continue
            if f["start_zone"] == f["end_zone"]:
                continue
            
            cls_name = self.model.names.get(int(f["main_clsid"]), str(int(f["main_clsid"])))
            od_counts[f["end_zone"]][f["start_zone"]][cls_name] += 1
            od_events.append({
                "frame": int(f["end_f"]),
                "track_id": tid,
                "origin": f["start_zone"],
                "dest": f["end_zone"],
                "class": cls_name,
                "conf": None,
                "count_method": "fallback_same_id",
            })
            counted_ids.add(tid)
            added_same += 1
            same_id_track_ids.append(tid)
        
        # B) Split-ID matching (fragments before and after switch)
        min_fragment_frames = 5
        starts = [
            f for f in frags 
            if int(f["id"]) not in counted_ids 
            and f["start_zone"] is not None
            and (int(f["end_f"]) - int(f["start_f"]) + 1) >= min_fragment_frames
        ]
        ends = [
            f for f in frags 
            if int(f["id"]) not in counted_ids 
            and f["end_zone"] is not None
            and (int(f["end_f"]) - int(f["start_f"]) + 1) >= min_fragment_frames
        ]
        
        starts.sort(key=lambda x: x["end_f"])
        ends.sort(key=lambda x: x["start_f"])
        
        used_end_ids = set()
        added_split = 0
        split_id_matches = []
        
        for a in starts:
            best = None
            best_score = float("inf")
            
            for b in ends:
                if int(b["id"]) == int(a["id"]):
                    continue
                if int(b["id"]) in used_end_ids:
                    continue
                
                # REQUIREMENT: Classes must match
                if int(a["main_clsid"]) != int(b["main_clsid"]):
                    continue
                
                gap = int(b["start_f"]) - int(a["end_f"])
                if gap < 1 or gap > self.max_gap:
                    continue
                
                if a["start_zone"] is None or b["end_zone"] is None:
                    continue
                if a["start_zone"] == b["end_zone"]:
                    continue
                
                # Velocity prediction
                pred = (
                    a["end_p"][0] + a["vel"][0] * gap,
                    a["end_p"][1] + a["vel"][1] * gap,
                )
                
                d_raw = self._euclid(a["end_p"], b["start_p"])
                d_pred = self._euclid(pred, b["start_p"])
                
                if d_raw > self.max_dist_raw and d_pred > self.max_dist_pred:
                    continue
                
                # Scoring: weighted combination
                score = 0.35 * d_raw + 0.65 * d_pred
                
                if score < best_score:
                    best_score = score
                    best = b
            
            if best is None:
                continue
            
            cls_name = self.model.names.get(int(best["main_clsid"]), str(int(best["main_clsid"])))
            od_counts[best["end_zone"]][a["start_zone"]][cls_name] += 1
            od_events.append({
                "frame": int(best["end_f"]),
                "track_id": int(a["id"]),
                "origin": a["start_zone"],
                "dest": best["end_zone"],
                "class": cls_name,
                "conf": None,
                "count_method": "fallback_split_id",
            })
            
            used_end_ids.add(int(best["id"]))
            counted_ids.add(int(a["id"]))
            added_split += 1

            split_id_matches.append({
                "start_id": int(a["id"]),
                "end_id": int(best["id"]),
                "origin": a["start_zone"],
                "dest": best["end_zone"],
                "score": float(best_score),
            })
        
        return {
            "added_same_id": added_same,
            "added_split_id": added_split,
            "same_id_track_ids": same_id_track_ids,
            "split_id_matches": split_id_matches,
        }
