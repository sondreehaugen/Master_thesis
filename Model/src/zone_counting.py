"""
zone_counting.py
Vehicle and person counting logic for O-D matrix tracking.

Handles:
- Zone assignment and dwell-based counting
- Origin-destination (O-D) trip counting
- Person detection and counting by zone
- Track state management
"""

from collections import defaultdict, deque
from typing import Dict, Tuple, Optional, Any
import numpy as np
from .geometry import tracking_point_xyxy, point_in_polygon


class ZoneCounter:
    """
    Manages zone-based counting for vehicles and persons.
    Tracks O-D trips, dwell times, and person counts.
    """
    
    def __init__(
        self,
        gate_zones: Dict,
        fps: float,
        model,
        zone_dwell_frames: int = 3,
        min_track_frames: int = 5,
        track_finalize_gap_seconds: float = 2.0,
    ):
        """
        Initialize zone counter.
        
        Args:
            gate_zones: Dict of {zone_name: {polygon, dir, type}}
            fps: Video frame rate
            model: YOLO model instance (for class names)
            zone_dwell_frames: Frames to consider zone stable
            min_track_frames: Min frames for stable track
            track_finalize_gap_seconds: Seconds of inactivity before finalizing
        """
        self.gate_zones = gate_zones
        self.fps = fps
        self.model = model
        self.zone_dwell_frames = zone_dwell_frames
        self.min_track_frames = min_track_frames
        self.track_finalize_gap = int(fps * track_finalize_gap_seconds)
        
        # Zone info lookup
        self.zone_info_by_dir = {
            str(zinfo.get("dir", zname)): zinfo 
            for zname, zinfo in gate_zones.items()
        }
        
        # Tracking state
        self.track_history = defaultdict(list)
        self.class_conf_history = defaultdict(lambda: defaultdict(list))
        self.track_frame_count = defaultdict(int)
        self.last_seen_frame = defaultdict(int)
        self.zone_hist = defaultdict(lambda: deque(maxlen=max(zone_dwell_frames, 1)))
        self.seen_zones_for_id = defaultdict(set)
        
        # O-D tracking
        self.first_gate = {}
        self.last_gate = {}
        self.counted_trip = set()
        self.od_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.od_events = []
        
        # Person counting
        self.person_dwell_counts = defaultdict(lambda: defaultdict(int))
        self.person_counted = defaultdict(set)
        self.person_dwell_threshold = max(int(zone_dwell_frames), 10)
    
    def get_track_class(self, tid: int) -> Optional[str]:
        """Determine most stable class for a track ID."""
        if tid not in self.class_conf_history:
            return None
        
        class_stats = {
            cid: {
                "frames": len(cvals),
                "mean_conf": float(np.mean(cvals)),
            }
            for cid, cvals in self.class_conf_history[tid].items()
            if len(cvals) > 0
        }
        if not class_stats:
            return None
        
        stable_classes = {
            cid: stats for cid, stats in class_stats.items() 
            if stats["frames"] >= 5
        }
        
        if stable_classes:
            best_cid = max(
                stable_classes, 
                key=lambda cid: stable_classes[cid]["mean_conf"]
            )
        else:
            best_cid = max(class_stats, key=lambda cid: class_stats[cid]["frames"])
        
        return self.model.names.get(best_cid, str(best_cid))
    
    def _is_person_class(self, clsid: Optional[int] = None, cls_name: Optional[str] = None) -> bool:
        """Return True only for pedestrian classes."""
        if cls_name is None and clsid is not None:
            cls_name = self.model.names.get(int(clsid), str(clsid))
        n = str(cls_name).strip().lower() if cls_name is not None else ""
        return n in {"person", "pedestrian", "people"}
    
    def process_detection(
        self,
        tid: int,
        box_full_frame: Tuple[float, float, float, float],
        clsid: int,
        conf: float,
        frame_idx: int,
    ) -> None:
        """
        Process a single detection for zone counting.
        
        Args:
            tid: Track ID
            box_full_frame: [x1, y1, x2, y2] in full frame coords
            clsid: Class ID
            conf: Detection confidence
            frame_idx: Current frame index
        """
        tid = int(tid)
        clsid = int(clsid)
        
        self.track_frame_count[tid] += 1
        self.class_conf_history[tid][clsid].append(float(conf))
        self.last_seen_frame[tid] = frame_idx
        
        p = tracking_point_xyxy(box_full_frame)
        p_int = (int(round(p[0])), int(round(p[1])))
        
        # Track history
        track = self.track_history[tid]
        track.append(p_int)
        if len(track) > 150:  # Keep last 150 points
            track.pop(0)
        
        cls_name = self.get_track_class(tid)
        if cls_name is None:
            cls_name = self.model.names.get(clsid, str(clsid))
        
        # Zone assignment
        current_zone = None
        for zone_name, zone_info in self.gate_zones.items():
            poly = zone_info.get("polygon", [])
            if len(poly) < 3:
                continue
            if point_in_polygon(p_int, poly):
                current_zone = zone_info.get("dir", zone_name)
                break
        
        self.zone_hist[tid].append(current_zone)
        
        # Stable zone detection
        stable_zone = None
        vals = [z for z in self.zone_hist[tid] if z is not None]
        if len(vals) >= self.zone_dwell_frames:
            tail = vals[-self.zone_dwell_frames:]
            if len(tail) == self.zone_dwell_frames and len(set(tail)) == 1:
                stable_zone = tail[-1]
        
        if stable_zone is not None and self.track_frame_count[tid] >= self.min_track_frames:
            self.seen_zones_for_id[tid].add(stable_zone)
            
            # Get zone type
            zone_info = (
                self.gate_zones.get(stable_zone) or 
                self.zone_info_by_dir.get(stable_zone) or 
                {}
            )
            zone_type = zone_info.get("type", "vehicle")
            
            if zone_type == "vehicle":
                # Vehicle origin-destination tracking
                if tid not in self.first_gate:
                    if not self._is_person_class(clsid=clsid, cls_name=cls_name):
                        self.first_gate[tid] = stable_zone
                        self.last_gate[tid] = stable_zone
                else:
                    self.last_gate[tid] = stable_zone
            
            elif zone_type == "person":
                # Person dwell-based counting
                if self._is_person_class(clsid=clsid, cls_name=cls_name):
                    self.person_dwell_counts[stable_zone][tid] += 1
                    if (
                        tid not in self.person_counted[stable_zone]
                        and self.person_dwell_counts[stable_zone][tid] >= self.person_dwell_threshold
                    ):
                        self.person_counted[stable_zone].add(tid)
                        self.od_events.append({
                            "frame": int(frame_idx),
                            "track_id": tid,
                            "origin": stable_zone,
                            "dest": stable_zone,
                            "class": "person",
                            "conf": float(conf),
                            "count_method": "person_dwell",
                        })
    
    def finalize_counts(self, current_frame_idx: int, force_all: bool = False) -> int:
        """
        Finalize O-D counts for inactive vehicles.
        
        Args:
            current_frame_idx: Current frame index
            force_all: Force finalization of all remaining tracks
        
        Returns:
            Number of tracks finalized
        """
        finalized_count = 0
        
        for tid in list(self.first_gate.keys()):
            if tid in self.counted_trip:
                continue
            
            # Check inactivity
            last_seen = self.last_seen_frame.get(tid, current_frame_idx)
            frames_inactive = int(current_frame_idx) - int(last_seen)
            
            if not force_all and frames_inactive < self.track_finalize_gap:
                continue
            
            origin = self.first_gate.get(tid)
            dest = self.last_gate.get(tid)
            
            if origin is None or dest is None or origin == dest:
                continue
            
            cls_name = self.get_track_class(tid) or "unknown"
            
            self.od_counts[dest][origin][cls_name] += 1
            self.od_events.append({
                "frame": int(current_frame_idx),
                "track_id": tid,
                "origin": origin,
                "dest": dest,
                "class": cls_name,
                "conf": None,
                "count_method": "finalized_inactive",
            })
            self.counted_trip.add(tid)
            finalized_count += 1
        
        return finalized_count
    
    def get_results(self) -> Dict[str, Any]:
        """Get all accumulated counting results."""
        return {
            "od_counts": {
                d: {o: dict(c) for o, c in self.od_counts[d].items()} 
                for d in self.od_counts
            },
            "od_events": self.od_events,
            "person_counts": {z: len(ids) for z, ids in self.person_counted.items()},
            "seen_zones_for_id": {tid: sorted(v) for tid, v in self.seen_zones_for_id.items()},
            "first_gate": dict(self.first_gate),
            "last_gate": dict(self.last_gate),
            "track_frame_count": dict(self.track_frame_count),
            "track_history": {tid: list(pts) for tid, pts in self.track_history.items()},
        }
