# Automated Intersection Analytics:
## Deep Learning-Based Traffic Monitoring under Real-World Conditions

A tracking pipeline developed for a master's thesis, for evaluating vehicle detection and tracking across day and night conditions with dynamic tracker switching and post-processing recovery.

---

## 📋 Overview

This system detects and tracks vehicles and pedestrians across dynamic lighting conditions using:
- **YOLO26n/YOLO26m** detection models fine-tuned on traffic data
- **ByteTrack, BoT-SORT, Day BoT-SORT, Night BoT-SORT (Conservative)** trackers
- **HSV saturation-based** day/night detection with hysteresis-based switching
- **Zone-based counting** for origin-destination (O-D) matrix analysis
- **Split-ID recovery** for continuity across tracker switches

### Key Results
| Scenario | Class Error | Route Error | Total Error |
|----------|-------------|-------------|-------------|
| **Daytime** | 2.3% | 6.2% | 0.5% |
| **Nighttime** | 9.0% | 3.7% | 1.5% |
| **Transition** | 0.0% | 0.0% | 0.0% |

Model used for reported results: YOLO26m (see `Model/` for weights).

---

## 🏗️ Architecture

### Module Structure
```
Model/src/
├── run_video_deploy.py       # Main orchestrator
├── day_night_detection.py    # Lighting detection + switching
├── zone_counting.py          # Zone tracking & O-D counting
├── split_id_fallback.py      # Split track ID recovery
├── evaluation.py             # Experiment pipeline
├── GT.py                     # Ground truth (aggregate counts)
├── tracking_utils.py         # Setup & visualization
├── geometry.py               # Point/polygon utilities
└── __init__.py               # Package exports
```

### Dependency Graph
```
Model/tracking.ipynb (MAIN NOTEBOOK)
    ├── run_video_deploy()           → Core tracking pipeline
    │   ├── DayNightDetector        → HSV-based detection
    │   ├── TrackerSwitcher         → Hysteresis switching
    │   ├── ZoneCounter             → Counting logic
    │   └── SplitIDRecovery         → Post-processing
    ├── run_experiments()            → Batch evaluation
    ├── display_model_results()      → O-D matrix visualization
    └── plot_class_distribution()    → Class analysis
```

### Class Taxonomy (8 classes)
```
Vehicle + pedestrian classes:
- Car
- Light Truck
- Heavy Truck
- Semi-trailer
- Bus
- Motorcycle
- Bicycle
- Person (pedestrian)
```

---

## 🚀 Quick Start

### 1. Installation

#### Local Setup
```bash
# Clone/download the project
cd Master_Thesis/Model

# Create Python environment (Python 3.10+ recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -U ultralytics opencv-python pandas numpy scikit-learn torch

# (Optional) GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Google Colab
```python
!git clone <repo-url>  # If using Git
!pip install -U ultralytics opencv-python pandas numpy scikit-learn
```

### 2. Prepare Data

Training outputs and datasets are available here:
https://drive.google.com/drive/folders/1NkWN7G3pADlT1_Fy0xBrRFxEqEnBW7Qv

The Drive folder contains:
- `YOLO results/`: YOLO26n / YOLO26m tuning results
- `Data/`: train/val/test split and the videos used for tracking

**Required Files Structure:**
```
Model/
├── Video/
│   ├── Fv587_Haukeland_15min.mp4           # Day video
│   ├── Fv587_Haukeland_night_1h.mp4        # Night video
│   └── Fv587_Haukeland_10min_transition.mp4 # Transition
├── src/
│   └── [source modules]
├── [pretrained models]
│   ├── YOLO26n_final.pt
│   ├── YOLO26m_final.pt
│   └── *.yaml (tracker configs)
└── tracking.ipynb
```

**Ground Truth Format:**
Ground truth is stored in `Model/src/GT.py` as aggregate route and class counts per video:
```python
GT_BY_VIDEO = {
    "Fv587_Haukeland_15min.mp4": {
        "S->N": {"car": 134, "light truck": 5, ...},
        "N->S": {"car": 71, "light truck": 4, ...},
        ...,
        "Gangfeltet": {"person": 3},
    },
    ...
}
```

### 3. Run the Notebook

```python
# In Model/tracking.ipynb, run cells sequentially:
# 1. Setup & Imports
# 2. Define Zones (customize polygon coordinates for your camera)
# 3. Run experiments on DAY/NIGHT/TRANSITION videos
# 4. View results and visualizations
```

---

## 🎯 Usage Examples

### Basic Single Video Tracking
```python
from src.run_video_deploy import run_video_deploy

results = run_video_deploy(
    model=model,
    video_source="path/to/video.mp4",
    runs_dir="./runs",
    day_tracker="day_botsort.yaml",
    night_tracker="night_botsort_conservative.yaml",
    gate_zones=GATE_ZONES,
    sat_threshold=20.0,  # HSV saturation threshold
)

print(f"Tracked {results['unique_ids']} vehicles")
print(f"O-D Counts: {results['od_counts']}")
```

### Custom Zone Definition
```python
GATE_ZONES = {
    "SOUTH": {
        "polygon": [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
        "dir": "SOUTH",
        "type": "vehicle"  # or "person"
    },
    "PEDESTRIAN_ZONE": {
        "polygon": [(p1, p2), ...],
        "dir": "CROSSING",
        "type": "person"
    }
}
```

### Batch Experiments
```python
all_results, all_pred_tables, all_diff_tables, summary_df, persons_df = run_experiments(
    models=["YOLO26n_final.pt", "YOLO26m_final.pt"],
    trackers={
        "day_botsort": "day_botsort.yaml",
        "night_botsort": "night_botsort_conservative.yaml"
    },
    video_name="Fv587_Haukeland_15min.mp4",
    gt_df=gt_df_vehicles,
    gt_persons=gt_persons,
    gate_zones=GATE_ZONES,
)

# Results automatically saved to runs/instance/
```

---

## 📊 Outputs

### Tracking Results
```python
{
    "mp4_path": "path/to/output_video.mp4",
    "elapsed_time": 123.5,  # seconds
    "frame_count": 3000,
    "unique_ids": 847,  # stable tracks
    
    # Origin-Destination Matrix
    "od_counts": {
        "SOUTH": {
            "NORTH": {"Car": 45, "Bus": 2, ...},
            "EAST": {"Car": 38, ...}
        },
        ...
    },
    
    # Detailed tracking events
    "od_events": [
        {
            "frame": 1234,
            "track_id": 42,
            "origin": "NORTH",
            "dest": "SOUTH",
            "class": "Car",
            "count_method": "finalized_inactive"
        },
        ...
    ],
    
    # Day/Night Statistics
    "mode_frames": {"day": 1800, "night": 1200},
    "mode_switches": [
        {"frame": 1500, "mode": "night", "sat": 18.5, "lum": 42.0},
        ...
    ],
    
    # Split ID Recovery Stats
    "fallback_stats": {
        "added_same_id": 5,
        "added_split_id": 12
    }
}
```

### Visualization Outputs
- **Annotated video** with trajectory tracking, zone overlays, O-D counts
- **O-D matrices** (confusion matrices) showing vehicle flow
- **Class distribution plots** by origin-destination pair
- **Person counting results** by zone
- **Mode switch timeline** (day → night transitions)

---

## ⚙️ Configuration Parameters

### Day/Night Detection
```python
DayNightDetector(
    sat_threshold=20.0,  # HSV saturation (< threshold = night)
    window=60            # Frames for rolling average (~2s @ 30fps)
)
```

### Tracker Switching
```python
TrackerSwitcher(
    initial_mode="day",
    fps=30.0,
    hold_seconds=5.0     # Seconds before committing to new mode
)
```

### Zone Counting
```python
ZoneCounter(
    gate_zones=GATE_ZONES,
    zone_dwell_frames=3,              # Frames in same zone = stable
    min_track_frames=5,               # Min frames for stable track
    track_finalize_gap_seconds=2.0    # Inactivity before finalizing
)
```

### Split ID Recovery
```python
SplitIDRecovery(
    enable=True,
    max_dist_raw=180.0,       # Max euclidean distance
    max_dist_pred=160.0,      # Max distance with velocity prediction
    max_gap_seconds=3.0       # Max temporal gap between fragments
)
```

---

## 📈 Evaluation Metrics

### Detection
- **mAP50-95**: YOLO mean average precision across IoU thresholds
- **mAP50**: Stricter single-threshold AP for validation
- **Per-class precision/recall** for each vehicle type

### Tracking
- **Sum MAE**: Sum of absolute errors in O-D counts (best metric)
- **Cell MAE**: Mean absolute error per O-D cell
- **Total Absolute % Error**: Percentage error on total counts

### Counting
- **Perfect routes**: 100% correct origin-destination pairs
- **Split routes**: Correctly recovered across tracker switches
- **Missed routes**: Untracked vehicles

---

## 🔧 Advanced Usage

### Custom Tracker Configuration
```python
# Create custom tracker YAML (e.g., my_tracker.yaml)
# Then use in run_video_deploy:
results = run_video_deploy(
    ...,
    day_tracker="path/to/my_tracker.yaml",
    night_tracker="path/to/my_night_tracker.yaml"
)
```

### Extract Tracking Data
```python
# Access all tracks for post-processing
all_tracks = results["all_tracks_by_frame"]
# {frame_idx: [(track_id, box, conf, class_id, ...)]

for frame_idx, tracks in all_tracks.items():
    for tid, box, conf, clsid, _ in tracks:
        print(f"Frame {frame_idx}: ID {tid} ({model.names[clsid]})")
```

### Disable Split ID Recovery
```python
results = run_video_deploy(
    ...,
    enable_split_id_fallback=False  # Skip post-processing
)
```

---

## 📚 Module Reference

### `run_video_deploy.py`
Main tracking function orchestrating all components.
- **Input**: YOLO model, video path, zone definitions
- **Output**: O-D counts, tracking events, video file

### `day_night_detection.py`
- `DayNightDetector.analyze_frame()`: Returns mode ("day"/"night") + metrics
- `TrackerSwitcher.update()`: Returns current active mode with hysteresis

### `zone_counting.py`
- `ZoneCounter.process_detection()`: Update tracking for single detection
- `ZoneCounter.finalize_counts()`: Finalize inactive tracks
- `ZoneCounter.get_results()`: Retrieve all counting data

### `split_id_fallback.py`
- `SplitIDRecovery.apply()`: Post-process to recover split IDs
  - Same-ID recovery: Tracks that skipped origin/dest
  - Split-ID matching: Fragment matching across switches

### `evaluation.py`
- `run_experiments()`: Batch evaluation across models/trackers
- `display_model_results()`: O-D matrix visualization
- `plot_class_distribution_detailed()`: Per-class analysis

### `GT.py`
- `load_gt_vehicles()`: Load ground truth vehicle annotations
- `load_gt_persons()`: Load pedestrian ground truth
- `CLASS_ORDER`, `FLOW_ORDER`: Class and flow direction constants

---

## 🐛 Troubleshooting

### Tracker switches too frequently
→ Increase `hold_seconds` in `TrackerSwitcher` or `tracker_hold_seconds` in `run_video_deploy()`

### Missing O-D counts
→ Check `zone_dwell_frames` and `min_track_frames` thresholds  
→ Verify zone polygons are correctly defined  
→ Enable split ID fallback for recovery

### CUDA out of memory
→ Reduce `conf_thres` for fewer detections  
→ Process video in smaller chunks with `max_frames`

### Poor night tracking
→ Verify `night_tracker` (Night BoT-SORT Conservative recommended)  
→ Adjust `sat_threshold` (lower = more conservative night mode)

---

## 📖 Citation

For thesis or publication:
```bibtex
@thesis{haugen_2026,
    author = {Sondre Eliassen Haugen and Eirik \AA rdal Hjelm},
  title = {Exploring Machine Vision for Vehicle and Object Detection in Road Environments},
    school = {Norwegian University of Life Sciences (NMBU)},
  year = {2026}
}
```

---

## 📝 License & Attribution

- **YOLO**: Ultralytics YOLOv8 (AGPL-3.0)
- **YOLO models**: YOLO26n / YOLO26m (custom weights; implementation based on the Ultralytics codebase)
- **Trackers**: ByteTrack, BoT-SORT (Apache 2.0, BSD)
- **Custom Code**: Master's Thesis Project

---

## ✅ Checklist for Running

- [ ] Python 3.10+ installed
- [ ] Dependencies installed (see Installation section)
- [ ] Video files in `Model/Video/` or accessible path
- [ ] Pretrained models (YOLO26m_final.pt, etc.) downloaded
- [ ] Tracker config files (*.yaml) in correct directory
- [ ] Ground truth annotations loaded in `GT.py`
- [ ] Zone polygons defined for your camera view
- [ ] GPU available (recommended for real-time performance)
- [ ] Run `Model/tracking.ipynb` sequentially from top

---

## 📧 Questions?

Refer to:
- Inline code comments for implementation details
- `Model/tracking.ipynb` for end-to-end workflow examples
- Module docstrings for function signatures

---

**Last Updated**: May 2026  
**Status**: Research / thesis code  
**Python**: 3.10+ (tested in the thesis workflow on newer versions)  
**Main Dependencies**: PyTorch, OpenCV, Ultralytics, Pandas
