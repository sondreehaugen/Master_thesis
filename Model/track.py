"""
Multi-Object Tracking (MOT) med YOLO26 på SVV-kamera-video.

Bruk:
    python track.py --source video.mp4                    # standard (BoT-SORT)
    python track.py --source video.mp4 --tracker bytetrack
    python track.py --source video.mp4 --model runs/train/svv_kamera/weights/best.pt
    python track.py --source 0                            # webkamera
    python track.py --source video.mp4 --save            # lagre annotert video
    python track.py --source video.mp4 --count           # tell kjøretøy

Trackere tilgjengelig:
    botsort.yaml   – standard, støtter ReID
    bytetrack.yaml – raskere, bra på tett trafikk
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Standard-verdier ──────────────────────────────────────────────────────────
DEFAULT_MODEL   = "yolo26n.pt"        # bytt til best.pt etter trening
DEFAULT_TRACKER = "botsort.yaml"      # eller bytetrack.yaml
DEFAULT_CONF    = 0.25
DEFAULT_IOU     = 0.7
TRAIL_LENGTH    = 40                  # antall frames å tegne spor for
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: "Buss",
    1: "Long combination vehicle",
    2: "Semi-trailer",
    3: "bicycle",
    4: "car",
    5: "motorcycle",
    6: "person",
    7: "truck",
}

# Farge per klasse (BGR)
CLASS_COLORS = {
    0: (0, 128, 255),    # Buss         – oransje
    1: (0, 0, 200),      # LCV          – rød
    2: (0, 60, 150),     # Semi-trailer – mørkerød
    3: (0, 255, 128),    # bicycle      – grønn
    4: (255, 200, 0),    # car          – blå
    5: (255, 0, 200),    # motorcycle   – lilla
    6: (200, 200, 200),  # person       – hvit
    7: (0, 100, 255),    # truck        – oransje
}


def parse_args():
    p = argparse.ArgumentParser(description="YOLO26 MOT på SVV-kamera")
    p.add_argument("--source",  required=True, help="Videofil, mappe, URL eller '0' for webkamera")
    p.add_argument("--model",   default=DEFAULT_MODEL)
    p.add_argument("--tracker", default=DEFAULT_TRACKER, choices=["botsort.yaml", "bytetrack.yaml"])
    p.add_argument("--conf",    default=DEFAULT_CONF, type=float, help="Konfidens-terskel")
    p.add_argument("--iou",     default=DEFAULT_IOU,  type=float, help="IoU-terskel for NMS")
    p.add_argument("--imgsz",   default=640,          type=int)
    p.add_argument("--device",  default="",           help="cpu | cuda | mps")
    p.add_argument("--save",    action="store_true",  help="Lagre annotert video")
    p.add_argument("--count",   action="store_true",  help="Tell unike objekt-IDer per klasse")
    p.add_argument("--reid",    action="store_true",  help="Aktiver ReID i BoT-SORT (bedre, litt tregere)")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)
    track_history: dict[int, list] = defaultdict(list)
    class_counts:  dict[int, set]  = defaultdict(set)

    source = int(args.source) if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise SystemExit(f"❌  Kan ikke åpne kilde: {args.source}")

    # Videoskriving
    writer = None
    if args.save:
        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        out_path = Path(args.source).stem + "_tracked.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"💾  Lagrer til: {out_path}")

    track_kwargs = dict(
        conf    = args.conf,
        iou     = args.iou,
        imgsz   = args.imgsz,
        tracker = args.tracker,
        persist = True,       # nødvendig for å beholde ID-er mellom frames
        verbose = False,
    )
    if args.device:
        track_kwargs["device"] = args.device

    print(f"\n🎯  Modell:   {args.model}")
    print(f"📡  Tracker: {args.tracker}")
    print(f"🎥  Kilde:   {args.source}")
    print("   Trykk 'q' for å avslutte\n")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(frame, **track_kwargs)
        result  = results[0]

        annotated = result.plot()   # tegner bokser og klasse-etiketter

        # ── Tegn spor og tell per klasse ─────────────────────────────────────
        if result.boxes is not None and result.boxes.id is not None:
            boxes      = result.boxes.xywh.cpu().numpy()
            track_ids  = result.boxes.id.int().cpu().tolist()
            class_ids  = result.boxes.cls.int().cpu().tolist()

            for box, tid, cid in zip(boxes, track_ids, class_ids):
                x, y = float(box[0]), float(box[1])
                history = track_history[tid]
                history.append((x, y))
                if len(history) > TRAIL_LENGTH:
                    history.pop(0)

                # Tegn spor-linje
                if len(history) > 1:
                    color  = CLASS_COLORS.get(cid, (200, 200, 200))
                    pts    = np.array(history, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(annotated, [pts], isClosed=False, color=color, thickness=2)

                if args.count:
                    class_counts[cid].add(tid)

        # ── Tell-overlay ──────────────────────────────────────────────────────
        if args.count:
            y_off = 30
            for cid, ids in sorted(class_counts.items()):
                label = f"{CLASS_NAMES.get(cid, cid)}: {len(ids)}"
                cv2.putText(annotated, label, (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            CLASS_COLORS.get(cid, (255, 255, 255)), 2)
                y_off += 28

        if writer:
            writer.write(annotated)

        cv2.imshow("YOLO26 – SVV Tracking", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── Sluttoppsummering ─────────────────────────────────────────────────────
    if args.count and class_counts:
        print("\n📊  Unike objekter sett:")
        total = 0
        for cid, ids in sorted(class_counts.items()):
            n = len(ids)
            total += n
            print(f"   {CLASS_NAMES.get(cid, cid):<30} {n}")
        print(f"   {'TOTALT':<30} {total}")


if __name__ == "__main__":
    main()
