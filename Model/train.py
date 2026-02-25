"""
Trener en YOLO26-modell på SVV-kamera-datasettet.

Standard-verdier er tilpasset 8 GB RAM (Mac med delt minne):
  - Modell:      yolo26n  (nano – minst minnebruk)
  - Batch:       4        (øk til 8 hvis maskinen tåler det)
  - Imgsz:       416      (lavere enn 640, reduserer minnebruk ~40 %)
  - Workers:     2        (færre parallelle dataprosesser)

Bruk:
    python train.py                          # standard (nano, batch 4, 416px)
    python train.py --model yolo26m.pt       # større modell hvis du har nok minne
    python train.py --batch 8 --imgsz 640    # større batch / bildestørrelse
    python train.py --epochs 50              # færre epoker
    python train.py --device mps             # Apple Silicon GPU
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# ── Standard-verdier (tilpasset 8 GB RAM) ────────────────────────────────────
DEFAULT_MODEL   = "yolo26n.pt"   # n=nano  s=small  m=medium  l=large  x=xlarge
DEFAULT_EPOCHS  = 100
DEFAULT_IMGSZ   = 416            # lavere enn 640 → ~40 % mindre minnebruk
DEFAULT_BATCH   = 4              # trygt for 8 GB; øk til 8 hvis OK
DEFAULT_WORKERS = 2              # færre parallelle dataprosesser
DEFAULT_DEVICE  = "mps"          # Apple Silicon GPU (raskere enn cpu)
DEFAULT_PROJECT = "runs/train"
DEFAULT_NAME    = "svv_kamera"
# ──────────────────────────────────────────────────────────────────────────────

DATASET_YAML = Path(__file__).parent / "dataset.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="Trener YOLOv8 på SVV-kamera-data")
    p.add_argument("--model",   default=DEFAULT_MODEL,   help="Modell-vekt-fil (f.eks. yolo26n.pt)")
    p.add_argument("--epochs",  default=DEFAULT_EPOCHS,  type=int)
    p.add_argument("--imgsz",   default=DEFAULT_IMGSZ,   type=int, help="Inngangsstørrelse i piksler")
    p.add_argument("--batch",   default=DEFAULT_BATCH,   type=int)
    p.add_argument("--workers", default=DEFAULT_WORKERS, type=int, help="Antall dataloader-tråder")
    p.add_argument("--device",  default=DEFAULT_DEVICE,  help="cpu | cuda | mps")
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--name",    default=DEFAULT_NAME)
    p.add_argument("--resume",  action="store_true",     help="Fortsett en avbrutt treningsøkt")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n🚀  Laster modell: {args.model}")
    model = YOLO(args.model)

    train_kwargs = dict(
        data      = str(DATASET_YAML),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        workers   = args.workers,  # begrens parallelle dataprosesser
        cache     = False,         # ikke last hele datasettet i RAM
        project   = args.project,
        name      = args.name,
        exist_ok  = True,          # overskriv ikke tidligere kjøringer
        pretrained= True,          # bruk forhåndstrente vekter som start
        patience  = 20,            # early stopping etter 20 epoker uten forbedring
        save      = True,
        plots     = True,
        resume    = args.resume,
    )

    if args.device:
        train_kwargs["device"] = args.device

    print(f"📊  Dataset:  {DATASET_YAML}")
    print(f"⚙️   Epoker:   {args.epochs}  |  Batch: {args.batch}  |  Bildestørrelse: {args.imgsz}  |  Workers: {args.workers}")
    print()

    results = model.train(**train_kwargs)

    print("\n✅  Trening ferdig!")
    print(f"   Beste vekter: {args.project}/{args.name}/weights/best.pt")

    # ── Rask validering ───────────────────────────────────────────────────────
    print("\n📈  Kjører validering på val-settet ...")
    metrics = model.val()
    print(f"   mAP50:    {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
