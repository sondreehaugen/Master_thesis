from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # svv_kamera/

VIDEO = BASE_DIR / "raw/1min/Fv587_Haukeland_1229050_1_00.mp4"

print(f"Bruker video: {VIDEO}")

model = YOLO("yolo26m.pt")

model.predict(
    source=str(VIDEO),
    save=True,
    project=str(BASE_DIR / "processed"),
    name="yolo_test",
    conf=0.3
)
