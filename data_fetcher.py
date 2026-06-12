import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentinelhub import (
    SHConfig,
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions
)
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"


def load_env_file(path: Path) -> None:
    """Populate environment variables from a simple KEY=VALUE .env file."""

    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            cleaned_value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), cleaned_value)


load_env_file(ENV_PATH)

config = SHConfig()
config.sh_client_id = os.getenv("SENTINELHUB_CLIENT_ID")
config.sh_client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET")

if not config.sh_client_id or not config.sh_client_secret:
    raise RuntimeError(
        "SENTINELHUB_CLIENT_ID and SENTINELHUB_CLIENT_SECRET must be set in the environment."
    )

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
IMAGES_DIR = DATA_DIR / "images"

TRAIN_CSV = RAW_DATA_DIR / "TRAIN.csv"
TEST_CSV = RAW_DATA_DIR / "TEST.csv"

TRAIN_IMG_DIR = IMAGES_DIR / "train"
TEST_IMG_DIR = IMAGES_DIR / "test"
TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
RESOLUTION = 10  

evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02"],
    output: { bands: 3 }
  };
}

function evaluatePixel(sample) {
  return [sample.B04, sample.B03, sample.B02];
}
"""

def download_images(csv_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(csv_path)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = output_dir / f"{row['id']}.png"
        if img_path.exists():
            continue

        try:
            lat = float(row["lat"])
            lon = float(row["long"])
        except (KeyError, ValueError) as exc:
            print(f"Skipping ID {row.get('id', 'UNKNOWN')}: invalid coordinates ({exc}).")
            continue

        bbox = BBox(
            bbox=[lon - 0.005, lat - 0.005, lon + 0.005, lat + 0.005],
            crs=CRS.WGS84
        )

        size = bbox_to_dimensions(bbox, resolution=RESOLUTION)

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=("2023-01-01", "2023-12-31"),
                    mosaicking_order="mostRecent"
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.PNG)
            ],
            bbox=bbox,
            size=size,
            config=config
        )

        try:
            image = request.get_data()[0]
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE))
            image.save(img_path)
        except Exception as e:
            print(f"Failed for ID {row['id']}: {e}")

if __name__ == "__main__":
    print("Downloading TRAIN satellite images...")
    download_images(TRAIN_CSV, TRAIN_IMG_DIR)

    print("Downloading TEST satellite images...")
    download_images(TEST_CSV, TEST_IMG_DIR)