from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class RealEstateDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        images_dir: Path,
        tabular_features: list,
        target_col: str | None = None,
        transform=None,
    ):
        csv_path = Path(csv_path)
        images_dir = Path(images_dir)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found at {csv_path}")

        if not images_dir.exists():
            raise FileNotFoundError(f"Image directory not found at {images_dir}")

        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.tabular_features = tabular_features
        self.target_col = target_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -------- IMAGE --------
        raw_id = row["id"]
        if isinstance(raw_id, float) and raw_id.is_integer():
            image_id = str(int(raw_id))
        else:
            image_id = str(raw_id).strip()
            if image_id.endswith(".0"):
                image_id = image_id[:-2]

        img_path = self.images_dir / f"{image_id}.png"
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image for ID {raw_id} not found at {img_path}. Run src/data_fetcher.py to generate imagery."
            )

        with Image.open(img_path) as pil_image:
            image = pil_image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        # -------- TABULAR --------
        tabular_values = row[self.tabular_features].astype("float32").values
        tabular = torch.tensor(tabular_values, dtype=torch.float32)

        # -------- TARGET --------
        if self.target_col is not None:
            target = torch.tensor(row[self.target_col], dtype=torch.float32)
            return image, tabular, target

        return image, tabular
