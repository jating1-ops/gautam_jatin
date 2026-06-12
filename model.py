import torch
import torch.nn as nn
import torchvision.models as models


# =========================
# IMAGE ENCODER
# =========================
class ImageEncoder(nn.Module):
    def __init__(self, output_dim=192):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        # ---- Partial unfreeze: only last block ----
        for name, param in backbone.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.backbone = backbone

        # ---- Projection head ----
        self.projection = nn.Sequential(
            nn.Linear(num_features, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection(x)
        return x


# =========================
# TABULAR ENCODER
# =========================
class TabularEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=192):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# MULTIMODAL REGRESSOR
# =========================
class MultimodalRegressor(nn.Module):
    def __init__(self, tabular_dim):
        super().__init__()

        self.image_encoder = ImageEncoder(output_dim=192)
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_dim,
            output_dim=192
        )

        self.regressor = nn.Sequential(
            nn.Linear(192 + 192, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, image, tabular):
        img_feat = self.image_encoder(image)
        tab_feat = self.tabular_encoder(tabular)

        fused = torch.cat([img_feat, tab_feat], dim=1)
        output = self.regressor(fused)

        return output.squeeze(1)
