import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet_Encoder(nn.Module):
    def __init__(
        self,
        version: str = "resnet18",
        embedding_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        if version == "resnet18":
            model = models.resnet18(pretrained=pretrained)
        elif version == "resnet50":
            model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Unsupported ResNet version")

        feature_dim = model.fc.in_features
        model.fc = nn.Identity()  # type: ignore
        self.backbone = model
        self.feature_dim = feature_dim
        self.projection = nn.Linear(feature_dim, embedding_dim)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet_Encoder(
        version="resnet18", embedding_dim=256, freeze_backbone=True
    ).to(device)

    x = torch.randn(8, 3, 224, 224).to(device)
    y = model(x)
    print(y.shape)  # torch.Size([8, 256])
