import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


def make_detector(num_classes: int = 2):
    """
    Faster R-CNN ROI detector.
    num_classes включает background => 2 (bg + roi)
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model


class NsaRegNet(nn.Module):
    """
    ResNet18 -> head predicts [cos(2t), sin(2t)]
    """
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet18(weights="DEFAULT")
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        f = self.backbone(x)
        y = self.head(f)
        return y


def pred2_to_deg(y2: torch.Tensor) -> torch.Tensor:
    """
    y2: (B,2)
    returns degrees in [0, 90]
    """
    y = F.normalize(y2, dim=1)
    c = y[:, 0]
    s = y[:, 1]
    ang2 = torch.atan2(s, c)        # [-pi, pi]
    ang = 0.5 * ang2
    ang = torch.abs(ang)
    ang = torch.clamp(ang, 0, torch.pi / 2)
    return ang * 180.0 / torch.pi


def load_detector_weights(det_path: str, map_location="cpu"):
    model = make_detector(num_classes=2)
    ckpt = torch.load(det_path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    return model


def load_regressor_weights(reg_path: str, map_location="cpu"):
    model = NsaRegNet()
    ckpt = torch.load(reg_path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    return model
