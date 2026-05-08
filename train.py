"""Training script for HW3 instance segmentation with Mask R-CNN."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile as tiff
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

CLASS_FILES = {
    1: "class1.tif",
    2: "class2.tif",
    3: "class3.tif",
    4: "class4.tif",
}
NUM_CLASSES = 5


def read_tif(path: Path) -> np.ndarray:
    """Read a TIF image or mask."""
    return tiff.imread(str(path))


def extract_instances_from_class_mask(
    class_mask: np.ndarray,
    class_id: int,
) -> List[Dict[str, np.ndarray]]:
    """Extract one binary mask per instance from a class mask."""
    instances = []
    instance_ids = np.unique(class_mask)
    instance_ids = instance_ids[instance_ids != 0]

    for instance_id in instance_ids:
        binary_mask = (class_mask == instance_id).astype(np.uint8)
        instances.append(
            {
                "class_id": class_id,
                "instance_id": int(instance_id),
                "mask": binary_mask,
            }
        )

    return instances


def get_bounding_box(mask: np.ndarray) -> Optional[List[int]]:
    """Compute the bounding box of a binary mask."""
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def load_training_sample(sample_dir: Path) -> Tuple[np.ndarray, List[Dict]]:
    """Load one image and all its annotated instances."""
    image = read_tif(sample_dir / "image.tif")
    all_instances = []

    for class_id, class_file in CLASS_FILES.items():
        mask_path = sample_dir / class_file

        if not mask_path.exists():
            continue

        class_mask = read_tif(mask_path)
        all_instances.extend(
            extract_instances_from_class_mask(class_mask, class_id)
        )

    enriched_instances = []
    for instance in all_instances:
        bbox = get_bounding_box(instance["mask"])
        if bbox is None:
            continue
        instance["bbox"] = bbox
        enriched_instances.append(instance)

    return image, enriched_instances


class CellInstanceDataset(Dataset):
    """Dataset for the HW3 cell instance segmentation task."""

    def __init__(self, train_dir: Path, use_transforms: bool = True) -> None:
        self.train_dir = Path(train_dir)
        self.use_transforms = use_transforms
        self.sample_dirs = sorted(
            path for path in self.train_dir.iterdir() if path.is_dir()
        )

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample_dir = self.sample_dirs[index]
        image, instances = load_training_sample(sample_dir)
        image_tensor = self._prepare_image(image)
        target = self._prepare_target(instances, index, image_tensor)

        if self.use_transforms:
            image_tensor, target = apply_train_transforms(image_tensor, target)

        return image_tensor, target

    @staticmethod
    def _prepare_image(image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32)

        if image.max() > 1.0:
            image = image / 255.0

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        if image.shape[-1] == 4:
            image = image[:, :, :3]

        return torch.from_numpy(image).permute(2, 0, 1)

    @staticmethod
    def _prepare_target(
        instances: List[Dict],
        image_id: int,
        image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        boxes = []
        labels = []
        masks = []

        for instance in instances:
            boxes.append(instance["bbox"])
            labels.append(instance["class_id"])
            masks.append(instance["mask"])

        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            masks_tensor = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            _, height, width = image.shape
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            masks_tensor = torch.zeros((0, height, width), dtype=torch.uint8)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        return {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor([image_id]),
            "area": area,
            "iscrowd": iscrowd,
        }


def horizontal_flip(
    image: torch.Tensor,
    target: Dict[str, torch.Tensor],
    probability: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Randomly apply an horizontal flip to the image and target."""
    if torch.rand(1).item() >= probability or target["boxes"].numel() == 0:
        return image, target

    image = torch.flip(image, dims=[2])
    _, _, width = image.shape
    boxes = target["boxes"].clone()

    xmin = width - boxes[:, 2]
    xmax = width - boxes[:, 0]
    boxes[:, 0] = xmin
    boxes[:, 2] = xmax

    target = target.copy()
    target["boxes"] = boxes
    target["masks"] = torch.flip(target["masks"], dims=[2])

    return image, target


def vertical_flip(
    image: torch.Tensor,
    target: Dict[str, torch.Tensor],
    probability: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Randomly apply a vertical flip to the image and target."""
    if torch.rand(1).item() >= probability or target["boxes"].numel() == 0:
        return image, target

    image = torch.flip(image, dims=[1])
    _, height, _ = image.shape
    boxes = target["boxes"].clone()

    ymin = height - boxes[:, 3]
    ymax = height - boxes[:, 1]
    boxes[:, 1] = ymin
    boxes[:, 3] = ymax

    target = target.copy()
    target["boxes"] = boxes
    target["masks"] = torch.flip(target["masks"], dims=[1])

    return image, target


def apply_train_transforms(
    image: torch.Tensor,
    target: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply training augmentations."""
    image, target = horizontal_flip(image, target)
    image, target = vertical_flip(image, target)
    return image, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[Tuple, Tuple]:
    """Collate function required by torchvision detection models."""
    return tuple(zip(*batch))


def get_model(num_classes: int) -> torch.nn.Module:
    """Build a Mask R-CNN model with custom heads."""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights="DEFAULT"
    )

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features_box,
            num_classes,
        )
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = (
        torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask,
            256,
            num_classes,
        )
    )

    return model


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Train the model for one epoch and return the average loss."""
    model.train()
    total_loss = 0.0

    for images, targets in tqdm(data_loader, desc="Training"):
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / max(len(data_loader), 1)


def validate_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute validation loss for one epoch."""
    model.train()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validation"):
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    return total_loss / max(len(data_loader), 1)


def create_data_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    dataset = CellInstanceDataset(args.train_dir, use_transforms=True)
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()

    val_size = int(len(dataset) * args.val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("hw3-data-release"))
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("best_maskrcnn_hw3.pth"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--eta-min", type=float, default=0.00001)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()
    args.train_dir = args.train_dir or args.data_dir / "train"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_data_loaders(args)
    model = get_model(NUM_CLASSES).to(device)

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.eta_min,
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss = validate_one_epoch(model, val_loader, device)
        scheduler.step()

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output)
            print(f"Best model saved to {args.output}")


if __name__ == "__main__":
    main()
