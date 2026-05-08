"""Inference script for HW3 Mask R-CNN submission generation."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import tifffile as tiff
import torch
import torchvision
from pycocotools import mask as mask_utils
from tqdm import tqdm

NUM_CLASSES = 5


def read_tif(path: Path) -> np.ndarray:
    """Read a TIF image."""
    return tiff.imread(str(path))


def prepare_test_image(image_path: Path) -> torch.Tensor:
    """Convert a test image to a normalized tensor."""
    image = read_tif(image_path).astype(np.float32)

    if image.max() > 1.0:
        image = image / 255.0

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    return torch.from_numpy(image).permute(2, 0, 1)


def get_model(num_classes: int) -> torch.nn.Module:
    """Build a Mask R-CNN model with custom heads."""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
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


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a trained Mask R-CNN checkpoint."""
    model = get_model(NUM_CLASSES)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Run inference on one image."""
    image = prepare_test_image(image_path).to(device)

    with torch.no_grad():
        prediction = model([image])[0]

    return prediction


def filter_prediction(
    prediction: Dict[str, torch.Tensor],
    min_score: float,
    min_mask_area: int,
    mask_threshold: float,
) -> List[Dict]:
    """Filter predictions by confidence score and mask area."""
    filtered = []
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    masks = prediction["masks"].detach().cpu()

    for index, score_tensor in enumerate(scores):
        score = score_tensor.item()

        if score < min_score:
            continue

        binary_mask = masks[index, 0] > mask_threshold
        mask_area = binary_mask.sum().item()

        if mask_area < min_mask_area:
            continue

        filtered.append(
            {
                "label": int(labels[index].item()),
                "score": float(score),
                "mask": binary_mask.numpy().astype(np.uint8),
            }
        )

    return filtered


def encode_binary_mask_to_rle(binary_mask: np.ndarray) -> Dict:
    """Encode a binary mask into COCO RLE format."""
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(binary_mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def prediction_to_submission_items(
    image_id: int,
    prediction: Dict[str, torch.Tensor],
    min_score: float,
    min_mask_area: int,
    mask_threshold: float,
) -> List[Dict]:
    """Convert one prediction to the expected submission format."""
    submission_items = []
    filtered_predictions = filter_prediction(
        prediction,
        min_score=min_score,
        min_mask_area=min_mask_area,
        mask_threshold=mask_threshold,
    )

    for prediction_item in filtered_predictions:
        submission_items.append(
            {
                "image_id": int(image_id),
                "category_id": int(prediction_item["label"]),
                "segmentation": encode_binary_mask_to_rle(
                    prediction_item["mask"]
                ),
                "score": float(prediction_item["score"]),
            }
        )

    return submission_items


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("hw3-data-release"))
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--mapping-path", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("test-results.json"))
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--min-mask-area", type=int, default=20)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """Generate the competition submission file."""
    args = parse_args()
    args.test_dir = args.test_dir or args.data_dir / "test_release"
    args.mapping_path = (
        args.mapping_path or args.data_dir / "test_image_name_to_ids.json"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    with open(args.mapping_path, "r", encoding="utf-8") as file:
        test_mapping = json.load(file)

    submission_results = []

    for item in tqdm(test_mapping, desc="Inference"):
        image_path = args.test_dir / item["file_name"]
        prediction = predict_image(model, image_path, device)
        submission_results.extend(
            prediction_to_submission_items(
                image_id=item["id"],
                prediction=prediction,
                min_score=args.min_score,
                min_mask_area=args.min_mask_area,
                mask_threshold=args.mask_threshold,
            )
        )

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(submission_results, file)

    print(f"Submission saved to {args.output}")
    print(f"Number of predictions: {len(submission_results)}")


if __name__ == "__main__":
    main()
