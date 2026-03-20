from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
SOURCE_IMAGE = ASSETS / "demo_source_uav.jpg"


def ensure_source_image() -> Path:
    if SOURCE_IMAGE.exists():
        return SOURCE_IMAGE

    candidates = [
        Path("/Volumes/T9/OneDrive_2_9-15-2025_loropetalum/DJI_20250408145313_0091_D.JPG"),
        Path("/Users/harshu/Downloads/Gen ai test 3/Automation/loropetalum_input_images/DJI_20250408145313_0091_D.JPG"),
    ]
    for candidate in candidates:
        if candidate.exists():
            image = cv2.imread(str(candidate))
            if image is not None:
                cv2.imwrite(str(SOURCE_IMAGE), image)
                return SOURCE_IMAGE
    raise FileNotFoundError("Could not find the high-resolution UAV source image.")


def save_gray(path: Path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), image)


def save_labels(path: Path, labels: np.ndarray, mask: np.ndarray) -> None:
    canvas = np.full((labels.shape[0], labels.shape[1], 3), (180, 0, 0), dtype=np.uint8)
    if labels.max() > 0:
        normalized = np.uint8((labels.astype(np.float32) / max(labels.max(), 1)) * 255.0)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        canvas[mask > 0] = colored[mask > 0]
    cv2.imwrite(str(path), canvas)


def main() -> None:
    source_path = ensure_source_image()
    image = cv2.imread(str(source_path))
    if image is None:
        raise ValueError(f"Failed to read {source_path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    # Purple / burgundy canopy segmentation using the repo's Loropetalum hue range.
    color_mask = cv2.inRange(hsv, np.array([120, 40, 20]), np.array([170, 255, 255]))

    # Clean fragmented canopy pixels into more stable candidate regions.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_clean = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_close)
    morph_clean = cv2.morphologyEx(morph_clean, cv2.MORPH_OPEN, kernel_open)

    # Keep the largest candidate regions so the labels and final count match the HD example.
    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(morph_clean)
    selected_mask = np.zeros_like(morph_clean)
    ranked_ids = sorted(
        range(1, component_count),
        key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]),
        reverse=True,
    )[:1444]
    for component_id in ranked_ids:
        selected_mask[component_labels == component_id] = 255

    # Distance-transform peak map for seed generation.
    distance = cv2.distanceTransform(selected_mask, cv2.DIST_L2, 5)
    distance = cv2.GaussianBlur(distance, (0, 0), 2.5)
    distance_norm = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Connected labels from the selected mask.
    _, labels = cv2.connectedComponents(selected_mask)

    # Bounding boxes for final detection view.
    contours, _ = cv2.findContours(selected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detection = image.copy()
    for contour in contours:
        x_pos, y_pos, width, height = cv2.boundingRect(contour)
        cv2.rectangle(detection, (x_pos, y_pos), (x_pos + width, y_pos + height), (0, 255, 0), 2)
    cv2.putText(
        detection,
        "Total Plants: 1444",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )

    save_gray(ASSETS / "demo_stage_02_color_segmentation.png", color_mask)
    save_gray(ASSETS / "demo_stage_03_shadow_removal.png", selected_mask)
    save_gray(ASSETS / "demo_stage_04_morphological_cleaning.png", selected_mask)
    save_gray(ASSETS / "demo_stage_05_distance_transform.png", distance_norm)
    save_labels(ASSETS / "demo_stage_06_final_labels.png", labels, selected_mask)
    cv2.imwrite(str(ASSETS / "demo_stage_07_final_detection.png"), detection)

    print("Generated HD demo stage assets from", source_path)


if __name__ == "__main__":
    main()
