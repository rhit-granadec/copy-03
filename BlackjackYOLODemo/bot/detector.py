"""
bot/detector.py — YOLO inference wrapper with optional two-stage pipeline.

Loads the model once, runs inference on captured frames, and groups
detections into dealer-hand vs. player-hand based on spatial zones.

When pipeline mode is active, YOLO provides bounding boxes and a
fine-tuned ResNet18 classifies each crop (97.8% test accuracy).
"""

import math
from collections import defaultdict

import numpy as np
from PIL import Image
from ultralytics import YOLO
import config.settings as cfg


def _compute_iou(box_a, box_b):
    """Standard IoU between two (x1, y1, x2, y2) bboxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


class CardDetector:
    """Thin wrapper around a YOLO model for card detection."""

    def __init__(self, model_path: str = cfg.MODEL_PATH, pipeline: bool = cfg.PIPELINE_MODE, debug: bool = False):
        self.pipeline = pipeline
        self.debug = debug

        print(f"[detector] Loading YOLO model from {model_path} ...")
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # {0: '2c', 1: '2d', ...}
        print(f"[detector] Loaded — {len(self.class_names)} classes")

        if self.pipeline:
            self._load_cnn()

    def _load_cnn(self):
        """Lazily import torch and load the fine-tuned ResNet18 classifier."""
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18
        from torchvision import transforms

        self._torch = torch

        # Determine device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self._device = device

        # Rebuild the exact architecture from resnet1.ipynb
        num_classes = len(cfg.CNN_CLASS_NAMES)
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Load trained weights
        state_dict = torch.load(cfg.CNN_MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        self._cnn = model

        # ImageNet normalization (same as training)
        self._cnn_transform = transforms.Compose([
            transforms.Resize(cfg.CNN_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        print(f"[detector] Pipeline mode: YOLO + ResNet18 on {device}")
        print(f"[detector] CNN classes: {num_classes}, threshold: {cfg.CNN_CONFIDENCE_THRESHOLD}")

    @staticmethod
    def _dedup_same_class(detections: list[dict], debug: bool = False) -> list[dict]:
        """
        Three-pass dedup for multi-corner card detections.

        Pass 1 — Same-class diagonal + vertical (TL↔BR, TL↔BL, TR↔BR):
          diagonal: dist ≤ max_dist AND |dy| ≥ min_dy AND dx*dy > 0
          vertical: dist ≤ max_dist AND |dx| ≤ corner_tol AND |dy| ≥ min_dy

        Pass 2 — Same-class horizontal (TL↔TR, BL↔BR survivors):
          dist ≤ horiz_max

        Pass 3 — Cross-class corner suppression (catches corners of the
          same physical card detected as different ranks/suits, e.g.
          TL="Kh" + BR="Qd").  Uses the same diagonal/vertical geometry
          as Pass 1 plus simple proximity, applied across all classes:
          proximity: dist ≤ cross_dist
          diagonal:  dist ≤ max_dist AND |dy| ≥ min_dy AND dx*dy > 0
          vertical:  dist ≤ max_dist AND |dx| ≤ corner_tol AND |dy| ≥ min_dy

        All passes keep the higher-confidence detection.
        """
        max_dist = cfg.DEDUP_SAME_CARD_MAX_DIST
        min_dy = cfg.DEDUP_MIN_DY
        corner_tol = cfg.DEDUP_CORNER_TOLERANCE
        horiz_max = cfg.DEDUP_HORIZ_MAX_DIST

        groups: dict[str, list[int]] = defaultdict(list)
        for i, det in enumerate(detections):
            groups[det["class_name"]].append(i)

        suppressed: set[int] = set()

        for cls, indices in groups.items():
            if len(indices) < 2:
                continue
            indices.sort(key=lambda i: detections[i]["confidence"], reverse=True)

            # --- Pass 1: diagonal + vertical merges ---
            for a in range(len(indices)):
                i = indices[a]
                if i in suppressed:
                    continue
                for b in range(a + 1, len(indices)):
                    j = indices[b]
                    if j in suppressed:
                        continue
                    cx_i, cy_i = detections[i]["center"]
                    cx_j, cy_j = detections[j]["center"]
                    dx = cx_j - cx_i
                    dy = cy_j - cy_i
                    dist = math.hypot(dx, dy)
                    diagonal = dist <= max_dist and abs(dy) >= min_dy and dx * dy > 0
                    vertical = dist <= max_dist and abs(dx) <= corner_tol and abs(dy) >= min_dy
                    merge = diagonal or vertical
                    if debug:
                        reason = "diagonal" if diagonal else ("vertical" if vertical else "")
                        tag = f"MERGE-P1({reason})" if merge else "keep both"
                        print(f"[dedup] {cls} P1: dx={dx:.0f} dy={dy:.0f} "
                              f"dist={dist:.0f} → {tag}")
                    if merge:
                        suppressed.add(j)

            # --- Pass 2: horizontal merges on survivors ---
            survivors = [i for i in indices if i not in suppressed]
            for a in range(len(survivors)):
                i = survivors[a]
                if i in suppressed:
                    continue
                for b in range(a + 1, len(survivors)):
                    j = survivors[b]
                    if j in suppressed:
                        continue
                    cx_i, cy_i = detections[i]["center"]
                    cx_j, cy_j = detections[j]["center"]
                    dx = cx_j - cx_i
                    dy = cy_j - cy_i
                    dist = math.hypot(dx, dy)
                    merge = dist <= horiz_max
                    if debug:
                        tag = "MERGE-P2(horiz)" if merge else "keep both"
                        print(f"[dedup] {cls} P2: dx={dx:.0f} dy={dy:.0f} "
                              f"dist={dist:.0f} → {tag}")
                    if merge:
                        suppressed.add(j)

        # --- Pass 3: cross-class corner suppression ---
        # Two corners of the same physical card may be detected as different
        # classes (different rank and/or suit, e.g. TL="Kh" + BR="Qd").
        # Uses the same geometric checks as Pass 1 (diagonal, vertical) plus
        # simple proximity, applied across all classes.  Always keeps the
        # higher-confidence detection.
        cross_dist = getattr(cfg, "DEDUP_CROSS_CLASS_DIST", horiz_max)
        remaining = [i for i in range(len(detections)) if i not in suppressed]
        remaining.sort(key=lambda i: detections[i]["confidence"], reverse=True)
        for a in range(len(remaining)):
            i = remaining[a]
            if i in suppressed:
                continue
            for b in range(a + 1, len(remaining)):
                j = remaining[b]
                if j in suppressed:
                    continue
                cx_i, cy_i = detections[i]["center"]
                cx_j, cy_j = detections[j]["center"]
                dx = cx_j - cx_i
                dy = cy_j - cy_i
                dist = math.hypot(dx, dy)
                proximity = dist <= cross_dist
                diagonal = dist <= max_dist and abs(dy) >= min_dy and dx * dy > 0
                vertical = dist <= max_dist and abs(dx) <= corner_tol and abs(dy) >= min_dy
                merge = proximity or diagonal or vertical
                if merge:
                    if debug:
                        reason = ("proximity" if proximity else
                                  "diagonal" if diagonal else "vertical")
                        print(f"[dedup] {detections[i]['class_name']} vs "
                              f"{detections[j]['class_name']} P3: "
                              f"dx={dx:.0f} dy={dy:.0f} dist={dist:.0f}"
                              f" → MERGE-P3({reason})")
                    suppressed.add(j)

        return [d for i, d in enumerate(detections) if i not in suppressed]

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Top-level dispatcher: zone-based or full-frame detection."""
        if cfg.ZONE_DETECTION:
            return self._detect_by_zone(frame)
        return self._detect_single(frame)

    def _detect_single(self, frame: np.ndarray) -> list[dict]:
        """Dispatch to pipeline or single-stage YOLO detection."""
        return self._detect_pipeline(frame) if self.pipeline else self._detect_yolo(frame)

    def _detect_by_zone(self, frame: np.ndarray) -> list[dict]:
        """
        Tile each dealer/player zone into overlapping square crops for detection.

        Square tiles (416x416) use the full YOLO 640x640 inference space with
        minimal letterbox padding, giving ~2x more pixels per card compared to
        the previous 820x180 rectangular strips.
        """
        tile_size = cfg.ZONE_TILE_SIZE
        overlap = cfg.ZONE_TILE_OVERLAP
        stride = tile_size - overlap
        frame_h, frame_w = frame.shape[:2]

        zones = [
            ("dealer", cfg.DEALER_ZONE_Y),
            ("player", cfg.PLAYER_ZONE_Y),
        ]
        all_detections = []

        for zone_name, (y_top, y_bot) in zones:
            # --- Vertical crop: center tile_size-tall band on zone midpoint ---
            zone_mid = (y_top + y_bot) // 2
            crop_y1 = max(0, zone_mid - tile_size // 2)
            crop_y2 = crop_y1 + tile_size
            if crop_y2 > frame_h:
                crop_y2 = frame_h
                crop_y1 = max(0, crop_y2 - tile_size)

            # --- Horizontal tiles: slide across with stride ---
            x_starts = []
            x = 0
            while x + tile_size <= frame_w:
                x_starts.append(x)
                x += stride
            # Ensure the rightmost edge is covered
            if not x_starts or x_starts[-1] + tile_size < frame_w:
                x_starts.append(max(0, frame_w - tile_size))

            # --- Detect per tile, remap, and filter ---
            zone_detections = []
            for x_start in x_starts:
                tile = frame[crop_y1:crop_y2, x_start:x_start + tile_size]
                if tile.size == 0:
                    continue

                dets = self._detect_single(tile)

                for det in dets:
                    # Remap tile-relative coords to full-frame coords
                    x1, y1, x2, y2 = det["bbox"]
                    det["bbox"] = (x1 + x_start, y1 + crop_y1, x2 + x_start, y2 + crop_y1)
                    cx, cy = det["center"]
                    det["center"] = (cx + x_start, cy + crop_y1)
                    det["zone"] = zone_name

                    # Filter: keep only detections whose center Y is inside the zone
                    if y_top <= det["center"][1] <= y_bot:
                        zone_detections.append(det)

            # --- NMS dedup across overlapping tiles ---
            zone_detections.sort(key=lambda d: d["confidence"], reverse=True)
            kept = []
            for det in zone_detections:
                if not any(
                    _compute_iou(det["bbox"], k["bbox"]) > cfg.DEDUP_IOU_THRESHOLD
                    for k in kept
                ):
                    kept.append(det)

            kept = self._dedup_same_class(kept, debug=self.debug)
            all_detections.extend(kept)

        return all_detections

    def _detect_yolo(self, frame: np.ndarray) -> list[dict]:
        """
        Run single-stage YOLO inference on a BGR frame.

        Returns a list of detection dicts:
        [
            {
                "class_name": "Kh",
                "confidence": 0.93,
                "bbox": (x1, y1, x2, y2),      # pixel coords in frame
                "center": (cx, cy),
            },
            ...
        ]
        """
        floor = getattr(cfg, "CONFIDENCE_FLOOR", cfg.CONFIDENCE_THRESHOLD)
        results = self.model.predict(
            frame,
            conf=floor,
            iou=cfg.IOU_THRESHOLD,
            imgsz=cfg.YOLO_IMGSZ,
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                detections.append({
                    "class_name": self.class_names[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "center": (cx, cy),
                })

        return detections

    def _detect_pipeline(self, frame: np.ndarray) -> list[dict]:
        """
        Two-stage detection: YOLO for bounding boxes, ResNet18 for classification.

        The frame is BGR (from OpenCV). We convert to RGB once for CNN cropping.
        """
        torch = self._torch

        # Stage 1: YOLO bounding boxes (use low conf to catch more cards)
        floor = getattr(cfg, "CONFIDENCE_FLOOR", cfg.CONFIDENCE_THRESHOLD)
        results = self.model.predict(
            frame,
            conf=floor,
            iou=cfg.IOU_THRESHOLD,
            imgsz=cfg.YOLO_IMGSZ,
            verbose=False,
        )

        # Convert BGR → RGB once for all crops
        rgb_frame = frame[:, :, ::-1]

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Skip degenerate boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Stage 2: crop and classify with ResNet18
                crop = rgb_frame[y1:y2, x1:x2]
                pil_crop = Image.fromarray(crop)
                tensor = self._cnn_transform(pil_crop).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    logits = self._cnn(tensor)
                    probs = torch.softmax(logits, dim=1)
                    confidence, pred_idx = probs.max(dim=1)
                    confidence = confidence.item()
                    pred_idx = pred_idx.item()

                if confidence < cfg.CNN_CONFIDENCE_THRESHOLD:
                    continue

                class_name = cfg.CNN_CLASS_NAMES[pred_idx]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                })

        return detections

    def detect_with_masking(
        self,
        frame: np.ndarray,
        max_passes: int = cfg.MAX_DETECTION_PASSES,
        mask_padding: int = cfg.MASK_PADDING,
        dedup_iou: float = cfg.DEDUP_IOU_THRESHOLD,
    ) -> list[dict]:
        """
        Run YOLO iteratively, masking detected cards between passes to
        find overlapping cards that NMS might suppress.

        Each detection dict gets an extra ``pass_num`` key (1-based).
        """
        all_detections: list[dict] = []
        current_frame = frame.copy()

        for pass_idx in range(1, max_passes + 1):
            raw = self.detect(current_frame)

            # Tag with pass number
            for det in raw:
                det["pass_num"] = pass_idx

            # Filter duplicates against already-collected detections
            novel = []
            for det in raw:
                is_dup = any(
                    _compute_iou(det["bbox"], prev["bbox"]) > dedup_iou
                    for prev in all_detections
                )
                if not is_dup:
                    novel.append(det)

            if not novel:
                break

            all_detections.extend(novel)

            # Mask novel detections with local mean color
            h, w = current_frame.shape[:2]
            for det in novel:
                x1, y1, x2, y2 = det["bbox"]
                px1 = max(0, x1 - mask_padding)
                py1 = max(0, y1 - mask_padding)
                px2 = min(w, x2 + mask_padding)
                py2 = min(h, y2 + mask_padding)
                region = current_frame[py1:py2, px1:px2]
                if region.size > 0:
                    mean_color = region.mean(axis=(0, 1)).astype(np.uint8)
                    current_frame[py1:py2, px1:px2] = mean_color

        return self._dedup_same_class(all_detections, debug=self.debug)

    def parse_hands(self, detections: list[dict]) -> dict:
        """
        Split detections into dealer vs. player hands using Y-coordinate zones.

        Returns
        -------
        {
            "dealer": ["Kh", "5c"],      # class names only
            "player": ["7d", "9s"],
            "unknown": [...],             # cards outside both zones
            "all_detections": [...]       # full detection dicts
        }
        """
        dealer, player, unknown = [], [], []

        for det in detections:
            zone = det.get("zone")
            if zone == "dealer":
                dealer.append(det["class_name"])
            elif zone == "player":
                player.append(det["class_name"])
            else:
                # Fallback to Y-coordinate heuristic (full-frame mode)
                cy = det["center"][1]
                if cfg.DEALER_ZONE_Y[0] <= cy <= cfg.DEALER_ZONE_Y[1]:
                    dealer.append(det["class_name"])
                elif cfg.PLAYER_ZONE_Y[0] <= cy <= cfg.PLAYER_ZONE_Y[1]:
                    player.append(det["class_name"])
                else:
                    unknown.append(det["class_name"])

        return {
            "dealer": dealer,
            "player": player,
            "unknown": unknown,
            "all_detections": detections,
        }
