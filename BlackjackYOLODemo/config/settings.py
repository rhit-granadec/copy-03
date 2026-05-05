"""
config/settings.py — All tunable parameters in one place.
"""

# ---------------------------------------------------------------------------
# YOLO Model
# ---------------------------------------------------------------------------
MODEL_PATH = "models/best.pt"    # Path to your trained YOLO weights
CONFIDENCE_THRESHOLD = 0.16      # Minimum detection confidence to accept
CONFIDENCE_FLOOR = 0.05          # Lowest confidence kept for P-R curve logging
IOU_THRESHOLD = 0.40             # NMS IoU threshold

# ---------------------------------------------------------------------------
# Spatial Zones — dividing dealer vs. player cards
# ---------------------------------------------------------------------------
# These are relative to the #table div captured by html2canvas (820x520).
DEALER_ZONE_Y = (20, 200)       # (top, bottom) of dealer card region
PLAYER_ZONE_Y = (320, 500)      # (top, bottom) of player card region

# ---------------------------------------------------------------------------
# Card Class Names
# ---------------------------------------------------------------------------
# Must match the class names your YOLO model was trained with.
# Format: "<rank><suit>" where suit is one of c, d, h, s
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["c", "d", "h", "s"]   # clubs, diamonds, hearts, spades

CLASS_NAMES = [f"{r}{s}" for r in RANKS for s in SUITS]  # 52 classes

# ---------------------------------------------------------------------------
# Hi-Lo Counting
# ---------------------------------------------------------------------------
DECKS_IN_SHOE = 1               # Single deck for faster convergence in demo

# ---------------------------------------------------------------------------
# Iterative Masking Detection
# ---------------------------------------------------------------------------
MAX_DETECTION_PASSES = 3        # Max YOLO passes with masking
MASK_PADDING = 5                # Pixels to pad around bbox when masking
DEDUP_IOU_THRESHOLD = 0.50      # IoU threshold for duplicate filtering
DEDUP_SAME_CARD_MAX_DIST = 150  # Max Euclidean distance (px) between same-class centers
DEDUP_MIN_DY = 30               # Min |dy| to consider two detections as card corners
DEDUP_CORNER_TOLERANCE = 25     # Max |dx| for vertical corner pair (TL+BL, TR+BR)
DEDUP_HORIZ_MAX_DIST = 70       # Pass-2 radius for horizontal corner pairs (TL+TR, BL+BR)
DEDUP_CROSS_CLASS_DIST = 100    # Pass-3 radius for cross-class corner suppression

# ---------------------------------------------------------------------------
# Two-Stage Pipeline (YOLO + ResNet18)
# ---------------------------------------------------------------------------
PIPELINE_MODE = False
CNN_MODEL_PATH = "models/cnn_resnet18_224x224.pt"
CNN_INPUT_SIZE = (224, 224)
CNN_CONFIDENCE_THRESHOLD = 0.30
CNN_CLASS_NAMES = sorted(CLASS_NAMES)  # alphabetical — matches ImageFolder training order

# ---------------------------------------------------------------------------
# Detection Tuning
# ---------------------------------------------------------------------------
YOLO_IMGSZ = 640                # Inference resolution (higher = more detail for small cards)
ZONE_DETECTION = True           # Crop dealer/player zones and detect separately
ZONE_TILE_SIZE = 416            # Square tile side length (416 = model training imgsz)
ZONE_TILE_OVERLAP = 100         # Horizontal overlap between tiles (>= card width ~80px)
