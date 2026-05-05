# BlackjackYOLO — Real-Time Card Detection & Strategy Bot

A real-time blackjack assistant that uses YOLO object detection to identify cards
from a browser-based blackjack game, then recommends optimal plays using Basic
Strategy and Hi-Lo card counting. The bot runs as a Flask web server with a
self-hosted HTML5 blackjack UI.

## Architecture

```
Browser (blackjack.html)
    │
    │  base64 screenshot via JS
    ▼
Flask API (app.py :5001)
    │
    ├─► /api/detect ──► bot/detector.py ──► YOLO (+ optional ResNet18)
    │                         │
    │                         ▼
    │                   bot/strategy.py ──► Basic Strategy + Hi-Lo count
    │
    ├─► /api/strategy ──► Direct card-name lookup (no detection needed)
    ├─► /api/new_hand  ──► Reset per-hand state
    ├─► /api/reset_count ──► Reset Hi-Lo counter for new shoe
    └─► /api/log_performance ──► Record detection accuracy to performance/
```

## Project Structure

```
BlackjackYOLO/
├── app.py                          # Flask server — main entry point
├── requirements.txt                # pip dependencies
├── config/
│   └── settings.py                 # All tunable parameters (model paths,
│                                   #   thresholds, zones, detection tuning)
├── bot/
│   ├── detector.py                 # YOLO inference wrapper + optional
│   │                               #   two-stage YOLO→ResNet18 pipeline
│   └── strategy.py                 # Basic Strategy tables + Hi-Lo counting
├── game/
│   ├── blackjack.html              # Self-hosted HTML5 blackjack game
│   ├── cards/                      # Card images (current art)
│   └── cards_old/                  # Card images (alternate art)
├── models/
│   ├── best.pt                     # YOLO detection model (primary)
│   ├── cnn_resnet18_224x224.pt     # ResNet18 classifier (pipeline mode)
│   ├── best_old.pt                 # Previous YOLO version
│   ├── best_46.pt                  # Alternative YOLO checkpoint
│   └── cnn_resnet18_224x224_old.pt # Previous ResNet18 version
├── performance/
│   ├── generate_pr_curve.py        # Precision-Recall curve generator
│   ├── pr_curve.png                # Generated P-R plot
│   ├── pr_curve_yolo.png           # YOLO-only P-R plot
│   └── card_stats_*.json           # Per-card detection accuracy snapshots
├── blackjack_yolo_training.ipynb   # YOLO model training notebook
├── YOLO_model_2 (1).ipynb          # Extended YOLO training notebook
├── pipeline.ipynb                  # Two-stage pipeline training notebook
└── resnet1.ipynb                   # ResNet18 fine-tuning notebook
```

## Setup

Requires **Python 3.9+**.

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
```

## Running

### Main Application

```bash
python app.py
```

Starts the Flask server on `http://localhost:5001`. Open that URL in your browser
to load the blackjack game with integrated card detection.

### Demo URL Parameters (`http://localhost:5001/?...`)

The front-end reads query parameters from `window.location.search`.

| Parameter | Example | Behavior |
|------|-------------|-------------|
| `seed` | `?seed=42` | Sets deterministic deck shuffle/order using a seeded PRNG. The specific seed used for the project demo run was **42**. |
| `speed` | `?speed=1`, `?speed=0.5`, `?speed=2` | Sets initial bot speed multiplier. Value is clamped to `[0.25, 10]`. |
| `demo` | `?demo=false` | Disables presentation autoplay demo mode (stop-conditions and demo flow). Any value other than `false` leaves demo mode enabled. |

Examples:

```text
http://localhost:5001/?seed=42
http://localhost:5001/?seed=42&demo=false
http://localhost:5001/?seed=42&speed=1&demo=false
```

Any other query parameters are currently ignored by the game.

**Flags:**

| Flag | Description |
|------|-------------|
| `--pipeline` | Use YOLO + ResNet18 two-stage detection (more accurate) |
| `--debug` | Save detection frames to `debug_frames/` for inspection |

```bash
python app.py --pipeline          # two-stage mode
python app.py --debug             # save debug frames
python app.py --pipeline --debug  # both
```

### Precision-Recall Analysis

```bash
python performance/generate_pr_curve.py
```

Reads `performance/pr_raw.jsonl` (generated during bot runs via the
`/api/log_performance` endpoint), sweeps confidence thresholds, and outputs a
P-R curve plot to `performance/pr_curve.png`.

## API Reference

All endpoints accept and return JSON. The server runs on port **5001**.

### `GET /`

Serves the blackjack game UI (`game/blackjack.html`).

---

### `POST /api/detect`

Run YOLO card detection on a screenshot.

**Request:**
```json
{
  "image": "<base64-encoded PNG (data URI prefix optional)>"
}
```

**Response:**
```json
{
  "detections": [
    {
      "class_name": "Kh",
      "confidence": 0.93,
      "bbox": [x1, y1, x2, y2],
      "zone": "dealer",
      "pass_num": 1
    }
  ],
  "dealer_cards": ["Kh", "5c"],
  "player_cards": ["7d", "9s"],
  "dealer_total": 15,
  "player_total": 16,
  "action": "stand",
  "running_count": 2,
  "true_count": 2.0,
  "cards_seen": 8,
  "total_passes": 2,
  "suggested_bet_units": 2
}
```

**Detection pipeline:**
1. Crops dealer and player zones into overlapping 416x416 tiles
2. Runs YOLO on each tile (or YOLO + ResNet18 in `--pipeline` mode)
3. Remaps tile coordinates to full-frame coordinates
4. Deduplicates across tiles and multi-corner detections (3-pass dedup)
5. Iterative masking: masks detected cards and re-runs YOLO to find
   overlapping cards (up to 3 passes)

---

### `POST /api/strategy`

Look up the optimal action from known card names (no detection needed).

**Request:**
```json
{
  "player_cards": ["Ks", "7h"],
  "dealer_upcard": "10c",
  "can_double": true
}
```

**Response:**
```json
{
  "action": "stand",
  "player_total": 17,
  "dealer_total": 10,
  "running_count": -1,
  "true_count": -1.0,
  "cards_seen": 3,
  "suggested_bet_units": 1
}
```

---

### `POST /api/new_hand`

Clear per-hand card tracking (keeps the running count).

**Response:** `{ "status": "ok" }`

---

### `POST /api/reset_count`

Reset the Hi-Lo counter for a new shoe.

**Response:** `{ "status": "ok", "running_count": 0, "true_count": 0.0 }`

---

### `POST /api/log_performance`

Record actual vs. detected cards for accuracy tracking.

**Request:**
```json
{
  "actual_dealer": ["Kh", "5c"],
  "actual_player": ["7d", "9s", "3h"],
  "detected_dealer": ["Kh"],
  "detected_player": ["7d", "9s"]
}
```

**Response:** `{ "status": "ok", "cumulative_rate": 85.7 }`

Appends raw detection data to `performance/pr_raw.jsonl` and updates
cumulative per-card stats in `performance/card_stats.json`.

## Detection Modes

| Mode | Command | How it works |
|------|---------|-------------|
| **Single-stage** | `python app.py` | YOLO only — faster |
| **Two-stage pipeline** | `python app.py --pipeline` | YOLO for bounding boxes, then ResNet18 classifies each crop (97.8% test accuracy) |

Both modes use zone-based tiling (dealer zone y:20-200, player zone y:320-500)
with iterative masking to detect overlapping cards.

## Configuration

All tunable parameters live in `config/settings.py`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `MODEL_PATH` | `models/best.pt` | YOLO weights path |
| `CONFIDENCE_THRESHOLD` | 0.16 | Minimum confidence for game decisions |
| `CONFIDENCE_FLOOR` | 0.05 | Lowest confidence kept for P-R logging |
| `IOU_THRESHOLD` | 0.40 | NMS IoU threshold |
| `DEALER_ZONE_Y` | (20, 200) | Dealer card region (pixel y-range) |
| `PLAYER_ZONE_Y` | (320, 500) | Player card region (pixel y-range) |
| `MAX_DETECTION_PASSES` | 3 | Iterative masking passes |
| `PIPELINE_MODE` | False | Default two-stage toggle |
| `CNN_MODEL_PATH` | `models/cnn_resnet18_224x224.pt` | ResNet18 weights |
| `YOLO_IMGSZ` | 640 | YOLO inference resolution |
| `ZONE_TILE_SIZE` | 416 | Square tile side for zone detection |

## YOLO Class Mapping

The detector expects 52 class names in the format `<rank><suit>`:
- Ranks: `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `J`, `Q`, `K`, `A`
- Suits: `c` (clubs), `d` (diamonds), `h` (hearts), `s` (spades)
- Examples: `2c`, `10h`, `Ks`, `Ad`

## Dependencies

| Package | Purpose |
|---------|---------|
| ultralytics | YOLO model loading and inference |
| opencv-python | Image processing (BGR conversion, frame annotation) |
| numpy | Array operations |
| flask | Web server and API |
| Pillow | Image decoding (base64 to PIL) |
| torch | Deep learning runtime (ResNet18 pipeline, YOLO backend) |
| torchvision | ResNet18 model architecture and transforms |
| matplotlib | *(optional)* Plotting for P-R curve generation |
