"""
app.py — Flask backend for the web-based Blackjack Bot.

Serves the game UI and exposes an API for YOLO card detection,
basic strategy lookup, and Hi-Lo card counting.
"""

import argparse
import base64
import io
import json
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
import config.settings as settings
from PIL import Image

from bot.detector import CardDetector
from bot.strategy import HiLoCounter, basic_strategy, hand_total

parser = argparse.ArgumentParser(description="Blackjack Bot server")
parser.add_argument("--pipeline", action="store_true",
                    help="Use YOLO+ResNet18 two-stage pipeline for better accuracy")
parser.add_argument("--debug", action="store_true",
                    help="Save detection images to debug/ folder for inspection")
args = parser.parse_args()

if args.debug:
    import os, time as _time
    DEBUG_DIR = "debug_frames"
    os.makedirs(DEBUG_DIR, exist_ok=True)
    print(f"[app] Debug mode enabled — saving detection images to {DEBUG_DIR}/", flush=True)

app = Flask(__name__, static_folder="game")

# ── Load model and state at startup ──────────────────────────────────────────
detector = CardDetector(pipeline=args.pipeline, debug=args.debug)
counter = HiLoCounter()
prev_seen: set[str] = set()  # cards seen this hand (avoid double-counting)
_last_raw_detections: list[dict] = []  # unfiltered detections for P-R logging


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the blackjack game page."""
    return send_from_directory("game", "blackjack.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    """
    Receive a base64 PNG screenshot, run YOLO detection, compute strategy.

    Request JSON: { "image": "<base64-encoded PNG>" }
    Response JSON: {
        "detections": [{ "class_name", "confidence", "bbox", "zone" }, ...],
        "dealer_cards": [...], "player_cards": [...],
        "dealer_total": int, "player_total": int,
        "action": str,
        "running_count": int, "true_count": float, "cards_seen": int
    }
    """
    data = request.get_json(force=True)
    image_b64 = data.get("image", "")

    # Strip data-URI prefix if present
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    # Decode base64 → PIL → BGR numpy array
    raw = base64.b64decode(image_b64)
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Run detection (YOLO or pipeline) with iterative masking
    # Returns all detections above CONFIDENCE_FLOOR for P-R logging
    all_detections = detector.detect_with_masking(frame)

    # Post-filter at the live threshold for game decisions
    conf_thresh = settings.CONFIDENCE_THRESHOLD
    detections = [d for d in all_detections if d["confidence"] >= conf_thresh]
    hands = detector.parse_hands(detections)

    # Stash unfiltered detections for log_performance P-R curve data
    global _last_raw_detections
    _last_raw_detections = all_detections

    # Log detections grouped by zone (always)
    h, w = frame.shape[:2]
    by_zone: dict[str, list] = {"dealer": [], "player": [], "unknown": []}
    for det in detections:
        z = det.get("zone", "unknown")
        by_zone.setdefault(z, []).append(det)

    for zone_name in ("dealer", "player", "unknown"):
        zone_dets = by_zone.get(zone_name, [])
        if zone_dets:
            cards_str = ", ".join(
                f"{d['class_name']} {d['confidence']:.2f}" for d in zone_dets
            )
            print(f"[detect] {zone_name.upper()} ({len(zone_dets)}): {cards_str}", flush=True)

    if not detections:
        print("[detect] No detections", flush=True)

    # Debug: save frame and zone tile images to disk
    if args.debug:
        ts = int(_time.time() * 1000)

        # Save full frame
        debug_path = f"{DEBUG_DIR}/frame_{ts}.png"
        saved = cv2.imwrite(debug_path, frame)
        print(f"[debug] {'Saved' if saved else 'FAILED to save'} {debug_path} | {w}x{h} | {len(detections)} detections", flush=True)

        # Save zone tiles (the actual square crops fed to YOLO)
        import config.settings as _cfg
        tile_size = _cfg.ZONE_TILE_SIZE
        overlap = _cfg.ZONE_TILE_OVERLAP
        stride = tile_size - overlap
        for zone_name, (y_top, y_bot) in [("dealer", _cfg.DEALER_ZONE_Y), ("player", _cfg.PLAYER_ZONE_Y)]:
            zone_mid = (y_top + y_bot) // 2
            crop_y1 = max(0, zone_mid - tile_size // 2)
            crop_y2 = crop_y1 + tile_size
            if crop_y2 > h:
                crop_y2 = h
                crop_y1 = max(0, crop_y2 - tile_size)

            x_starts = []
            x = 0
            while x + tile_size <= w:
                x_starts.append(x)
                x += stride
            if not x_starts or x_starts[-1] + tile_size < w:
                x_starts.append(max(0, w - tile_size))

            for i, x_start in enumerate(x_starts):
                tile = frame[crop_y1:crop_y2, x_start:x_start + tile_size]
                tile_path = f"{DEBUG_DIR}/frame_{ts}_{zone_name}_tile{i}.png"
                th, tw = tile.shape[:2]
                cv2.imwrite(tile_path, tile)
                print(f"[debug] Saved {tile_path} ({tw}x{th})", flush=True)

    # Update Hi-Lo count with newly seen cards
    global prev_seen
    all_cards = hands["dealer"] + hands["player"]
    new_cards = [c for c in all_cards if c not in prev_seen]
    if new_cards:
        counter.update(new_cards)
        prev_seen.update(new_cards)

    # Compute strategy
    action = "none"
    dealer_total_val = 0
    player_total_val = 0

    if hands["player"]:
        player_total_val, _ = hand_total(hands["player"])

    if hands["dealer"]:
        dealer_total_val, _ = hand_total(hands["dealer"])

    if hands["player"] and hands["dealer"]:
        can_double = len(hands["player"]) == 2
        action = basic_strategy(
            hands["player"],
            hands["dealer"][0],  # upcard
            can_double=can_double,
            can_split=can_double,
        )

    # Build response detections with zone info and pass number
    det_response = []
    total_passes = 0
    for det in detections:
        zone = det.get("zone", "unknown")
        pass_num = det.get("pass_num", 1)
        total_passes = max(total_passes, pass_num)
        det_response.append({
            "class_name": det["class_name"],
            "confidence": round(det["confidence"], 2),
            "bbox": det["bbox"],
            "zone": zone,
            "pass_num": pass_num,
        })

    return jsonify({
        "detections": det_response,
        "dealer_cards": hands["dealer"],
        "player_cards": hands["player"],
        "dealer_total": dealer_total_val,
        "player_total": player_total_val,
        "action": action,
        "running_count": counter.running_count,
        "true_count": round(counter.true_count, 1),
        "cards_seen": counter.cards_seen,
        "total_passes": total_passes,
        "suggested_bet_units": counter.suggested_bet_units,
    })


@app.route("/api/strategy", methods=["POST"])
def strategy():
    """
    Receive card names directly from JS game state, return strategy action.

    Request JSON: { "player_cards": ["Ks","7h"], "dealer_upcard": "10c", "can_double": true }
    """
    data = request.get_json(force=True)
    player_cards = data["player_cards"]
    dealer_upcard = data["dealer_upcard"]
    can_double = data.get("can_double", True)

    # Update count with newly seen cards
    global prev_seen
    all_cards = player_cards + [dealer_upcard]
    new_cards = [c for c in all_cards if c not in prev_seen]
    if new_cards:
        counter.update(new_cards)
        prev_seen.update(new_cards)

    p_total, _ = hand_total(player_cards)
    d_total, _ = hand_total([dealer_upcard])

    action = basic_strategy(
        player_cards, dealer_upcard,
        can_double=can_double, can_split=can_double,
    )

    return jsonify({
        "action": action,
        "player_total": p_total,
        "dealer_total": d_total,
        "running_count": counter.running_count,
        "true_count": round(counter.true_count, 1),
        "cards_seen": counter.cards_seen,
        "suggested_bet_units": counter.suggested_bet_units,
    })


@app.route("/api/new_hand", methods=["POST"])
def new_hand():
    """Clear prev_seen for a new hand (keep the running count)."""
    global prev_seen
    prev_seen = set()
    return jsonify({"status": "ok"})


@app.route("/api/reset_count", methods=["POST"])
def reset_count():
    """Reset the Hi-Lo counter for a new shoe."""
    global prev_seen
    counter.reset()
    prev_seen = set()
    return jsonify({"status": "ok", "running_count": 0, "true_count": 0.0})


# ── Performance tracking ─────────────────────────────────────────────────────

PERF_DIR = os.path.join(os.path.dirname(__file__) or ".", "performance")
PERF_FILE = os.path.join(PERF_DIR, "card_stats.json")
PR_LOG_FILE = os.path.join(PERF_DIR, "pr_raw.jsonl")

os.makedirs(PERF_DIR, exist_ok=True)


def _load_perf_stats() -> dict:
    """Load accumulated per-card-type stats from disk."""
    if os.path.exists(PERF_FILE):
        with open(PERF_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_perf_stats(stats: dict):
    """Write accumulated stats to disk (atomic via temp file)."""
    tmp = PERF_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    os.replace(tmp, PERF_FILE)


@app.route("/api/log_performance", methods=["POST"])
def log_performance():
    """
    Receive actual vs. detected cards for a completed hand and accumulate
    per-card-type detection stats into performance/card_stats.json.

    Request JSON: {
        "actual_dealer": ["Kh", "5c"],
        "actual_player": ["7d", "9s", "3h"],
        "detected_dealer": ["Kh"],
        "detected_player": ["7d", "9s"]
    }

    For each actual card, records whether it was detected or missed.
    Also tracks false positives per card type.
    """
    data = request.get_json(force=True)
    actual = data.get("actual_dealer", []) + data.get("actual_player", [])
    detected = data.get("detected_dealer", []) + data.get("detected_player", [])

    stats = _load_perf_stats()

    # Match detected against actual (same logic as JS compareCards)
    det_pool = list(detected)
    for card in actual:
        entry = stats.setdefault(card, {"appeared": 0, "detected": 0, "missed": 0, "false_positive": 0})
        entry["appeared"] += 1
        idx = None
        for i, d in enumerate(det_pool):
            if d == card:
                idx = i
                break
        if idx is not None:
            entry["detected"] += 1
            det_pool.pop(idx)
        else:
            entry["missed"] += 1

    # Remaining in det_pool are false positives
    for card in det_pool:
        entry = stats.setdefault(card, {"appeared": 0, "detected": 0, "missed": 0, "false_positive": 0})
        entry["false_positive"] += 1

    _save_perf_stats(stats)

    # Append raw detections (all confidence levels) for P-R curve analysis
    global _last_raw_detections
    pr_record = {
        "actual": actual,
        "raw_detections": [
            {"class_name": d["class_name"], "confidence": round(d["confidence"], 4),
             "zone": d.get("zone", "unknown")}
            for d in _last_raw_detections
        ],
    }
    with open(PR_LOG_FILE, "a") as f:
        f.write(json.dumps(pr_record) + "\n")

    # Compute summary for response
    total_appeared = sum(e["appeared"] for e in stats.values())
    total_detected = sum(e["detected"] for e in stats.values())
    rate = round(total_detected / total_appeared * 100, 1) if total_appeared else 0

    print(f"[perf] Hand logged: {len(actual)} actual, {len(detected)} detected | "
          f"Cumulative: {total_detected}/{total_appeared} ({rate}%)", flush=True)

    return jsonify({"status": "ok", "cumulative_rate": rate})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
