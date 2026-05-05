"""
bot/strategy.py — Basic Strategy lookup and Hi-Lo card counting.

Basic Strategy table is the mathematically optimal play for each combination
of (player_total, dealer_upcard).  Hi-Lo overlays a running/true count for
bet sizing.
"""

from config.settings import DECKS_IN_SHOE

# ── Card value helpers ──────────────────────────────────────────────────────

def card_rank(class_name: str) -> str:
    """Extract rank from a YOLO class name like '10h' → '10' or 'Ks' → 'K'."""
    return class_name[:-1]


def card_value(class_name: str) -> int:
    """Blackjack point value of a card.  Ace = 11 (handled as soft later)."""
    rank = card_rank(class_name)
    if rank in ("J", "Q", "K"):
        return 10
    if rank == "A":
        return 11
    return int(rank)


def hand_total(cards: list[str]) -> tuple[int, bool]:
    """
    Compute the best blackjack total for a list of card class-names.

    Returns (total, is_soft).
    is_soft means the hand contains an Ace counted as 11.
    """
    total = sum(card_value(c) for c in cards)
    aces = sum(1 for c in cards if card_rank(c) == "A")

    # Reduce aces from 11 → 1 until we're at 21 or below
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1

    return total, aces > 0


def is_pair(cards: list[str]) -> bool:
    """True if the hand is exactly two cards of the same rank."""
    if len(cards) != 2:
        return False
    return card_value(cards[0]) == card_value(cards[1])


# ── Basic Strategy Tables ───────────────────────────────────────────────────
# Keys:   dealer upcard value (2–11, where 11 = Ace)
# Values: dict mapping player total → action string
#
# Actions: "H" = Hit, "S" = Stand, "D" = Double (hit if not allowed),
#          "P" = Split, "Ds" = Double/Stand
#
# Three tables: hard totals, soft totals, and pairs.

# Hard totals (player_total → {dealer_upcard → action})
_HARD = {
    5:  {2:"H",3:"H",4:"H",5:"H",6:"H",7:"H",8:"H",9:"H",10:"H",11:"H"},
    6:  {2:"H",3:"H",4:"H",5:"H",6:"H",7:"H",8:"H",9:"H",10:"H",11:"H"},
    7:  {2:"H",3:"H",4:"H",5:"H",6:"H",7:"H",8:"H",9:"H",10:"H",11:"H"},
    8:  {2:"H",3:"H",4:"H",5:"H",6:"H",7:"H",8:"H",9:"H",10:"H",11:"H"},
    9:  {2:"H",3:"D",4:"D",5:"D",6:"D",7:"H",8:"H",9:"H",10:"H",11:"H"},
    10: {2:"D",3:"D",4:"D",5:"D",6:"D",7:"D",8:"D",9:"D",10:"H",11:"H"},
    11: {2:"D",3:"D",4:"D",5:"D",6:"D",7:"D",8:"D",9:"D",10:"D",11:"D"},
    12: {2:"H",3:"H",4:"S",5:"S",6:"S",7:"H",8:"H",9:"H",10:"H",11:"H"},
    13: {2:"S",3:"S",4:"S",5:"S",6:"S",7:"H",8:"H",9:"H",10:"H",11:"H"},
    14: {2:"S",3:"S",4:"S",5:"S",6:"S",7:"H",8:"H",9:"H",10:"H",11:"H"},
    15: {2:"S",3:"S",4:"S",5:"S",6:"S",7:"H",8:"H",9:"H",10:"H",11:"H"},
    16: {2:"S",3:"S",4:"S",5:"S",6:"S",7:"H",8:"H",9:"H",10:"H",11:"H"},
    17: {2:"S",3:"S",4:"S",5:"S",6:"S",7:"S",8:"S",9:"S",10:"S",11:"S"},
    18: {2:"S",3:"S",4:"S",5:"S",6:"S",7:"S",8:"S",9:"S",10:"S",11:"S"},
    19: {2:"S",3:"S",4:"S",5:"S",6:"S",7:"S",8:"S",9:"S",10:"S",11:"S"},
    20: {2:"S",3:"S",4:"S",5:"S",6:"S",7:"S",8:"S",9:"S",10:"S",11:"S"},
    21: {2:"S",3:"S",4:"S",5:"S",6:"S",7:"S",8:"S",9:"S",10:"S",11:"S"},
}

# Soft totals (player_total with Ace=11)
_SOFT = {
    13: {2:"H",3:"H",4:"H",5:"D",6:"D",7:"H",8:"H",9:"H",10:"H",11:"H"},
    14: {2:"H",3:"H",4:"H",5:"D",6:"D",7:"H",8:"H",9:"H",10:"H",11:"H"},
    15: {2:"H",3:"H",4:"D",5:"D",6:"D",7:"H",8:"H",9:"H",10:"H",11:"H"},
    16: {2:"H",3:"H",4:"D",5:"D",6:"D",7:"H",8:"H",9:"H",10:"H",11:"H"},
    17: {2:"H",3:"D",4:"D",5:"D",6:"D",7:"H",8:"H",9:"H",10:"H",11:"H"},
    18: {2:"Ds",3:"Ds",4:"Ds",5:"Ds",6:"Ds",7:"S",8:"S",9:"H",10:"H",11:"H"},
    19: {2:"S",3:"S",4:"S",5:"S",6:"Ds",7:"S",8:"S",9:"S",10:"S",11:"S"},
    20: {2:"S",3:"S",4:"S",5:"S",6:"S",7:"S",8:"S",9:"S",10:"S",11:"S"},
    21: {2:"S",3:"S",4:"S",5:"S",6:"S",7:"S",8:"S",9:"S",10:"S",11:"S"},
}

# Pair splitting (card value of the pair → {dealer_upcard → action})
_PAIRS = {
    2:  {2:"P",3:"P",4:"P",5:"P",6:"P",7:"P",8:"H",9:"H",10:"H",11:"H"},
    3:  {2:"P",3:"P",4:"P",5:"P",6:"P",7:"P",8:"H",9:"H",10:"H",11:"H"},
    4:  {2:"H",3:"H",4:"H",5:"P",6:"P",7:"H",8:"H",9:"H",10:"H",11:"H"},
    5:  {2:"D",3:"D",4:"D",5:"D",6:"D",7:"D",8:"D",9:"D",10:"H",11:"H"},
    6:  {2:"P",3:"P",4:"P",5:"P",6:"P",7:"H",8:"H",9:"H",10:"H",11:"H"},
    7:  {2:"P",3:"P",4:"P",5:"P",6:"P",7:"P",8:"H",9:"H",10:"H",11:"H"},
    8:  {2:"P",3:"P",4:"P",5:"P",6:"P",7:"P",8:"P",9:"P",10:"P",11:"P"},
    9:  {2:"P",3:"P",4:"P",5:"P",6:"P",7:"S",8:"P",9:"P",10:"S",11:"S"},
    10: {2:"S",3:"S",4:"S",5:"S",6:"S",7:"S",8:"S",9:"S",10:"S",11:"S"},
    11: {2:"P",3:"P",4:"P",5:"P",6:"P",7:"P",8:"P",9:"P",10:"P",11:"P"},  # Aces
}


def basic_strategy(player_cards: list[str], dealer_upcard: str,
                    can_double: bool = True, can_split: bool = True) -> str:
    """
    Look up the Basic Strategy action.

    Parameters
    ----------
    player_cards : list of YOLO class names, e.g. ["7h", "9s"]
    dealer_upcard : single YOLO class name, e.g. "Kc"
    can_double : whether doubling is allowed at this point
    can_split : whether splitting is allowed at this point

    Returns
    -------
    One of: "hit", "stand", "double", "split"
    """
    dealer_val = card_value(dealer_upcard)
    if dealer_val == 1:          # if Ace slipped through as 1
        dealer_val = 11

    p_total, soft = hand_total(player_cards)

    # ── Pairs ───────────────────────────────────────────────
    if can_split and is_pair(player_cards):
        pair_val = card_value(player_cards[0])
        if pair_val == 11:
            pair_val = 11       # Aces stored as 11
        action = _PAIRS.get(pair_val, {}).get(dealer_val, "H")
        if action == "P":
            return "split"

    # ── Soft hands ──────────────────────────────────────────
    if soft and p_total <= 21:
        action = _SOFT.get(p_total, {}).get(dealer_val, "H")
    else:
        # Clamp to table range
        clamped = max(5, min(21, p_total))
        action = _HARD.get(clamped, {}).get(dealer_val, "H")

    # ── Resolve conditional actions ─────────────────────────
    if action == "D" and not can_double:
        action = "H"
    if action == "Ds" and not can_double:
        action = "S"
    if action == "Ds":
        action = "D"

    return {"H": "hit", "S": "stand", "D": "double", "P": "split"}[action]


# ── Hi-Lo Card Counting ────────────────────────────────────────────────────

class HiLoCounter:
    """
    Maintains a running count and computes the true count.

    Hi-Lo values:  2-6 → +1,  7-9 → 0,  10/J/Q/K/A → -1
    """

    _HILO = {
        "2": 1, "3": 1, "4": 1, "5": 1, "6": 1,
        "7": 0, "8": 0, "9": 0,
        "10": -1, "J": -1, "Q": -1, "K": -1, "A": -1,
    }

    def __init__(self, num_decks: int = DECKS_IN_SHOE):
        self.num_decks = num_decks
        self.running_count = 0
        self.cards_seen = 0

    def update(self, card_class_names: list[str]):
        """Feed newly-visible cards into the counter."""
        for name in card_class_names:
            rank = card_rank(name)
            self.running_count += self._HILO.get(rank, 0)
            self.cards_seen += 1

    @property
    def true_count(self) -> float:
        """Running count ÷ estimated remaining decks."""
        total_cards = self.num_decks * 52
        remaining = max(total_cards - self.cards_seen, 26)  # floor at half-deck
        decks_remaining = remaining / 52
        return self.running_count / decks_remaining

    @property
    def suggested_bet_units(self) -> int:
        """Simple bet ramp: 1 unit base, +1 per true count above +1."""
        tc = int(self.true_count)
        return max(1, tc)

    def reset(self):
        """Reset for a new shoe."""
        self.running_count = 0
        self.cards_seen = 0

    def __repr__(self):
        return (f"HiLo(running={self.running_count}, "
                f"true={self.true_count:.1f}, "
                f"seen={self.cards_seen})")
