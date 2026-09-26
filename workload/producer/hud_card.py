"""The HUD card drawn into the live stream (egress.py).

The trainer pushes the card's state (`PUT /overlay/hud`, its own identity,
about once a second): up to four tiles as words and the unit's row. This
module keeps the latest state and paints it as an RGBA picture the egress
feeds ffmpeg as a second input, composited over the view at the
bottom-left; the phone's own WHEP view is untouched.

The producer does not interpret the words: whatever the trainer says in
`value`, `detail` and `label` is drawn, bounded in length and count, so
this file carries no vocabulary of its own. What it draws is the layout
the phone's HUD has: a translucent dark card with a thin light ring, the
unit's name and its mode on the first row with a level bar under them,
then the tiles that have something to say (a tile whose `state` is `none`,
a reading the session has no source for, is left out), in one row of up
to three or two rows of two, each an icon glyph, a small-caps label and a
value in a larger face. The fills are composited, not written: a tile's
well lightens the card's ink under it rather than punching a window
through it. With no state, or a state older than STATE_TTL_S, the card
reads `paused`. Pillow renders it; DejaVu Sans when the image has it
(fonts-dejavu-core), Pillow's bundled face otherwise.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CARD_SIZE = (688, 232)
MARGIN = 16
TILE_GAP = 8
# What a state may carry: the trainer's words, bounded so a runaway payload
# cannot grow the card or the process.
MAX_TILES = 4
MAX_TEXT = 40
MAX_LABEL = 24
STATE_TTL_S = 15.0
# The one word of the card's own: what it says with nothing current to
# show (no state yet, or the trainer's pushes older than STATE_TTL_S).
EMPTY_WORD = "paused"
# A tile in this state has no reading behind it and is not drawn.
SKIPPED_TILE_STATE = "none"
# The palette: the phone's ink, bone and rose/mint, as RGBA.
INK = (11, 10, 16, 176)
RING = (255, 255, 255, 28)
WELL = (255, 255, 255, 16)
BONE = (244, 238, 242, 255)
BONE_DIM = (185, 173, 184, 255)
ROSE = (240, 167, 204, 255)
ROSE_DEEP = (196, 88, 158, 255)
MINT = (125, 226, 195, 255)
AMBER = (245, 194, 107, 255)
TRACK = (255, 255, 255, 40)
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
# The glyphs that stand for the tiles, by key; a key the trainer adds later
# gets a dot.
GLYPHS = {"clench": "\u223f", "vocal": "\u224b", "heart": "\u2665", "posture": "\u2020"}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A TrueType face at `size`: DejaVu when the image has it, else the
    face Pillow ships (`load_default(size)`, Pillow 10.1 on), else the
    bitmap default."""
    candidates = FONT_CANDIDATES if bold else tuple(reversed(FONT_CANDIDATES))
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _text(value, cap: int) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return text[:cap]


def sanitize_state(body: dict) -> dict:
    """The trainer's state, bounded: at most MAX_TILES tiles of bounded
    words, a unit row of bounded words with a level and a max, and whether
    the reading is stale. Anything else is left out (a `fans` count from an
    older trainer among it: the card draws no audience)."""
    if not isinstance(body, dict):
        raise ValueError("expected an object")
    tiles = []
    for tile in (body.get("tiles") or [])[:MAX_TILES]:
        if not isinstance(tile, dict):
            continue
        tiles.append({
            "key": _text(tile.get("key"), 16),
            "label": _text(tile.get("label"), MAX_LABEL),
            "value": _text(tile.get("value"), MAX_TEXT),
            "unit": _text(tile.get("unit"), 8),
            "detail": _text(tile.get("detail"), MAX_TEXT) or None,
            "state": _text(tile.get("state"), 16) or "none",
        })
    unit = body.get("unit")
    unit_row = None
    if isinstance(unit, dict):
        level = unit.get("level")
        top = unit.get("max")
        unit_row = {
            "name": _text(unit.get("name"), MAX_LABEL),
            "detail": _text(unit.get("detail"), MAX_TEXT),
            "tone": _text(unit.get("tone"), 8) or "neutral",
            "level": int(level) if isinstance(level, (int, float)) and not isinstance(level, bool) else None,
            "max": int(top) if isinstance(top, (int, float)) and not isinstance(top, bool) and top > 0 else None,
        }
    return {"tiles": tiles, "unit": unit_row, "stale": bool(body.get("stale"))}


class _Painter:
    """Two layers over the card's ink. Translucent fills (the tiles' wells,
    the level bar's track) go on a layer of their own that is composited
    over the ink, so a well lightens what is under it; Pillow's draw on an
    RGBA image would replace the ink with the well's own near-transparent
    pixels, and the video showed through. Words, dots and opaque bars are
    kept until the fills are down, then drawn on top."""

    def __init__(self, image: Image.Image):
        self.image = image
        self.fills = Image.new("RGBA", image.size, (0, 0, 0, 0))
        self._fill_draw = ImageDraw.Draw(self.fills)
        self._measure = ImageDraw.Draw(image)
        self._later: list = []

    def fill_rect(self, box, radius: int, colour) -> None:
        """A rounded rectangle: translucent ones onto the fills layer, opaque ones on top later."""
        if colour[3] < 255:
            self._fill_draw.rounded_rectangle(box, radius=radius, fill=colour)
        else:
            self._later.append(("rect", box, radius, colour))

    def ellipse(self, box, fill=None) -> None:
        self._later.append(("ellipse", box, fill))

    def text(self, xy, words: str, font=None, fill=None) -> None:
        self._later.append(("text", xy, words, font, fill))

    def textlength(self, words: str, font=None) -> float:
        return self._measure.textlength(words, font=font)

    def finish(self) -> Image.Image:
        image = Image.alpha_composite(self.image, self.fills)
        draw = ImageDraw.Draw(image)
        for op in self._later:
            if op[0] == "rect":
                draw.rounded_rectangle(op[1], radius=op[2], fill=op[3])
            elif op[0] == "ellipse":
                draw.ellipse(op[1], fill=op[2])
            else:
                draw.text(op[1], op[2], font=op[3], fill=op[4])
        return image


class HudCard:
    """The latest state and its picture. `set_state` takes the trainer's
    body; `render` paints the card (RGBA bytes of CARD_SIZE), a card that
    reads EMPTY_WORD when no state has come or the last one is older than
    STATE_TTL_S (the trainer stopped pushing: the card must not read as
    live)."""

    def __init__(self, size: tuple[int, int] = CARD_SIZE, clock=time.monotonic):
        self.size = size
        self.clock = clock
        self._lock = threading.Lock()
        self._state: dict | None = None
        self._set_at = 0.0
        self._fonts = {
            "label": _font(15, bold=True),
            "value": _font(28, bold=True),
            "word": _font(22, bold=True),
            "small": _font(15),
            "glyph": _font(20, bold=True),
        }

    def set_state(self, body: dict) -> dict:
        state = sanitize_state(body)
        with self._lock:
            self._state = state
            self._set_at = self.clock()
        return state

    def describe(self) -> dict:
        """For /statz: whether a state is held and how old it is."""
        with self._lock:
            if self._state is None:
                return {"held": False, "ageS": None}
            return {"held": True, "ageS": round(self.clock() - self._set_at, 1),
                    "tiles": len(self._state["tiles"])}

    def _current(self) -> dict | None:
        with self._lock:
            if self._state is None or self.clock() - self._set_at > STATE_TTL_S:
                return None
            return self._state

    def render(self) -> bytes:
        """The card as raw RGBA bytes, `size` wide and high."""
        width, height = self.size
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ImageDraw.Draw(image).rounded_rectangle((0, 0, width - 1, height - 1), radius=24, fill=INK, outline=RING, width=1)
        painter = _Painter(image)
        state = self._current()
        if state is None:
            self._empty(painter)
        else:
            y = self._unit_row(painter, state, MARGIN)
            self._tiles(painter, state, y)
        return painter.finish().tobytes()

    def _empty(self, draw) -> None:
        """Nothing current to show: the dim dot and EMPTY_WORD where the
        unit's row would be, never a word that reads as live."""
        draw.ellipse((MARGIN, MARGIN + 7, MARGIN + 8, MARGIN + 15), fill=BONE_DIM)
        draw.text((MARGIN + 16, MARGIN), EMPTY_WORD, font=self._fonts["label"], fill=BONE_DIM)

    def _unit_row(self, draw, state: dict, y: int) -> int:
        width = self.size[0]
        unit = state.get("unit")
        if unit is None:
            return y
        dot = BONE_DIM if state.get("stale") else {"live": MINT, "warn": AMBER}.get(unit.get("tone"), BONE_DIM)
        x = MARGIN
        draw.ellipse((x, y + 7, x + 8, y + 15), fill=dot)
        x += 16
        draw.text((x, y), unit["name"], font=self._fonts["label"], fill=BONE)
        detail = unit["detail"]
        if detail:
            right = width - MARGIN
            length = draw.textlength(detail, font=self._fonts["small"])
            draw.text((right - length, y), detail, font=self._fonts["small"], fill=BONE_DIM)
        y += 26
        if unit["level"] is not None and unit["max"]:
            track = (MARGIN, y + 3, width - MARGIN - 64, y + 7)
            draw.fill_rect(track, 2, TRACK)
            share = max(0.0, min(1.0, unit["level"] / unit["max"]))
            if share > 0:
                fill_right = track[0] + int((track[2] - track[0]) * share)
                draw.fill_rect((track[0], track[1], max(track[0] + 4, fill_right), track[3]), 2, ROSE)
            readout = f"{unit['level']}/{unit['max']}"
            length = draw.textlength(readout, font=self._fonts["small"])
            draw.text((width - MARGIN - length, y - 4), readout, font=self._fonts["small"], fill=BONE_DIM)
            y += 18
        return y + 4

    @staticmethod
    def drawn_tiles(state: dict) -> list[dict]:
        """The tiles the card draws: the state's, less the ones with nothing behind them."""
        return [tile for tile in (state.get("tiles") or []) if tile.get("state") != SKIPPED_TILE_STATE]

    @staticmethod
    def columns_for(count: int) -> int:
        """Up to three tiles share one row; four take two rows of two."""
        return count if count <= 3 else 2

    def _tiles(self, draw, state: dict, y: int) -> None:
        width, height = self.size
        tiles = self.drawn_tiles(state)
        if not tiles:
            return
        columns = self.columns_for(len(tiles))
        rows = (len(tiles) + columns - 1) // columns
        tile_w = (width - 2 * MARGIN - TILE_GAP * (columns - 1)) // columns
        tile_h = max(40, (height - y - MARGIN - TILE_GAP * (rows - 1)) // rows)
        dim_all = bool(state.get("stale"))
        for index, tile in enumerate(tiles):
            row, col = divmod(index, columns)
            x0 = MARGIN + col * (tile_w + TILE_GAP)
            y0 = y + row * (tile_h + TILE_GAP)
            draw.fill_rect((x0, y0, x0 + tile_w - 1, y0 + tile_h - 1), 14, WELL)
            dim = dim_all or tile["state"] == "stale"
            tint = MINT if tile["key"] == "posture" else ROSE
            colour = BONE_DIM if dim else tint
            glyph = GLYPHS.get(tile["key"], "\u2022")
            draw.text((x0 + 12, y0 + tile_h // 2 - 12), glyph, font=self._fonts["glyph"], fill=colour)
            tx = x0 + 42
            draw.text((tx, y0 + 8), tile["label"].upper(), font=self._fonts["label"], fill=BONE_DIM)
            if tile["state"] == "calibrating":
                draw.fill_rect((tx, y0 + 30, tx + 56, y0 + 44), 4, TRACK)
                continue
            word = tile["key"] == "posture"
            font = self._fonts["word"] if word else self._fonts["value"]
            value_y = y0 + 26 if word else y0 + 22
            draw.text((tx, value_y), tile["value"], font=font, fill=colour)
            cursor = tx + draw.textlength(tile["value"], font=font) + 4
            if tile["unit"]:
                draw.text((cursor, value_y + (8 if not word else 4)), tile["unit"], font=self._fonts["small"], fill=BONE_DIM)
                cursor += draw.textlength(tile["unit"], font=self._fonts["small"]) + 10
            if tile["detail"]:
                detail_colour = MINT if word else BONE_DIM
                draw.text((cursor, value_y + (8 if not word else 4)), tile["detail"], font=self._fonts["small"], fill=detail_colour)
