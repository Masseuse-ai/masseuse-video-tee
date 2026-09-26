"""The HUD card's picture (hud_card.py), in pixels: the wells are composited
over the ink rather than cut out of it, a tile with nothing behind it is
not drawn and the rest share the row, the card with nothing current reads
`paused` and never `live`, and no audience count is drawn wherever an
older trainer might still send one."""

from __future__ import annotations

from PIL import Image

import hud_card
from hud_card import CARD_SIZE, HudCard, MARGIN, TILE_GAP, sanitize_state

TILES = [
    {"key": "clench", "label": "Clench rate", "value": "15", "unit": "/min", "state": "live"},
    {"key": "vocal", "label": "Vocalizations", "value": "7", "unit": "/min", "detail": "+2 dB", "state": "live"},
    {"key": "heart", "label": "Heart rate", "value": "\u2014", "state": "none"},
    {"key": "posture", "label": "Posture", "value": "Settled", "state": "live"},
]
UNIT = {"name": "MK-312BT", "detail": "Waves", "tone": "live", "level": 31, "max": 70}


def picture(card: HudCard) -> Image.Image:
    return Image.frombytes("RGBA", card.size, card.render())


def test_the_wells_lighten_the_ink_instead_of_cutting_windows_through_it():
    card = HudCard()
    card.set_state({"tiles": TILES, "unit": UNIT})
    image = picture(card)
    ink_alpha = hud_card.INK[3]
    # Inside a tile's well, away from any glyph or word: the ink with a little
    # white composited over it, more opaque than the bare card, never the
    # well's own alpha of 16 on its own.
    well = image.getpixel((CARD_SIZE[0] - MARGIN - 12, CARD_SIZE[1] - MARGIN - 8))
    assert well[3] > ink_alpha, well
    assert well[3] != hud_card.WELL[3], well
    assert all(well[i] > hud_card.INK[i] for i in range(3)), "lightened, not darkened"
    # The bare card, between the ring and the first row, keeps the ink's own alpha.
    bare = image.getpixel((CARD_SIZE[0] // 2, 3))
    assert bare[3] == ink_alpha and bare[:3] == hud_card.INK[:3]


def test_a_tile_with_nothing_behind_it_is_not_drawn_and_the_rest_share_the_row():
    state = sanitize_state({"tiles": TILES, "unit": UNIT})
    drawn = HudCard.drawn_tiles(state)
    assert [tile["key"] for tile in drawn] == ["clench", "vocal", "posture"]
    assert HudCard.columns_for(3) == 3 and HudCard.columns_for(4) == 2 and HudCard.columns_for(2) == 2
    card = HudCard()
    card.set_state({"tiles": TILES, "unit": UNIT})
    image = picture(card)
    # Three wells in one row: the third column, at the right edge, is a well;
    # with two columns of two the right third of that band would be the
    # second tile's and the lower band a second row.
    width, height = CARD_SIZE
    tile_w = (width - 2 * MARGIN - TILE_GAP * 2) // 3
    ink_alpha = hud_card.INK[3]
    third_column = image.getpixel((MARGIN + 2 * (tile_w + TILE_GAP) + tile_w - 12, height - MARGIN - 8))
    assert third_column[3] > ink_alpha
    # The gap between the second and third wells is bare card: one row of three, not two of two.
    gap_x = MARGIN + 2 * (tile_w + TILE_GAP) - TILE_GAP // 2
    gap = image.getpixel((gap_x, height - MARGIN - 8))
    assert gap[3] == ink_alpha, gap
    # With all four live the card falls back to two rows of two, and the same
    # column boundary is inside the second tile's well.
    all_live = [dict(tile, state="live") for tile in TILES]
    card.set_state({"tiles": all_live, "unit": UNIT})
    assert HudCard.columns_for(len(HudCard.drawn_tiles(sanitize_state({"tiles": all_live})))) == 2
    assert picture(card).getpixel((gap_x, height - MARGIN - 8))[3] > ink_alpha
    assert card.describe()["tiles"] == 4, "describe counts the state's tiles, drawn or not"


def test_the_card_with_nothing_current_reads_paused_and_never_live():
    assert hud_card.EMPTY_WORD == "paused"
    clock = {"now": 100.0}
    card = HudCard(clock=lambda: clock["now"])
    empty = picture(card)
    # Something is written where the unit's row would be: the dim dot and the word.
    dot = empty.getpixel((MARGIN + 4, MARGIN + 11))
    assert dot[:3] == hud_card.BONE_DIM[:3]
    row = [empty.getpixel((x, MARGIN + 8)) for x in range(MARGIN + 16, MARGIN + 80)]
    assert any(px[:3] != hud_card.INK[:3] for px in row), "the word is drawn"
    # A state keeps the card current for STATE_TTL_S; past it the card is the empty one again.
    card.set_state({"tiles": TILES, "unit": UNIT})
    assert picture(card).tobytes() != empty.tobytes()
    clock["now"] += hud_card.STATE_TTL_S + 1
    assert picture(card).tobytes() == empty.tobytes()
    # The source of the old bug: no code path writes "live" into the card.
    import inspect
    assert '"live"' not in inspect.getsource(hud_card.HudCard.render)
    assert '"live"' not in inspect.getsource(hud_card.HudCard._empty)


def test_no_audience_is_drawn_whatever_an_older_trainer_sends():
    with_count = HudCard()
    with_count.set_state({"tiles": TILES, "unit": UNIT, "fans": {"watching": 12, "controlling": 3}})
    without = HudCard()
    without.set_state({"tiles": TILES, "unit": UNIT})
    assert with_count.render() == without.render()
    assert "fans" not in sanitize_state({"fans": {"watching": 1, "controlling": 1}})
    # A stale reading dims the unit's dot and the tiles rather than dropping the row.
    stale = HudCard()
    stale.set_state({"tiles": TILES, "unit": UNIT, "stale": True})
    assert picture(stale).getpixel((MARGIN + 4, MARGIN + 11))[:3] == hud_card.BONE_DIM[:3]
