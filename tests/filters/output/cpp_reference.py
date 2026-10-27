"""Step-by-step transcription of upstream ScanTailor Advanced 1.2.1 edge smoothing.

This is the oracle for the tests: it follows ``hitMissMatch`` and
``hitMissReplaceInPlace`` in ``Morphology.cpp``, and ``hitMissReplaceAllDirections``
and ``morphologicalSmoothInPlace`` in ``OutputGenerator.cpp``, with explicit
loops, and shares no code with the port. Black pixels are True. It is slow; use
small images.
"""

import numpy as np

# morphologicalSmoothInPlace: (pattern, width, height), in its order.
SMOOTHING_PATTERNS = (
    ("XXX -    ", 3, 3),
    ("X ?X  X- X- X  X ?", 3, 6),
    ("X ?X ?X  X- X- X- X  X ?X ?", 3, 9),
    ("XX?XX?XX X+ X+ X+ XX XX?XX?", 3, 9),
    ("XX?XX X+ X+ XX XX?", 3, 6),
    ("   X+XXXX", 3, 3),
)


def _is_black(img: np.ndarray, x: int, y: int) -> bool:
    """A pixel of ``img``, with WHITE surroundings."""
    h, w = img.shape
    return 0 <= x < w and 0 <= y < h and bool(img[y, x])


def hit_miss_replace_in_place(
    img: np.ndarray, pattern: str, width: int, height: int
) -> None:
    """hitMissReplaceInPlace with WHITE surroundings."""
    minus_pos = pattern.find("-")
    plus_pos = pattern.find("+")
    if minus_pos >= 0 and plus_pos >= 0:
        origin_pos = min(minus_pos, plus_pos)
    elif minus_pos >= 0:
        origin_pos = minus_pos
    elif plus_pos >= 0:
        origin_pos = plus_pos
    else:
        return
    origin_x, origin_y = origin_pos % width, origin_pos // width

    hits, misses, white_to_black, black_to_white = [], [], [], []
    for y in range(height):
        for x in range(width):
            offset = (x - origin_x, y - origin_y)
            cell = pattern[y * width + x]
            if cell == "-":
                black_to_white.append(offset)
                hits.append(offset)
            elif cell == "X":
                hits.append(offset)
            elif cell == "+":
                white_to_black.append(offset)
                misses.append(offset)
            elif cell == " ":
                misses.append(offset)
            elif cell != "?":
                raise ValueError("hitMissReplace: invalid character in pattern")

    # hitMissMatch: a mark at each pixel of the image where the origin may sit.
    h, w = img.shape
    matches = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            matches[y, x] = all(
                _is_black(img, x + dx, y + dy) for dx, dy in hits
            ) and not any(_is_black(img, x + dx, y + dy) for dx, dy in misses)

    for dx, dy in white_to_black:
        for y, x in zip(*np.nonzero(matches), strict=True):
            if 0 <= x + dx < w and 0 <= y + dy < h:
                img[y + dy, x + dx] = True
    for dx, dy in black_to_white:
        for y, x in zip(*np.nonzero(matches), strict=True):
            if 0 <= x + dx < w and 0 <= y + dy < h:
                img[y + dy, x + dx] = False


def hit_miss_replace_all_directions(
    img: np.ndarray, pattern: str, width: int, height: int
) -> None:
    """hitMissReplaceAllDirections: as given, then turned 90, 180 and 270 degrees."""
    hit_miss_replace_in_place(img, pattern, width, height)

    turned = [" "] * (width * height)
    new_width, new_height = height, width
    for y in range(height):
        for x in range(width):
            turned[x * new_width + (height - 1 - y)] = pattern[y * width + x]
    hit_miss_replace_in_place(img, "".join(turned), new_width, new_height)

    new_width, new_height = width, height
    for y in range(height):
        for x in range(width):
            turned[(height - 1 - y) * new_width + (width - 1 - x)] = pattern[
                y * width + x
            ]
    hit_miss_replace_in_place(img, "".join(turned), new_width, new_height)

    new_width, new_height = height, width
    for y in range(height):
        for x in range(width):
            turned[(width - 1 - x) * new_width + y] = pattern[y * width + x]
    hit_miss_replace_in_place(img, "".join(turned), new_width, new_height)


def morphological_smooth(img: np.ndarray) -> np.ndarray:
    """morphologicalSmoothInPlace, on a copy."""
    result = img.copy()
    for pattern, width, height in SMOOTHING_PATTERNS:
        hit_miss_replace_all_directions(result, pattern, width, height)
    return result
