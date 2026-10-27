"""Output's edge smoothing against upstream, and the strokes it must leave alone."""

import cv2
import numpy as np
import pytest

from scantailor.core.models import Dpi
from scantailor.filters.output.binarization import BinarizationOptions
from scantailor.filters.output.color_mode import ColorMode
from scantailor.filters.output.despeckle import DespeckleLevel
from scantailor.filters.output.generator import _morphological_smooth, generate_output
from scantailor.filters.output.params import Params
from tests.filters.output import cpp_reference as cpp


def _ragged_text(seed: int) -> np.ndarray:
    """Anti-aliased text under gray noise, thresholded: edges as ragged as a scan's."""
    rng = np.random.default_rng(seed)
    gray = np.full((48, 120), 230, dtype=np.int16)
    cv2.putText(
        gray, "Rag edge", (2, 36), cv2.FONT_HERSHEY_TRIPLEX, 1.1, 20, 2, cv2.LINE_AA
    )
    return np.where(gray + rng.normal(0, 40, gray.shape) < 125, 0, 255).astype(np.uint8)


def _strokes(width: int) -> np.ndarray:
    """Vertical, horizontal and diagonal strokes ``width`` pixels wide; 0 is ink."""
    page = np.full((80, 80), 255, dtype=np.uint8)
    page[10:70, 20 : 20 + width] = 0
    page[50 : 50 + width, 30:75] = 0
    for i in range(40):
        page[5 + i, 35 + i : 35 + i + width] = 0
    return page


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_smoothing_ragged_text_matches_upstream(seed: int) -> None:
    binary = _ragged_text(seed)
    expected = cpp.morphological_smooth(binary == 0)

    got = _morphological_smooth(binary)

    assert (expected != (binary == 0)).any()
    np.testing.assert_array_equal(got == 0, expected)


@pytest.mark.parametrize("width", [1, 2])
def test_smoothing_keeps_a_thin_stroke(width: int) -> None:
    page = _strokes(width)

    np.testing.assert_array_equal(_morphological_smooth(page), page)


def test_black_and_white_output_keeps_a_two_pixel_stroke() -> None:
    gray = np.full((100, 100), 230, dtype=np.uint8)
    gray[10:90, 48:50] = 30
    params = Params(
        color_mode=ColorMode.BLACK_AND_WHITE,
        output_dpi=Dpi.uniform(300),
        binarization=BinarizationOptions(morphological_smoothing=True),
        despeckle_level=DespeckleLevel.OFF,
    )

    result = generate_output(gray, params)

    assert (result.image[10:90, 47:51] == 0).any(axis=1).all()
