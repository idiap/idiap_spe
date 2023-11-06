# SPDX-FileCopyrightText: Idiap Research Institute
# SPDX-FileContributor: Enno Hermann <enno.hermann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

"""Unit tests for speech signal analysis module."""

from idiap_spe.speech_analysis import load_signal


def test_load_signal() -> None:
    """Test that builtin speech signals can be loaded."""
    utterance = "f_s1_t1_a"
    data = load_signal(utterance)
    assert data[0] == -126  # noqa: PLR2004
