# SPDX-FileCopyrightText: Idiap Research Institute
# SPDX-FileContributor: Enno Hermann <enno.hermann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

"""Unit tests for G2P exercise."""

import cmudict
import phonetisaurus as ps


def test_load_cmudict() -> None:
    """Test that the CMU lexicon can be correctly loaded."""
    lexicon = ps.load_lexicon(cmudict.dict_string().decode().split("\n"))
    assert lexicon["either"] == [["IY1", "DH", "ER0"], ["AY1", "DH", "ER0"]]
