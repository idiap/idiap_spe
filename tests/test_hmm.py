# SPDX-FileCopyrightText: Idiap Research Institute
# SPDX-FileContributor: Enno Hermann <enno.hermann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

"""Unit tests for hidden Markov model module."""

import pytest

from idiap_spe.hmm import HMM, Gaussian


@pytest.fixture()
def gaussian() -> Gaussian:
    """Gaussian fixture."""
    return Gaussian(mean=[0, 0], cov=[[0, 0], [0, 0]], allow_singular=True)


def test_gmm_repr(gaussian: Gaussian) -> None:
    """Test how Gaussians are printed."""
    assert str(gaussian) == "mean=[0.0, 0.0], cov=[[0.0, 0.0], [0.0, 0.0]]"


def test_hmm_repr(gaussian: Gaussian) -> None:
    """Test how HMMs are printed."""
    hmm = HMM(labels=["x"], transitions=[[1.0]], gaussians=[gaussian])
    assert str(hmm) == (
        "HMM(transitions=[[1.0]], "
        "gaussians=[mean=[0.0, 0.0], cov=[[0.0, 0.0], [0.0, 0.0]]])"
    )
