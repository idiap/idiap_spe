# SPDX-FileCopyrightText: Idiap Research Institute
# SPDX-FileContributor: Enno Hermann <enno.hermann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

"""Hidden Markov models."""

from collections.abc import Sequence
from typing import TypedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.linalg
from cycler import cycler
from graphviz import Digraph
from scipy.stats._multivariate import multivariate_normal_frozen


class Gaussian(multivariate_normal_frozen):  # type: ignore[misc]
    """Multivariate Gaussian distribution."""

    def __repr__(self) -> str:
        """Represent a Gaussian by its means and covariances."""
        return f"mean={self.mean.tolist()}, cov={self.cov.tolist()}"


class HMM:
    """Hidden Markov Model for multivariate Gaussian-distributed observations.

    Keyword arguments:
    transitions -- Transition probability matrix
    gaussians -- List of Gaussians that define the emission probabilities
    labels -- Optional list of state labels
    """

    def __init__(
        self,
        *,
        transitions: npt.ArrayLike,
        gaussians: Sequence[Gaussian],
        labels: Sequence[str] | None = None,
    ) -> None:
        """Create an HMM."""
        self.transitions = np.atleast_1d(transitions)
        if len(self.transitions) != len(gaussians) or len(gaussians) == 0:
            msg = "Transition matrix must match number of Gaussians and be non-zero"
            raise ValueError(msg)

        # Transition probability matrix `A`
        with np.errstate(divide="ignore"):  # Ignore divide by zero warning
            self.log_transitions = np.log(self.transitions)

        # Emission probabilities `b(x_t)` modelled by Gaussians
        self.gaussians = gaussians

        # Optional state labels
        self.labels = labels if labels else len(gaussians) * [""]

        self.n_states = len(gaussians)
        self.dim = len(gaussians[0].mean)

    def __repr__(self) -> str:
        """Represent an HMM by its transitions and Gaussians."""
        return (
            f"HMM(transitions={self.transitions.tolist()}, gaussians={self.gaussians})"
        )

    def sample(
        self, *, plot: bool = False
    ) -> tuple[npt.NDArray[np.float64], list[int], list[str]]:
        """Draw a sequence of samples from the HMM.

        Keyword arguments:
        plot -- Whether to plot the sequence

        Returns:
        - Observations: (2, N - 2) array
        - States: (N, 1) array
        - List of state labels
        """

        def _sample_state(local_transitions: npt.NDArray[np.float64]) -> int:
            """Sample a state given local transition probabilities."""
            cumsum = np.cumsum(local_transitions)
            rng = np.random.default_rng()
            uniform = rng.random()
            state = 0
            while uniform >= cumsum[state]:
                state += 1
            return state

        # Generate the state sequence
        states = [0]  # Begin with the initial state
        labels = [self.labels[0]]
        t = 1
        while states[-1] != self.n_states - 1:
            t += 1
            state = _sample_state(self.transitions[states[-1]])
            states.append(state)
            labels.append(self.labels[state])

        # Generate observations for each state except initial and final
        observations = np.zeros((len(states) - 2, 2))
        for i, state in enumerate(states[1:-1]):
            observations[i] = self.gaussians[state].rvs()

        if plot:
            self.plot_sample(observations, states)
        return observations, states, labels

    def forward(self, observations: npt.NDArray[np.float64]) -> float:
        """Return log likelihood of the observed sequence w.r.t. the HMM.

        Uses the forward algorithm.
        """
        # Precompute the emission probabilities b(x), i.e. pdfs, for all
        # observations and emitting states
        log_bs = np.zeros((len(observations), self.n_states))
        for state in range(1, self.n_states - 1):
            log_bs[:, state] = np.log(self.gaussians[state].pdf(observations))

        # Compute the initial alphas
        alphas = np.ones((len(observations), self.n_states)) * -np.inf
        alphas[0] = self.log_transitions[0] + log_bs[0]

        # Forward recursion
        for t in range(1, len(observations)):
            for j in range(1, self.n_states - 1):
                alphas[t, j] = log_bs[t, j] + np.logaddexp.reduce(
                    alphas[t - 1] + self.log_transitions[:, j]
                )

        # Termination
        return float(
            np.logaddexp.reduce(alphas[-1] + self.log_transitions[:, self.n_states - 1])
        )

    def viterbi(
        self, observations: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Return most likely state sequence and its log likelihood w.r.t. to the HMM.

        Uses the Viterbi algorithm.
        """
        # Precompute the emission probabilities b(x), i.e. pdfs, for all
        # observations and emitting states
        log_bs = np.zeros((len(observations), self.n_states))
        for state in range(1, self.n_states - 1):
            log_bs[:, state] = np.log(self.gaussians[state].pdf(observations))

        # Compute the initial deltas
        deltas = np.ones((len(observations), self.n_states)) * -np.inf
        deltas[0] = self.log_transitions[0] + log_bs[0]

        # Initialize backpointers
        pointers = np.zeros((len(observations), self.n_states), dtype=int)

        # Viterbi recursion
        for t in range(1, len(observations)):
            for j in range(1, self.n_states - 1):
                deltas[t, j] = (
                    np.max(deltas[t - 1] + self.log_transitions[:, j]) + log_bs[t, j]
                )
                pointers[t, j] = np.argmax(deltas[t - 1] + self.log_transitions[:, j])

        # Termination
        log_likelihood = np.max(deltas[-1] + self.log_transitions[:, self.n_states - 1])

        # Backtracking
        path = np.zeros((len(observations) + 2), dtype=int)
        path[-1] = self.n_states - 1
        path[-2] = np.argmax(deltas[-1] + self.log_transitions[:, self.n_states - 1])
        for t in range(len(observations) - 1, 0, -1):
            path[t] = pointers[t, path[t + 1]]
        path[0] = 0

        return path, log_likelihood

    def plot(self) -> Digraph:
        """Plot the states and transitions of the HMM."""
        g = Digraph()
        g.attr(rankdir="LR")  # left-to-right layout
        g.attr("edge", fontsize="10")

        # Initial and final state
        g.attr("node", shape="doublecircle")
        g.node("0", self.labels[0])
        g.node(str(len(self.transitions) - 1), self.labels[-1])

        # Other states
        g.attr("node", shape="circle")
        for i, label in enumerate(self.labels):
            g.node(str(i), label)

        # Edges
        for i in range(len(self.transitions)):
            for j in range(len(self.transitions)):
                if self.transitions[i][j] > 0:
                    g.edge(str(i), str(j), label=str(self.transitions[i][j]))
        return g

    def pprint(self) -> Digraph:
        """Print and plot a nice representation of the HMM."""
        print(f"States: {self.labels}\n")

        print("Transition matrix:")
        print(self.transitions)

        print("\nGraph:", end="")
        return self.plot()

    def plot_sample(
        self,
        observations: npt.NDArray[np.float64] | None = None,
        states: Sequence[int] | None = None,
    ) -> None:
        """Plot a sample of the HMM."""
        if observations is None:
            observations, states, _ = self.sample()

        _fig = plt.figure(figsize=(16, 8))
        ax0 = plt.subplot(2, 4, (1, 2))
        ax1 = plt.subplot(2, 4, (5, 6))
        ax2 = plt.subplot(1, 4, (3, 4))
        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        gray = "0.8"
        _cmap = cycler(color=mpl.colormaps["Dark2"](range(self.n_states - 2)))

        ax0.set_title("First dimension")
        ax0.plot(observations[:, 0], c=gray)

        ax1.set_title("Second dimension")
        ax1.set_xlabel("Steps")
        ax1.plot(observations[:, 1], c=gray)

        ax2.set_title("2D")
        ax2.set_xlabel("First dimension")
        ax2.set_ylabel("Second dimension")
        ax2.plot(observations[:, 0], observations[:, 1], c=gray)

        if states is not None:
            # Discard the non-emitting initial and final states
            states_np = np.array(states)[1:-1, np.newaxis]
            if len(observations) != len(states_np):
                msg = "Number of observations and states does not match"
                raise ValueError(msg)

            for state in range(1, self.n_states - 1):
                filtered = np.where((states_np == state), observations, np.nan)
                ax0.plot(filtered[:, 0], marker="o", ls="")
                ax1.plot(filtered[:, 1], marker="o", ls="")
                (base,) = ax2.plot(
                    filtered[:, 0],
                    filtered[:, 1],
                    marker="o",
                    ls="",
                    label=f"State {state}: {self.labels[state]}",
                )
                ax2.plot(
                    self.gaussians[state].mean[0],
                    self.gaussians[state].mean[1],
                    marker="+",
                    mew=3,
                    ms=20,
                    color=base.get_color(),
                )
                t = np.linspace(-np.pi, np.pi, num=100)
                stdev = scipy.linalg.sqrtm(self.gaussians[state].cov)
                circle = np.array([np.cos(t), np.sin(t)]).T.dot(stdev) + np.tile(
                    self.gaussians[state].mean, (100, 1)
                )
                ax2.plot(circle[:, 0], circle[:, 1], color=base.get_color())

            ax2.legend()

        plt.show()  # type: ignore[no-untyped-call,unused-ignore]

    def compare_sequences(
        self,
        observations: npt.NDArray[np.float64],
        states1: npt.NDArray[np.float64],
        states2: npt.NDArray[np.float64],
    ) -> None:
        """Compare two alignments of the observation sequence."""
        plt.figure(figsize=(16, 8))
        ax0 = plt.subplot(2, 1, 1)
        ax1 = plt.subplot(2, 1, 2)
        plt.subplots_adjust(wspace=0.5, hspace=0.3)
        gray = "0.8"
        cycler(color=mpl.colormaps["Dark2"](range(self.n_states - 2)))

        ax0.set_title("First state sequence")
        ax0.plot(observations[:, 0], c=gray)

        ax1.set_title("Second state sequence")
        ax1.set_xlabel("Steps")
        ax1.plot(observations[:, 0], c=gray)

        # Discard the non-emitting initial and final states
        states1 = states1[1:-1, np.newaxis]
        states2 = states2[1:-1, np.newaxis]
        if not len(observations) == len(states1) == len(states2):
            msg = "Number of observations and states does not match"
            raise ValueError(msg)

        # Identify misalignments
        misalignments = np.where((states1 != states2), observations, np.nan)

        for state in range(1, self.n_states - 1):
            filtered1 = np.where((states1 == state), observations, np.nan)
            filtered2 = np.where((states2 == state), observations, np.nan)
            ax0.plot(
                filtered1[:, 0],
                marker="o",
                ls="",
                label=f"State {state}: {self.labels[state]}",
            )
            ax1.plot(filtered2[:, 0], marker="o", ls="")

        # Plot misalignments
        class PlotDict(TypedDict):
            marker: str
            ls: str
            markerfacecolor: str
            markeredgecolor: str
            markersize: int

        kwargs: PlotDict = {
            "marker": "o",
            "ls": "",
            "markerfacecolor": "none",
            "markeredgecolor": "r",
            "markersize": 15,
        }
        label = "No misalignments" if np.isnan(misalignments).all() else "Misalignments"
        ax0.plot(misalignments[:, 0], label=label, **kwargs)
        ax1.plot(misalignments[:, 0], **kwargs)

        ax0.legend()

        plt.show()  # type: ignore[no-untyped-call,unused-ignore]
