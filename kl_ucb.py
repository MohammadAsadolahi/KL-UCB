"""
KL-UCB: Kullback-Leibler Upper Confidence Bound Algorithm
==========================================================

A high-performance implementation of the KL-UCB policy for bounded stochastic
multi-armed bandits, as described in:

    Garivier, A. & Cappé, O. (2011).
    "The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond."
    Proceedings of the 24th Annual Conference on Learning Theory (COLT).
    https://arxiv.org/abs/1102.2490

KL-UCB achieves asymptotically optimal regret bounds by leveraging the
Kullback-Leibler divergence to construct tighter confidence intervals than
classical UCB strategies.

Usage:
    python kl_ucb.py
    python kl_ucb.py --arms 3 --horizon 10000 --experiments 20
    python kl_ucb.py --arms 2 --horizon 5000 --probabilities 0.3 0.8
    python kl_ucb.py --scenario all
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Core Algorithm
# ---------------------------------------------------------------------------

class KLUCBPolicy:
    """Kullback-Leibler Upper Confidence Bound policy for Bernoulli bandits.

    The KL-UCB policy selects the arm that maximises an upper confidence index
    derived from the KL divergence, yielding tighter bounds than standard UCB1.

    Parameters
    ----------
    n_arms : int
        Number of arms (actions) in the bandit problem.
    exploration_rate : float, optional
        Scaling constant *c* for the exploration term ``log(t) + c·log(log(t))``.
        Default is 3, as suggested in the original paper.
    precision : float, optional
        Tolerance for the binary-search computation of the KL-UCB index.
        Default is 1e-4.
    """

    def __init__(
        self,
        n_arms: int,
        exploration_rate: float = 3.0,
        precision: float = 1e-4,
    ) -> None:
        if n_arms < 1:
            raise ValueError("n_arms must be >= 1")

        self.n_arms = n_arms
        self.exploration_rate = exploration_rate
        self.precision = precision

        # Per-arm statistics
        self.pull_counts: np.ndarray = np.zeros(n_arms, dtype=np.float64)
        self.reward_sums: np.ndarray = np.zeros(n_arms, dtype=np.float64)

    # -- KL divergence -------------------------------------------------------

    @staticmethod
    def kl_bernoulli(p: float, q: float) -> float:
        """Compute the Kullback-Leibler divergence between Bernoulli(p) and Bernoulli(q).

        KL(p || q) = p·ln(p/q) + (1-p)·ln((1-p)/(1-q))

        Returns ``0`` when *p* is 0, and ``inf`` for degenerate cases.
        """
        if p == 0.0:
            if q >= 1.0:
                return float("inf")
            return -math.log(1.0 - q) if q > 0.0 else 0.0
        if p == 1.0:
            if q <= 0.0:
                return float("inf")
            return -math.log(q) if q < 1.0 else 0.0
        if q <= 0.0 or q >= 1.0:
            return float("inf")
        return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))

    # -- Index computation ---------------------------------------------------

    def _compute_index(self, arm: int, t: int) -> float:
        """Compute the KL-UCB index for *arm* at time step *t* via binary search.

        Finds the largest q in [p, 1] such that:
            KL(p, q) <= (log(t) + c·log(log(t))) / N_k
        """
        n_k = self.pull_counts[arm]
        if n_k == 0:
            return float("inf")  # force exploration of un-pulled arms

        p = self.reward_sums[arm] / n_k
        log_t = math.log(t)
        threshold = (log_t + self.exploration_rate *
                     math.log(max(log_t, 1e-10))) / n_k

        # Binary search for the maximal q
        q = p
        step = (1.0 - p) / 2.0
        while step > self.precision:
            if self.kl_bernoulli(p, q + step) <= threshold:
                q += step
            step /= 2.0
        return q

    # -- Arm selection -------------------------------------------------------

    def select_arm(self) -> int:
        """Select the arm with the highest KL-UCB index.

        Returns the index of the chosen arm.
        """
        t = int(np.sum(self.pull_counts))

        # Force round-robin initialisation for un-pulled arms
        for k in range(self.n_arms):
            if self.pull_counts[k] == 0:
                return k

        indices = np.array([self._compute_index(k, t)
                           for k in range(self.n_arms)])
        return int(np.argmax(indices))

    # -- Update --------------------------------------------------------------

    def update(self, arm: int, reward: float) -> None:
        """Record the *reward* obtained from pulling *arm*."""
        self.pull_counts[arm] += 1
        self.reward_sums[arm] += reward


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for a single bandit experiment."""
    n_arms: int = 2
    horizon: int = 5000
    reward_probabilities: Sequence[float] = (0.1, 0.9)
    n_experiments: int = 10
    exploration_rate: float = 3.0
    label: str = ""

    def __post_init__(self) -> None:
        if len(self.reward_probabilities) != self.n_arms:
            raise ValueError(
                f"Expected {self.n_arms} probabilities, "
                f"got {len(self.reward_probabilities)}"
            )
        for p in self.reward_probabilities:
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"Probability must be in [0, 1], got {p}")


@dataclass
class ExperimentResult:
    """Stores results from a bandit experiment."""
    config: ExperimentConfig
    cumulative_rewards: list[np.ndarray] = field(default_factory=list)
    total_rewards: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a complete bandit experiment with the KL-UCB policy.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment parameters.

    Returns
    -------
    ExperimentResult
        Collected metrics across all experiment runs.
    """
    result = ExperimentResult(config=config)
    rng = np.random.default_rng()

    for _ in range(config.n_experiments):
        policy = KLUCBPolicy(
            n_arms=config.n_arms,
            exploration_rate=config.exploration_rate,
        )

        actions = np.zeros((config.n_arms, config.horizon), dtype=int)
        rewards = np.zeros((config.n_arms, config.horizon), dtype=int)

        # Initialisation: pull each arm once
        for arm in range(config.n_arms):
            r = rng.binomial(1, config.reward_probabilities[arm])
            policy.update(arm, r)

        # Main loop
        for t in range(config.n_arms, config.horizon):
            arm = policy.select_arm()
            actions[arm, t] = 1
            r = rng.binomial(1, config.reward_probabilities[arm])
            rewards[arm, t] = r
            policy.update(arm, r)

        cumulative = np.cumsum(rewards, axis=1)
        total = np.sum(cumulative, axis=0)

        result.cumulative_rewards.append(cumulative)
        result.total_rewards.append(total)
        result.actions.append(actions)

    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

# Colour palette inspired by modern data-viz standards
ARM_COLORS = [
    "#2563EB",  # blue
    "#DC2626",  # red
    "#16A34A",  # green
    "#D97706",  # amber
    "#7C3AED",  # violet
    "#0891B2",  # cyan
    "#DB2777",  # pink
    "#65A30D",  # lime
]

ARM_LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


def plot_results(
    result: ExperimentResult,
    ax: plt.Axes | None = None,
    show: bool = False,
) -> plt.Axes:
    """Plot cumulative rewards per arm across experiments.

    Parameters
    ----------
    result : ExperimentResult
        Output from ``run_experiment``.
    ax : matplotlib.axes.Axes, optional
        Target axes. Created if not provided.
    show : bool
        If True, call ``plt.show()`` at the end.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FAFAFA")

    config = result.config
    probs_str = ", ".join(f"{p:.2f}" for p in config.reward_probabilities)
    title = f"KL-UCB — Cumulative Rewards  |  Arms: {config.n_arms}  |  p = [{probs_str}]  |  T = {config.horizon}"
    if config.label:
        title = f"{config.label}: {title}"

    legend_handles = []
    for arm_idx in range(config.n_arms):
        color = ARM_COLORS[arm_idx % len(ARM_COLORS)]
        ls = ARM_LINESTYLES[arm_idx % len(ARM_LINESTYLES)]

        for exp_idx in range(config.n_experiments):
            line, = ax.plot(
                result.cumulative_rewards[exp_idx][arm_idx],
                linestyle=ls,
                color=color,
                alpha=0.6,
                linewidth=1.2,
            )
        # One legend entry per arm
        legend_handles.append(
            plt.Line2D([0], [0], color=color, linestyle=ls, linewidth=2,
                       label=f"Arm {arm_idx + 1}  (p={config.reward_probabilities[arm_idx]:.2f})")
        )

    ax.legend(handles=legend_handles, loc="upper left",
              framealpha=0.9, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time Step (t)", fontsize=11)
    ax.set_ylabel("Cumulative Reward", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


# ---------------------------------------------------------------------------
# Pre-defined Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, ExperimentConfig] = {
    "easy": ExperimentConfig(
        n_arms=2,
        horizon=5000,
        reward_probabilities=(0.1, 0.9),
        n_experiments=10,
        label="Easy",
    ),
    "moderate": ExperimentConfig(
        n_arms=2,
        horizon=5000,
        reward_probabilities=(0.4, 0.7),
        n_experiments=10,
        label="Moderate",
    ),
    "hard": ExperimentConfig(
        n_arms=2,
        horizon=5000,
        reward_probabilities=(0.45, 0.55),
        n_experiments=10,
        label="Hard",
    ),
}


def run_all_scenarios() -> None:
    """Run and visualise all pre-defined scenarios side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        "KL-UCB Algorithm  —  Benchmark Scenarios",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    for ax, (name, config) in zip(axes, SCENARIOS.items()):
        print(
            f"[*] Running scenario: {name} (p={list(config.reward_probabilities)})")
        result = run_experiment(config)
        plot_results(result, ax=ax)

    plt.tight_layout()
    plt.savefig("kl_ucb_results.png", dpi=150,
                bbox_inches="tight", facecolor="#FAFAFA")
    print("[+] Results saved to kl_ucb_results.png")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KL-UCB: Kullback-Leibler Upper Confidence Bound for Multi-Armed Bandits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenario",
        choices=["easy", "moderate", "hard", "all"],
        default=None,
        help="Run a pre-defined scenario (overrides other args). "
             "Use 'all' to run all scenarios.",
    )
    parser.add_argument(
        "--arms", "-k",
        type=int,
        default=2,
        help="Number of arms (default: 2)",
    )
    parser.add_argument(
        "--horizon", "-T",
        type=int,
        default=5000,
        help="Time horizon — total number of rounds (default: 5000)",
    )
    parser.add_argument(
        "--experiments", "-n",
        type=int,
        default=10,
        help="Number of independent experiments (default: 10)",
    )
    parser.add_argument(
        "--probabilities", "-p",
        type=float,
        nargs="+",
        default=None,
        help="Reward probabilities for each arm (must match --arms). "
             "Example: --probabilities 0.3 0.7",
    )
    parser.add_argument(
        "--exploration-rate", "-c",
        type=float,
        default=3.0,
        help="Exploration constant c (default: 3.0)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save the plot to the given file path instead of showing it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Pre-defined scenarios
    if args.scenario == "all":
        run_all_scenarios()
        return

    if args.scenario is not None:
        config = SCENARIOS[args.scenario]
        print(
            f"[*] Running scenario: {args.scenario} (p={list(config.reward_probabilities)})")
        result = run_experiment(config)
        ax = plot_results(result)
        if args.save:
            plt.savefig(args.save, dpi=150, bbox_inches="tight",
                        facecolor="#FAFAFA")
            print(f"[+] Plot saved to {args.save}")
        else:
            plt.tight_layout()
            plt.show()
        return

    # Custom configuration
    probabilities = args.probabilities
    if probabilities is None:
        probabilities = [round(i / (args.arms + 1), 2)
                         for i in range(1, args.arms + 1)]
        print(
            f"[*] No probabilities specified. Using auto-generated: {probabilities}")

    config = ExperimentConfig(
        n_arms=args.arms,
        horizon=args.horizon,
        reward_probabilities=tuple(probabilities),
        n_experiments=args.experiments,
        exploration_rate=args.exploration_rate,
    )

    probs_str = ", ".join(f"{p:.2f}" for p in config.reward_probabilities)
    print(
        f"[*] KL-UCB  |  K={config.n_arms}  |  T={config.horizon}  "
        f"|  p=[{probs_str}]  |  runs={config.n_experiments}  |  c={config.exploration_rate}"
    )

    result = run_experiment(config)
    plot_results(result)

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight",
                    facecolor="#FAFAFA")
        print(f"[+] Plot saved to {args.save}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
