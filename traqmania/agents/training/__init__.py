"""Backend-agnostic training loops (shared between classical and quantum agents)."""

from traqmania.agents.training.dqn import DEFAULT_EVAL_EPISODES, Adam, DQNTrainer

__all__ = ["DEFAULT_EVAL_EPISODES", "Adam", "DQNTrainer"]
