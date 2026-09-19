"""
Deep Q-Network (DQN) implementation in PyTorch.

This module intentionally avoids high-level RL libraries so every piece of
the algorithm is visible and editable for learning purposes.

Key components:
  - QNetwork     : a small fully-connected network that maps state -> Q(s,a)
  - ReplayBuffer : stores (s, a, r, s', terminated) transitions for replay
  - DQNAgent     : the training loop, epsilon-greedy policy, target-net sync
"""
import random
from collections import deque
from pathlib import Path
from typing import Self

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

# ── Neural network ────────────────────────────────────────────────────


class QNetwork(nn.Module):
    """EXERCISE 2a: the network that maps a state to one Q-value per action.

    Build a small fully-connected net:

        state_dim -> hidden -> hidden -> action_dim

    with a ReLU after each hidden layer. There is NO activation on the output
    layer: these are Q-values (here they are all negative), not probabilities.

    Tip: nn.Sequential(nn.Linear(...), nn.ReLU(), ...) is the shortest route.
    Tip: remember super().__init__() before assigning any submodule.
    Tip: forward() receives a batch of shape (B, state_dim) and must return
         shape (B, action_dim).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        # No activation on the output layer: these are Q-values, and in
        # MountainCar every one of them is negative (the return is minus the
        # episode length). A ReLU or sigmoid here would clamp them to >= 0 and
        # make the target unreachable.
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ────────────────────────────────────────────────────


class ReplayBuffer:
    """Fixed-size FIFO buffer that stores transitions for experience replay."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ── Agent ─────────────────────────────────────────────────────────────


class DQNAgent:
    """
    Deep Q-Network agent implemented from scratch.

    Hyperparameters are intentionally exposed as constructor args so you
    can experiment with them directly.
    """

    def __init__(
        self,
        env_id: str,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 10,
        hidden: int = 128,
        explore_repeat: int = 20,
    ) -> None:
        self.env_id = env_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.target_update_freq = target_update_freq
        self.hidden = hidden
        # EXERCISE 3: upper bound on how many consecutive steps one exploratory
        # action is held for. 1 recovers textbook per-step epsilon-greedy.
        self.explore_repeat = explore_repeat
        self.training_episodes = 0

        # Per-episode exploration state (see select_action). Reset in train().
        self._sticky_action = 0
        self._sticky_left = 0

        env = gym.make(env_id)
        self.state_dim = int(env.observation_space.shape[0])  # type: ignore[index]
        self.action_dim = int(env.action_space.n)  # type: ignore[attr-defined]
        env.close()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_capacity)

    # ── policy ────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, *, deterministic: bool = False) -> int:
        """Epsilon-greedy with TEMPORALLY CORRELATED exploration (persistent actions).

        EXERCISE 3 — the diagnosis, then the fix.

        THE SYMPTOM. With textbook per-step epsilon-greedy, DQN on MountainCar
        reports a perfectly flat -200 forever. Not unstable, not slow: flat.

        THE EVIDENCE (measured, not assumed -- see `scripts/diagnostico_ex3.py`):

          * A uniformly random policy reaches the flag in 0 out of 300 episodes.
            Not "rarely" -- never. The agent has never once observed the event
            it is supposed to learn to cause.
          * Every reward is -1, so with no successful episode in the buffer,
            every state is worth the same thing: the discounted sum of -1
            forever, which is -1/(1-gamma) = -100 for gamma=0.99.
          * Printing the network's Q-values confirms it: the mean crawls toward
            -100 and the SPREAD ACROSS ACTIONS within each state is ~0.00. The
            network says all three actions are equally good everywhere.

        THE READING. The learning code is correct. The network has correctly
        learned that nothing it does matters -- which, given the data it was
        shown, is true. The failure is upstream of learning: it is in how the
        data is COLLECTED.

        WHY UNIFORM EXPLORATION CANNOT WORK HERE. Escaping the valley needs a
        sustained run of roughly 20 pushes in the same direction (the pumping
        motion of a child on a swing). Per-step uniform sampling draws a fresh
        action from 3 choices every step, so consecutive actions are
        statistically INDEPENDENT:

            P(same action 20 times in a row) = (1/3)^20 ~ 3 in 10 billion

        Independent zero-mean pushes average out to a random walk that leaves
        the car sitting in the valley. More episodes will never fix this; the
        required behaviour is effectively impossible for that policy to emit.

        THE FIX. Give exploration the property it lacks: temporal correlation.
        When we decide to explore, we commit to the sampled action for a random
        run of 1..`explore_repeat` steps instead of re-drawing every step. That
        alone produces sustained pushes, the car starts rocking, and some
        episodes reach the flag -- which is all the learning code ever needed.

        Note this changes ONLY exploration: not the reward, not the environment,
        not the learning rule. And `deterministic=True` stays pure greedy, so
        evaluation numbers remain honest.
        """
        if deterministic:
            with torch.no_grad():
                t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                return int(self.q_net(t).argmax(dim=1).item())

        # Still inside a committed exploratory run: hold the same action.
        if self._sticky_left > 0:
            self._sticky_left -= 1
            return self._sticky_action

        if random.random() < self.epsilon:
            self._sticky_action = random.randrange(self.action_dim)
            # `- 1` because this call already consumes the first step of the run.
            self._sticky_left = random.randint(1, max(1, self.explore_repeat)) - 1
            return self._sticky_action

        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q_net(t).argmax(dim=1).item())

    def predict(self, obs: np.ndarray, *, deterministic: bool = True) -> tuple[int, None]:
        return self.select_action(obs, deterministic=deterministic), None

    # ── learning step ─────────────────────────────────────────────────

    def _tensor(self, x, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.array(x), dtype=dtype, device=self.device)

    def _learn(self) -> float:
        """Sample a mini-batch from the buffer and take one gradient step.

        Returns the batch loss value.
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, terminateds = zip(*batch)

        states_t = self._tensor(states)
        actions_t = self._tensor(actions, torch.int64).unsqueeze(1)
        rewards_t = self._tensor(rewards).unsqueeze(1)
        next_states_t = self._tensor(next_states)
        terminateds_t = self._tensor(terminateds).unsqueeze(1)

        # EXERCISE 2b: the DQN learning step. Four things to do:
        #
        #   1. current_q : Q(s, a) from the ONLINE net, for the actions that
        #      were actually taken. self.q_net(states_t) is (B, action_dim);
        #      you want (B, 1). Tip: .gather(1, actions_t) picks one column
        #      per row.
        #
        #   2. next_q : max_a' Q_target(s', a') from the FROZEN TARGET net.
        #      Tip: .max(dim=1, keepdim=True).values
        #      Tip: wrap this in `with torch.no_grad():` -- no gradient should
        #      flow into the target, that is the whole point of a target net.
        #
        #   3. target_q : the Bellman target, r + gamma * next_q, but with the
        #      bootstrap term zeroed out wherever terminateds_t is 1.
        #      Tip: multiplying by (1.0 - terminateds_t) does this branchlessly.
        #
        #   4. Take one gradient step on self.loss_fn(current_q, target_q).
        #      Tip: zero_grad() -> backward() -> step(), in that order.
        #
        # Return the scalar loss value (.item()).

        # 1. Q(s, a) from the ONLINE net, only for the actions actually taken.
        #    q_net(states_t) is (B, action_dim); gather picks one column per
        #    row, leaving (B, 1) -- the same shape as the target below.
        current_q = self.q_net(states_t).gather(1, actions_t)

        # 2-3. The Bellman target, built from the FROZEN TARGET net and with no
        #      gradient flowing into it. `(1 - terminateds_t)` zeroes the
        #      bootstrap term exactly on the transitions that reached the flag.
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True).values
            target_q = rewards_t + self.gamma * next_q * (1.0 - terminateds_t)

        assert current_q.shape == target_q.shape, (
            f"shape mismatch: {current_q.shape} vs {target_q.shape} -- "
            "a silent broadcast here trains on nonsense"
        )

        # 4. One gradient step.
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    # ── training loop ─────────────────────────────────────────────────

    def train(self, total_episodes: int = 500, log_interval: int = 10) -> list[float]:
        env = gym.make(self.env_id)
        rewards_history: list[float] = []

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False

            # EXERCISE 3: clear the per-episode exploration state so a
            # committed run never leaks across the episode boundary.
            self._sticky_left = 0

            while not done:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Store `terminated`, not `done`: hitting the 200-step time
                # limit is not a real terminal state, so we must keep
                # bootstrapping through it.
                self.buffer.push(obs, action, float(reward), next_obs, terminated)
                self._learn()

                obs = next_obs
                total_reward += reward

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.training_episodes += 1
            rewards_history.append(total_reward)

            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            if episode % log_interval == 0:
                avg = np.mean(rewards_history[-log_interval:])
                print(
                    f"Episode {episode}/{total_episodes} | "
                    f"Avg Reward: {avg:.2f} | "
                    f"Epsilon: {self.epsilon:.4f} | "
                    f"Buffer: {len(self.buffer)}"
                )

        env.close()
        return rewards_history

    # ── persistence ───────────────────────────────────────────────────

    _HPARAMS = (
        "env_id",
        "lr",
        "gamma",
        "epsilon_end",
        "epsilon_decay",
        "batch_size",
        "buffer_capacity",
        "target_update_freq",
        "hidden",
        "explore_repeat",  # EXERCISE 3: must round-trip through save/load
    )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: getattr(self, k) for k in self._HPARAMS}
        data["q_net_state"] = self.q_net.state_dict()
        data["optimizer_state"] = self.optimizer.state_dict()
        data["epsilon"] = self.epsilon
        data["training_episodes"] = self.training_episodes
        torch.save(data, path)
        print(f"Saved DQN agent to {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        data = torch.load(path, weights_only=False)
        agent = cls(
            data["env_id"],
            epsilon_start=data["epsilon"],
            **{k: data[k] for k in cls._HPARAMS if k != "env_id"},
        )
        # The target net starts as a copy of the online net; it re-syncs during
        # training anyway, so there is no need to persist it separately.
        agent.q_net.load_state_dict(data["q_net_state"])
        agent.target_net.load_state_dict(data["q_net_state"])
        agent.optimizer.load_state_dict(data["optimizer_state"])
        agent.training_episodes = data["training_episodes"]
        return agent

    def info(self) -> str:
        params = sum(p.numel() for p in self.q_net.parameters())
        return (
            f"DQN agent for {self.env_id}\n"
            f"  Episodes trained  : {self.training_episodes}\n"
            f"  Network params    : {params:,}\n"
            f"  Epsilon           : {self.epsilon:.4f}\n"
            f"  LR / Gamma        : {self.lr} / {self.gamma}\n"
            f"  Batch size        : {self.batch_size}\n"
            f"  Target update     : every {self.target_update_freq} episodes\n"
            f"  Explore repeat    : up to {self.explore_repeat} steps per exploratory action\n"
            f"  Device            : {self.device}"
        )
