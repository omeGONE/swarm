from __future__ import annotations

import numpy as np

from topologies import TOPOLOGIES, build_neighbors


class PSO:

    def __init__(
        self,
        n_particles: int = 30,
        topology: str = "star",
        n_neighbors: int = 1,
        n_clusters: int = 4,
        alpha: float = 0.95,
        beta: float = 0.2,
        gamma: float = 0.2,
        max_iter: int = 100,
        v_max: float | None = None,
        tol: float | None = None,
        stall_iter: int | None = None,
        radius_tol: float | None = None,
        random_state: int | None = 42,
    ):
        if topology not in TOPOLOGIES:
            raise ValueError(f"topology must be one of {TOPOLOGIES}")
        if n_particles < 1:
            raise ValueError("n_particles must be >= 1")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")

        self.n_particles = n_particles
        self.topology = topology
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.v_max = v_max
        self.tol = tol
        self.stall_iter = stall_iter
        self.radius_tol = radius_tol
        self.random_state = random_state

        self.best_position_: np.ndarray | None = None
        self.best_value_: float | None = None
        self.history_: np.ndarray | None = None
        self.positions_history_: np.ndarray | None = None
        self.neighbors_: list[np.ndarray] | None = None
        self.n_iterations_: int = 0
        self.stop_reason_: str = ""
        self._d0: float = 1.0

    def _social_best(self, P: np.ndarray, P_val: np.ndarray) -> np.ndarray:
        L = np.empty_like(P)
        for i, nb in enumerate(self.neighbors_):
            L[i] = P[nb[np.argmin(P_val[nb])]]
        return L

    def _swarm_radius(self, X: np.ndarray) -> float:
        r_max = np.max(np.linalg.norm(X - self.best_position_, axis=1))
        return float(r_max / self._d0)

    def minimize(
        self,
        func,
        bounds: np.ndarray,
        x0: np.ndarray | None = None,
    ):
        bounds = np.asarray(bounds, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (dim, 2)")
        low, high = bounds[:, 0], bounds[:, 1]
        if np.any(high <= low):
            raise ValueError("each bound must satisfy low < high")

        dim = bounds.shape[0]
        rng = np.random.RandomState(self.random_state)
        self.neighbors_ = build_neighbors(
            self.topology, self.n_particles, self.n_neighbors, self.n_clusters
        )

        if x0 is None:
            X = low + rng.rand(self.n_particles, dim) * (high - low)
        else:
            X = np.asarray(x0, dtype=float).copy()
            if X.shape != (self.n_particles, dim):
                raise ValueError("x0 must have shape (n_particles, dim)")
        V = np.zeros_like(X)

        P = X.copy()
        P_val = np.asarray(func(X), dtype=float).ravel()

        self._d0 = float(np.linalg.norm(high - low))
        v_lim = None if self.v_max is None else self.v_max * (high - low)

        j = int(np.argmin(P_val))
        self.best_position_ = P[j].copy()
        self.best_value_ = float(P_val[j])
        history = [self.best_value_]
        positions = [X.copy()]
        self.n_iterations_ = 0
        self.stop_reason_ = "max_iter"

        for it in range(1, self.max_iter + 1):
            L = self._social_best(P, P_val)

            r1 = rng.rand(self.n_particles, dim)
            r2 = rng.rand(self.n_particles, dim)
            V = self.alpha * V + self.beta * r1 * (P - X) + self.gamma * r2 * (L - X)
            if v_lim is not None:
                V = np.clip(V, -v_lim, v_lim)

            X = np.clip(X + V, low, high)

            values = np.asarray(func(X), dtype=float).ravel()
            improved = values < P_val
            P[improved] = X[improved]
            P_val[improved] = values[improved]

            j = int(np.argmin(P_val))
            self.best_position_ = P[j].copy()
            self.best_value_ = float(P_val[j])
            self.n_iterations_ = it
            history.append(self.best_value_)
            positions.append(X.copy())

            if self.tol is not None and self.best_value_ <= self.tol:
                self.stop_reason_ = "tol"
                break
            if (
                 self.stall_iter is not None
                and it >= self.stall_iter
                and history[-self.stall_iter - 1] == self.best_value_
            ):
                self.stop_reason_ = "stall"
                break
            if self.radius_tol is not None and self._swarm_radius(X) < self.radius_tol:
                self.stop_reason_ = "radius"
                break

        self.history_ = np.asarray(history, dtype=float)
        self.positions_history_ = np.asarray(positions, dtype=float)
        return self