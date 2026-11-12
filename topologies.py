from __future__ import annotations

import numpy as np

TOPOLOGIES = ("star", "ring", "wheel", "clusters")


def build_neighbors(
    topology: str = "star",
    n_particles: int = 30,
    n_neighbors: int = 1,
    n_clusters: int = 4,
) -> list[np.ndarray]:
    if topology not in TOPOLOGIES:
        raise ValueError(f"topology must be one of {TOPOLOGIES}")
    if n_particles < 1:
        raise ValueError("n_particles must be >= 1")

    adj: list[set[int]] = [set() for _ in range(n_particles)]

    def link(i: int, j: int) -> None:
        if i != j:
            adj[i].add(j)
            adj[j].add(i)

    if topology == "star":
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                link(i, j)

    elif topology == "ring":
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        for i in range(n_particles):
            for k in range(1, n_neighbors + 1):
                link(i, (i + k) % n_particles)
                link(i, (i - k) % n_particles)

    elif topology == "wheel":
        for i in range(1, n_particles):
            link(0, i)

    elif topology == "clusters":
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        groups = np.array_split(np.arange(n_particles), n_clusters)
        for g in groups:
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    link(int(g[a]), int(g[b]))
        for k in range(n_clusters):
            g1, g2 = groups[k], groups[(k + 1) % n_clusters]
            if len(g1) == 0 or len(g2) == 0:
                continue
            link(int(g1[0]), int(g2[0]))
            link(int(g1[-1]), int(g2[-1]))

    # else:  # von_neumann
    #     cols = int(np.ceil(np.sqrt(n_particles)))
    #     rows = int(np.ceil(n_particles / cols))
    #     for i in range(n_particles):
    #         r, c = divmod(i, cols)
    #         for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
    #             j = (rr % rows) * cols + (cc % cols)
    #             if j < n_particles:
    #                 link(i, j)

    return [np.array(sorted(s | {i}), dtype=int) for i, s in enumerate(adj)]