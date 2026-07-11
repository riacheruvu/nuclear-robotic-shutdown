# Related tools (literature-informed map)

This project does **not** try to replace the tools below. It fills a gap:
**scenario → formal + rare-event qualification report** for latency-aware remote
hazardous robotics.

| Layer | Examples | Overlap with us |
|---|---|---|
| Robot platforms | Boston Dynamics Spot SDK | Deployment, not qualification science |
| Rad-aware sim | Wright et al. Gazebo radiation (2021); RCI_radiation (ROS 2) | Environment fidelity |
| Industry ALARA / DMU | VISIPLAN, DEMplus, Coppelia, Isaac Sim | Planning / training viz |
| Set-based reachability | JuliaReach, CORA, CommonRoad-Reach, PyBDR | Heavier backends (future plugins) |
| UQ / rare events | UQpy, OpenTURNS | General reliability; we keep domain-specific IS |
| Falsification | Scenic, VerifAI, Adaptive Stress Testing | Find failures; we also *estimate rates* |

See also the Medium article linked from the README for the project narrative.
