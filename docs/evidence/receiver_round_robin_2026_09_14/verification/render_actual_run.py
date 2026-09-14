"""Render saved real trajectories and offline labels; no inference or metric writes."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

run, out = map(Path, sys.argv[1:3])
tasks = [json.loads(p.read_text()) for p in sorted((run / "tasks").glob("*.json"))]
labels = {r["sample_id"]: r for r in json.loads(
    (run / "offline" / "selected_labels.json").read_text())}
assert len(tasks) == 12 and len(labels) == 2
out.mkdir(parents=True, exist_ok=True)
colors = {"Ego": "#377eb8", "alternating": "#e66101", "one_shot": "#1b9e77"}
for g in (5526, 7007):
    selected = [t for t in tasks if t["row"]["g"] == g]
    label = labels[selected[0]["row"]["sample_id"]]
    gt = np.asarray(label["waypoints"])
    fig, (axis, error) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.subplots_adjust(left=.065, right=.985, bottom=.29, top=.84, wspace=.27)
    observed = np.vstack(([0., 0.], gt))
    axis.plot(observed[:, 0], observed[:, 1], color="#222222", marker="s",
              linewidth=1.7, label="Observed trajectory (offline)")
    for task in selected:
        points = np.asarray(task["episode"]["plans"][-1]["output"]["waypoints"])
        v2 = task["receiver"].endswith("_v2")
        arm = task["arm"]
        style = dict(color=colors[arm], linestyle="-" if v2 else "--",
                     marker="o" if v2 else "x", linewidth=1.5)
        curve = np.vstack(([0., 0.], points))
        axis.plot(curve[:, 0], curve[:, 1],
                  label=("v2 " if v2 else "v1 ") + arm, **style)
        error.plot(label["times_seconds"], np.linalg.norm(points - gt, axis=1), **style)
    axis.set(xlabel="x in current ego LiDAR frame (m)", ylabel="y (m)",
             title="Final trajectories (equal spatial scale)")
    axis.set_aspect("equal", adjustable="box")
    error.set(xlabel="Future time (s)", ylabel="Distance to observed trajectory (m)",
              title="Per-waypoint imitation error", xticks=label["times_seconds"])
    for item in (axis, error):
        item.grid(alpha=.25)
    fig.legend(*axis.get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(.5, .035), ncol=4, fontsize=8)
    fig.suptitle("Receiver round-robin smoke | g%d | diagnostic only" % g, fontsize=12)
    fig.savefig(out / ("g%d_receiver_trajectories.png" % g), dpi=160)
    plt.close(fig)
