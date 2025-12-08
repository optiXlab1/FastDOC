import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})

CSV_PATH = "route_points.csv"
SWAP_XY = False

xs, ys, zs, v_mps = [], [], [], []

with open(CSV_PATH, newline="") as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        x = float(r["x"])
        y = float(r["y"])
        z = float(r["z"])
        if SWAP_XY:
            xx, yy = y, x
        else:
            xx, yy = x, y
        xs.append(xx)
        ys.append(yy)
        zs.append(z)
        v_mps.append(float(r["target_speed_mps"]))

xs = np.array(xs)
ys = np.array(ys)
speeds = np.array(v_mps) * 3.6  # m/s → km/h

points = np.array([xs, ys]).T.reshape(-1, 1, 2)
segs = np.concatenate([points[:-1], points[1:]], axis=1)

fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=120)

vmin, vmax = 0, 25
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
lc = LineCollection(segs, cmap="turbo", norm=norm, linewidths=1.6)
lc.set_array(speeds)
line = ax.add_collection(lc)

cb = fig.colorbar(line, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Target speed (km/h)")
cb.set_ticks(np.linspace(vmin, vmax, 6))  # 0, 5, 10, 15, 20, 25

ax.set_aspect("equal", "box")
ax.set_title("Route with target speed colormap")

ax.set_xticklabels([])
ax.set_yticklabels([])

ax.set_xlim(xs.min() - 10, xs.max() + 5)
ax.set_ylim(ys.min() - 5, ys.max() + 5)
ax.xaxis.set_major_locator(MultipleLocator(20.0))
ax.yaxis.set_major_locator(MultipleLocator(20.0))

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()

out_svg = "route_speed_map.svg"
plt.savefig(out_svg, format="svg", bbox_inches="tight")
print(f"[save] {out_svg}")
plt.close(fig)
