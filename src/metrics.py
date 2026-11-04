"""
metrics.py - evaluation functions used by train.py.
"""
import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root mean squared error (mg/dL). Large errors count extra."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error (mg/dL): the typical size of an error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def clarke_zone(ref: float, pred: float) -> str:
    """Clarke error grid zone of ONE prediction (mg/dL).
    A = clinically accurate, B = benign error, C = overcorrection,
    D = dangerous failure to detect, E = erroneous (treat low as high or vice versa).
    """
    x, y = ref, pred
    if (x <= 70 and y <= 70) or (0.8 * x <= y <= 1.2 * x):
        return "A"
    if (x >= 180 and y <= 70) or (x <= 70 and y >= 180):
        return "E"
    if (70 <= x <= 290 and y >= x + 110) or (130 <= x <= 180 and y <= 7 / 5 * x - 182):
        return "C"
    if (x >= 240 and 70 <= y <= 180) or (x <= 175 / 3 and 70 <= y <= 180) \
            or (175 / 3 <= x <= 70 and y >= 6 / 5 * x):
        return "D"
    return "B"


def clarke_ab(y_true, y_pred) -> float:
    """Share of predictions in Clarke zones A+B (%). Target for a good model: > 99 %."""
    zones = [clarke_zone(a, b) for a, b in zip(y_true, y_pred)]
    return 100 * np.mean([z in ("A", "B") for z in zones])


def clarke_a(y_true, y_pred) -> float:
    """Share of predictions in zone A only (%)."""
    zones = [clarke_zone(a, b) for a, b in zip(y_true, y_pred)]
    return 100 * np.mean([z == "A" for z in zones])


def mard(y_true, y_pred) -> float:
    """Mean absolute relative difference (%): typical error as a percent of the true value.
    The same measure used for CGM sensor accuracy (good sensors: ~8-10 %)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(100 * np.mean(np.abs(y_pred - y_true) / y_true))


def clarke_shares(y_true, y_pred) -> dict:
    """Share (%) of predictions in each Clarke zone A-E."""
    zones = np.array([clarke_zone(a, b) for a, b in zip(y_true, y_pred)])
    return {z: 100 * np.mean(zones == z) for z in "ABCDE"}


def plot_clarke(ax, y_true, y_pred, title: str) -> None:
    """Draw the Clarke error grid: true value (x) vs predicted value (y), with zone borders."""
    ink, muted = "#0b0b0b", "#8a8984"
    ax.scatter(y_true, y_pred, s=2, color="#2a78d6", alpha=0.15, linewidths=0)
    # zone borders (standard Clarke 1987 layout, mg/dL)
    lines = [([0, 400], [0, 400]), ([0, 175 / 3], [70, 70]), ([175 / 3, 400 / 1.2], [70, 400]),
             ([70, 70], [84, 400]), ([0, 70], [180, 180]), ([70, 290], [180, 400]),
             ([70, 70], [0, 56]), ([70, 400], [56, 320]), ([180, 180], [0, 70]),
             ([180, 400], [70, 70]), ([240, 240], [70, 180]), ([240, 400], [180, 180]),
             ([130, 180], [0, 70])]
    for xs, ys in lines:
        ax.plot(xs, ys, color=ink, lw=0.8, ls=":" if xs == [0, 400] else "-")
    for label, x, y in [("A", 30, 15), ("B", 370, 260), ("B", 280, 370), ("C", 160, 370),
                        ("C", 160, 15), ("D", 30, 140), ("D", 370, 120), ("E", 30, 370), ("E", 370, 15)]:
        ax.text(x, y, label, fontsize=11, color=ink, ha="center", va="center")
    s = clarke_shares(y_true, y_pred)
    ax.set_title(f"{title}\nA {s['A']:.1f}%  B {s['B']:.1f}%  C {s['C']:.1f}%  D {s['D']:.1f}%  E {s['E']:.1f}%",
                 fontsize=10, color=ink, loc="left")
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_aspect("equal")
    ax.set_xlabel("True glucose (mg/dL)", color=ink, fontsize=9)
    ax.set_ylabel("Predicted glucose (mg/dL)", color=ink, fontsize=9)
    ax.tick_params(colors=muted, labelsize=8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)