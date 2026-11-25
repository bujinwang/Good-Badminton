"""Derive a stance/formation panel from detections.jsonl.

The upstream project outputs per-frame player court coordinates and speed but
does not aggregate them into the "站位俯视图 / 前后场占比 / 覆盖面积 / 网前时间
/ 站位时间线" panel. This script computes those for singles (front = near net).

Usage:
    python analyze_formation.py outputs/demo1/detections.jsonl outputs/demo1/formation.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial import ConvexHull

COURT_W, COURT_L = 6.1, 13.4
NET_Y = COURT_L / 2
NET_ZONE = 3.0  # "网前" = within 3 m of the net
MIN_HOLD_SEC = 0.4  # ignore shorter stance flips

FONT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "simhei.ttf"))
fm.fontManager.addfont(FONT_PATH)
FONT_PROP = fm.FontProperties(fname=FONT_PATH)
plt.rcParams["font.sans-serif"] = [FONT_PROP.get_name()]
plt.rcParams["axes.unicode_minus"] = False

BG = "#0d1b2a"
PANEL = "#152a3d"
FG = "#e8f1f8"
MUTED = "#9db2c4"
FRONT = "#f5a623"
BACK = "#9b6bd6"
COLORS = ["#f5d76e", "#3fd0d4"]


def load(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    times = {"upper": [], "lower": []}
    coords = {"upper": [], "lower": []}
    speeds = {"upper": [], "lower": []}
    for r in rows:
        players = r.get("players") or {}
        for side in ("upper", "lower"):
            p = players.get(side) or {}
            c = p.get("court")
            if not c or len(c) != 2:
                continue
            x, y = float(c[0]), float(c[1])
            if not (0 <= x <= COURT_W and 0 <= y <= COURT_L):
                continue
            times[side].append(float(r["time_sec"]))
            coords[side].append((x, y))
            speeds[side].append(float(p.get("speed") or 0.0))
    n = min(len(coords["upper"]), len(coords["lower"]))
    t = np.asarray(times["upper"][:n])
    pts = {s: np.asarray(coords[s][:n]) for s in ("upper", "lower")}
    spd = {s: np.asarray(speeds[s][:n]) for s in ("upper", "lower")}
    return t, pts, spd


def summarize(t, pts, spd, side):
    p = pts[side]
    x, y = p[:, 0], p[:, 1]
    dist = float(np.trapz(spd[side], t))
    try:
        area = float(ConvexHull(p).volume) if len(p) >= 3 else 0.0
    except Exception:
        area = 0.0
    near_net = y > NET_Y - NET_ZONE if side == "upper" else y < NET_Y + NET_ZONE
    front_pct = float(near_net.mean() * 100)

    state = near_net.astype(int)
    # de-bounce: remove runs shorter than MIN_HOLD_SEC
    runs = []
    start = 0
    for i in range(1, len(state) + 1):
        if i == len(state) or state[i] != state[start]:
            runs.append((state[start], t[start], t[i - 1] if i < len(state) else t[-1]))
            start = i
    kept = [r for r in runs if r[2] - r[1] >= MIN_HOLD_SEC]
    switches = sum(1 for i in range(1, len(kept)) if kept[i][0] != kept[i - 1][0])
    return {
        "dist": dist,
        "area": area,
        "front_pct": front_pct,
        "back_pct": 100 - front_pct,
        "switches": switches,
        "state": state,
        "current": p[-1],
        "points": p,
    }


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "outputs/demo1/detections.jsonl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "outputs/demo1/formation.png"
    t, pts, spd = load(src)
    if len(t) < 2:
        raise SystemExit("Not enough data")

    a, b = summarize(t, pts, spd, "upper"), summarize(t, pts, spd, "lower")
    duration = t[-1] - t[0]

    fig = plt.figure(figsize=(15, 8.6), facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.85, 1.15], height_ratios=[1, 0.5],
                          left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.12, hspace=0.28)

    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(PANEL)
    ax.add_patch(Rectangle((0, 0), COURT_W, COURT_L, fill=False, edgecolor=MUTED, lw=2))
    ax.plot([0, COURT_W], [NET_Y, NET_Y], color=FG, lw=3)
    for y in (NET_Y - 1.98, NET_Y + 1.98):
        ax.plot([0, COURT_W], [y, y], color=MUTED, lw=1, alpha=0.7)
    ax.plot([COURT_W / 2, COURT_W / 2], [0, NET_Y - 1.98], color=MUTED, lw=1, alpha=0.7)
    ax.plot([COURT_W / 2, COURT_W / 2], [NET_Y + 1.98, COURT_L], color=MUTED, lw=1, alpha=0.7)
    for i, (s, label) in enumerate(((a, "球员 A"), (b, "球员 B"))):
        p = s["points"]
        ax.scatter(p[::4, 0], p[::4, 1], s=8, color=COLORS[i], alpha=0.28, edgecolors="none")
        cx, cy = s["current"]
        ax.scatter([cx], [cy], s=110, color=COLORS[i], edgecolors=BG, lw=1.5, zorder=5)
        ax.annotate(label, (cx, cy), xytext=(8, 8), textcoords="offset points",
                    color=COLORS[i], fontsize=11, fontproperties=FONT_PROP)
        try:
            hull = ConvexHull(p)
            for simplex in hull.simplices:
                ax.plot(p[simplex, 0], p[simplex, 1], color=COLORS[i], lw=1, alpha=0.65)
        except Exception:
            pass
    ax.text(COURT_W / 2, NET_Y + 0.18, "网", ha="center", va="bottom", color=FG,
            fontsize=10, fontproperties=FONT_PROP)
    ax.set_xlim(-0.45, COURT_W + 0.45)
    ax.set_ylim(-0.3, COURT_L + 0.3)
    ax.set_aspect("equal")
    ax.set_title("站位俯视图 · 场地坐标（米）", color=FG, fontsize=13,
                 fontproperties=FONT_PROP, pad=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(MUTED)

    panel = fig.add_subplot(gs[0, 1])
    panel.set_facecolor(PANEL)
    panel.axis("off")
    panel.set_title("单打前后场站位 · 数据来自逐帧坐标", color=FG, fontsize=13,
                    fontproperties=FONT_PROP, loc="left", pad=10)
    rows = [
        ("指标", "球员 A", "球员 B"),
        ("跑动距离（米）", f'{a["dist"]:.1f}', f'{b["dist"]:.1f}'),
        ("覆盖面积（平方米）", f'{a["area"]:.1f}', f'{b["area"]:.1f}'),
        ("网前时间占比（距网 3 米内）", f'{a["front_pct"]:.0f}%', f'{b["front_pct"]:.0f}%'),
        ("前后场切换次数", str(a["switches"]), str(b["switches"])),
        ("击球次数", "未检测", "未检测"),
    ]
    y0 = 0.86
    for i, row in enumerate(rows):
        y = y0 - i * 0.145
        color = MUTED if i == 0 else FG
        panel.text(0.03, y, row[0], transform=panel.transAxes, color=color,
                   fontsize=11, fontproperties=FONT_PROP, va="center")
        panel.text(0.66, y, row[1], transform=panel.transAxes, color=COLORS[0] if i else MUTED,
                   fontsize=11, fontproperties=FONT_PROP, va="center")
        panel.text(0.86, y, row[2], transform=panel.transAxes, color=COLORS[1] if i else MUTED,
                   fontsize=11, fontproperties=FONT_PROP, va="center")
        if i > 0:
            panel.plot([0.02, 0.98], [y - 0.07, y - 0.07], transform=panel.transAxes,
                       color=MUTED, lw=0.5, alpha=0.35)

    tl = fig.add_subplot(gs[1, 1])
    tl.set_facecolor(PANEL)
    for i, (s, label) in enumerate(((a, "球员 A"), (b, "球员 B"))):
        y = 1 - i * 1.1
        state = s["state"]
        k = 0
        while k < len(state):
            j = k
            while j + 1 < len(state) and state[j + 1] == state[k]:
                j += 1
            if t[j] - t[k] >= MIN_HOLD_SEC:
                tl.barh(y, t[j] - t[k], left=t[k], height=0.55,
                        color=FRONT if state[k] else BACK, alpha=0.9)
            k = j + 1
        tl.text(-duration * 0.055, y, label, ha="right", va="center", color=COLORS[i],
                fontsize=10, fontproperties=FONT_PROP)
    tl.set_xlim(t[0], t[-1])
    tl.set_ylim(-0.6, 1.6)
    tl.set_title("站位时间线（橙 = 前场 / 紫 = 后场）", color=FG, fontsize=11,
                 fontproperties=FONT_PROP, loc="left", pad=8)
    tl.tick_params(colors=MUTED, labelsize=9)
    for spine in tl.spines.values():
        spine.set_color(MUTED)
    tl.set_xlabel("秒", color=MUTED, fontproperties=FONT_PROP)

    fig.suptitle("站位分析 · 单打前后场（数据来源：逐帧坐标）", color=FG,
                 fontsize=16, fontproperties=FONT_PROP, y=0.97)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    fig.savefig(dst, dpi=150, facecolor=BG)
    print("saved", dst)
    print(f"A: {a['dist']:.1f}m / {a['area']:.1f}m2 / 网前 {a['front_pct']:.0f}% / 切换 {a['switches']}")
    print(f"B: {b['dist']:.1f}m / {b['area']:.1f}m2 / 网前 {b['front_pct']:.0f}% / 切换 {b['switches']}")


if __name__ == "__main__":
    main()
