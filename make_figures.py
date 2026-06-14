#!/usr/bin/env python3
"""Regenerate Figures 1-4 for the Collaboration Code resubmission from the
verified bundle data. Neutral hypothesis titles; no 'Figure N' burned in;
no editorial annotations; task-level DV labeled 'usage intensity' (not adoption).

Reads the built datasets from ./data/ (produced by 01_build_dataset.py) and
writes Figure 1-4.jpg next to this script. Paths are resolved relative to the
script location so the figures regenerate on any machine."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_HERE = os.path.dirname(os.path.abspath(__file__))
BUN = _HERE
OUT = _HERE

t = pd.read_csv(f"{BUN}/data/task_level_dataset.csv")
c = pd.read_csv(f"{BUN}/data/cluster_level_dataset.csv")

def linreg(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x); xb, yb = x.mean(), y.mean()
    sxx = ((x-xb)**2).sum()
    b1 = ((x-xb)*(y-yb)).sum()/sxx
    b0 = yb - b1*xb
    yhat = b0 + b1*x
    ss_res = ((y-yhat)**2).sum(); ss_tot = ((y-yb)**2).sum()
    r2 = 1 - ss_res/ss_tot
    r = np.corrcoef(x, y)[0,1]
    return b0, b1, r2, r, n

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "axes.edgecolor": "#666666",
})
SCAT = dict(s=14, alpha=0.35, color="#4C72B0", edgecolors="none")
LINE = dict(color="#C44E52", lw=2.4, zorder=5)
BOX = dict(boxstyle="round,pad=0.5", fc="#FBF6E9", ec="#B59A52", lw=1.0)

# ---------------- FIGURE 1 : H1 (TAM), two panels ----------------
d = t[["augmentation_index","task_conversation_share"]].dropna()
x, y = d["augmentation_index"].values, d["task_conversation_share"].values
b0, b1, r2, _, n = linreg(x, y)
ylog = np.log(y)
lb0, lb1, lr2, _, _ = linreg(x, ylog)
assert abs(b1-0.0596) < 0.003, ("H1 linear b", b1)
assert abs(lb1-2.8144) < 0.02, ("H1 log b", lb1)
assert abs(r2-0.011) < 0.004 and abs(lr2-0.399) < 0.01, (r2, lr2)

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.9))
fig.suptitle("H1 (Technology Acceptance Model): Augmentation Index and Task-Level AI Usage Intensity",
             fontsize=12.5, fontweight="bold", y=0.99)
xs = np.linspace(x.min(), x.max(), 100)
ax[0].scatter(x, y, **SCAT)
ax[0].plot(xs, b0+b1*xs, **LINE)
ax[0].set_title("Linear scale", fontsize=11.5)
ax[0].set_xlabel("Augmentation Index")
ax[0].set_ylabel("AI Usage Intensity (task_conversation_share)")
ax[0].text(0.04, 0.96, f"OLS  β = 0.060***\nSE = 0.004\nR² = .011\nN = {n:,}",
           transform=ax[0].transAxes, va="top", ha="left", fontsize=10, bbox=BOX)
ax[1].scatter(x, ylog, **SCAT)
ax[1].plot(xs, lb0+lb1*xs, **LINE)
ax[1].set_title("Log-linear scale", fontsize=11.5)
ax[1].set_xlabel("Augmentation Index")
ax[1].set_ylabel("ln(task_conversation_share)")
ax[1].text(0.04, 0.96, f"Log-OLS  β = 2.814***\nSE = 0.055\nR² = .399\nN = {n:,}",
           transform=ax[1].transAxes, va="top", ha="left", fontsize=10, bbox=BOX)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(f"{OUT}/Figure 1.jpg", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------------- FIGURE 2 : H2a (PMT) ----------------
d2 = t[["risk_management_index","task_conversation_share"]].dropna()
x2, y2 = d2["risk_management_index"].values, d2["task_conversation_share"].values
b0_2, b1_2, r2_2, _, n2 = linreg(x2, y2)
assert abs(b1_2-0.6649) < 0.01 and n2 == 3364, (b1_2, n2)

fig, ax = plt.subplots(figsize=(8.4, 5.0))
ax.set_title("H2a (Protection Motivation Theory): Risk-Management Behaviors and Task-Level AI Usage Intensity",
             fontsize=11.6, fontweight="bold")
ax.scatter(x2, y2, **SCAT)
xs2 = np.linspace(x2.min(), x2.max(), 100)
ax.plot(xs2, b0_2+b1_2*xs2, **LINE)
ax.set_xlabel("Risk Management Index (feedback_loop + validation)")
ax.set_ylabel("AI Usage Intensity (task_conversation_share)")
ax.text(0.03, 0.97, "OLS  β = 0.665***\nSE = 0.163\nR² = .085\nN = 3,364",
        transform=ax.transAxes, va="top", ha="left", fontsize=10, bbox=BOX)
fig.tight_layout()
fig.savefig(f"{OUT}/Figure 2.jpg", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------------- FIGURE 3 : H3 (SET), cluster level ----------------
d3 = c[["cluster_augmentation_index","cluster_thinking_fraction"]].dropna()
x3, y3 = d3["cluster_augmentation_index"].values, d3["cluster_thinking_fraction"].values
b0_3, b1_3, r2_3, r3, n3 = linreg(x3, y3)
assert abs(b1_3+0.0611) < 0.003 and n3 == 508 and abs(r3+0.327) < 0.01, (b1_3, n3, r3)

fig, ax = plt.subplots(figsize=(8.4, 5.0))
ax.set_title("H3 (Social Exchange Theory): Collaborative Augmentation and Cluster-Level Extended Thinking",
             fontsize=11.6, fontweight="bold")
ax.scatter(x3, y3, **SCAT)
xs3 = np.linspace(x3.min(), x3.max(), 100)
ax.plot(xs3, b0_3+b1_3*xs3, **LINE)
ax.set_xlabel("Cluster Augmentation Index")
ax.set_ylabel("Cluster Thinking Fraction")
ax.text(0.97, 0.96, "OLS  β = −0.0611***\nSE = 0.009\nr = −0.327\nN = 508",
        transform=ax.transAxes, va="top", ha="right", fontsize=10, bbox=BOX)
fig.tight_layout()
fig.savefig(f"{OUT}/Figure 3.jpg", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------------- FIGURE 4 : H5 (TAM x SET) mediation diagram ----------------
fig, ax = plt.subplots(figsize=(9.2, 5.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("H5 (TAM × SET): Cross-Level Mediation of Augmentation and Usage Intensity",
             fontsize=12.5, fontweight="bold")

def box(cx, cy, w, h, text, fc="#EAF1F8"):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec="#34495E", lw=1.6))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=10.5, fontweight="bold")

# level guide line
ax.axhline(6.0, xmin=0.04, xmax=0.96, color="#999999", ls=(0,(4,4)), lw=1)
ax.text(0.3, 6.2, "Cluster level", fontsize=9, style="italic", color="#777777")
ax.text(0.3, 5.55, "Task level", fontsize=9, style="italic", color="#777777")

box(5.0, 8.1, 3.0, 1.25, "Cluster Extended Thinking\n(Cluster Level)")
box(1.95, 3.3, 2.7, 1.25, "Augmentation Index\n(Task Level)")
box(8.05, 3.3, 2.7, 1.25, "Task Conversation Share\n(Task Level)")

def arrow(x1, y1, x2, y2, color):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=18, lw=2.2, color=color, shrinkA=2, shrinkB=2))
arrow(2.7, 3.95, 4.1, 7.5, "#C0392B")   # a
arrow(5.9, 7.5, 7.3, 3.95, "#27AE60")   # b
ax.add_patch(FancyArrowPatch((3.3, 3.3), (6.7, 3.3), arrowstyle="-|>",
             mutation_scale=18, lw=2.2, color="#C0392B", ls=(0,(5,3)), shrinkA=2, shrinkB=2))  # c'
ax.text(3.0, 6.1, "a = −0.017**\n(p = .007)", fontsize=10, color="#C0392B", ha="center")
ax.text(7.0, 6.1, "b = 6.19***\n(p < .001)", fontsize=10, color="#27AE60", ha="center")
ax.text(5.0, 3.62, "c′ = −0.90*** (direct effect)", fontsize=10, color="#C0392B", ha="center")

ax.text(5.0, 1.35,
        "Total effect (c) = −1.01***        Indirect effect (a × b) = −0.105\n"
        "Bias-corrected bootstrap 95% CI [−0.206, −0.029], 5,000 resamples",
        ha="center", va="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", fc="#F4F6F7", ec="#AAB7B8", lw=1.0))
ax.text(5.0, 0.2, "* p < .05   ** p < .01   *** p < .001", ha="center", fontsize=8.5, color="#555555")
fig.tight_layout()
fig.savefig(f"{OUT}/Figure 4.jpg", dpi=300, bbox_inches="tight"); plt.close(fig)

print("OK")
print(f"F1 linear b={b1:.4f} R2={r2:.4f} | log b={lb1:.4f} R2={lr2:.4f} | N={n}")
print(f"F2 b={b1_2:.4f} R2={r2_2:.4f} N={n2}")
print(f"F3 b={b1_3:.4f} R2={r2_3:.4f} r={r3:.4f} N={n3}")
