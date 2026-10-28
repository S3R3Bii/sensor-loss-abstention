"""Decision and optimization figures built solely from retained experiment outputs."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
F = ROOT / "figures"
F.mkdir(exist_ok=True)
selection = json.loads((R / "selection.json").read_text())
winner = selection["selected_model"]
base = selection["clean_only_comparator"]
test = pd.read_csv(R / "test_metrics.csv")
search = pd.read_csv(R / "search.csv")
curve = pd.read_csv(R / "threshold_search.csv")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#b9c4d0",
        "text.color": "#17283d",
        "axes.labelcolor": "#17283d",
        "xtick.color": "#526273",
        "ytick.color": "#526273",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)
TEAL = "#087f8c"
ORANGE = "#d18337"
GRAY = "#a5b2c2"
INK = "#17283d"
conditions = ["intact", "acc_missing", "gyro_missing"]
labels = ["Intact sensors", "Accelerometer missing", "Gyroscope missing"]


def save(fig, name):
    fig.text(
        0.01,
        1.12,
        "N1 CORRECTED RETROSPECTIVE · UCI measured windows, controlled missing-sensor scenarios",
        fontsize=9,
        color="#526273",
    )
    fig.savefig(F / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
x = np.arange(3)
for offset, (name, label, color) in zip(
    [-0.18, 0.18],
    [
        (base, "Clean-only training", ORANGE),
        (winner, "Dropout-augmented training", TEAL),
    ],
):
    frame = test[test.model == name].set_index("condition").reindex(conditions)
    for ax, metric in zip(axes, ["coverage", "selective_accuracy"]):
        vals = frame[metric].to_numpy() * 100
        ax.bar(x + offset, vals, width=0.32, color=color, label=label)
        for xx, value in zip(x + offset, vals):
            if np.isfinite(value):
                ax.text(xx, value + 2, f"{value:.1f}", ha="center", fontsize=9)
        ax.set_xticks(x, labels, rotation=12, ha="right")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Percent")
axes[0].set_title("How much measurement is retained?", loc="left", fontweight="bold")
axes[0].legend(
    frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.30)
)
axes[1].axhline(90, color=GRAY, linestyle="--", lw=1)
axes[1].text(2.4, 91, "Validation target", ha="right", fontsize=9, color="#526273")
axes[1].set_title("Accuracy among retained windows", loc="left", fontweight="bold")
fig.suptitle(
    "A refusal policy must retain useful measurements and control errors",
    x=0.01,
    ha="left",
    fontsize=16,
    fontweight="bold",
    y=1.04,
)
fig.text(
    0.01,
    -0.17,
    "Test: source-held-out participants · Missing modalities are controlled interventions, not observed failures.\nConfidence cutoffs were fixed on validation participants. The 90% validation target is not a test guarantee.",
    fontsize=9,
    color="#526273",
)
fig.tight_layout(w_pad=3)
save(fig, "decision-summary")

fig, ax = plt.subplots(figsize=(11, 8))
order = search[search.status == "completed"].sort_values(
    ["participant_condition_macro_f1", "name"], ascending=[False, True]
)
names = [
    n.replace("logistic_C", "Logistic C=")
    .replace("_aug1", " · dropout training")
    .replace("_aug0", " · clean training")
    .replace("forest_d", "Forest depth ")
    .replace("_leaf", " / leaf ")
    for n in order.name
]
colors = [TEAL if n == winner else ORANGE if n == base else GRAY for n in order.name]
ax.barh(
    np.arange(len(order)),
    order.participant_condition_macro_f1,
    color=colors,
    height=0.68,
)
ax.set_yticks(np.arange(len(order)), names)
ax.invert_yaxis()
ax.set_xlim(0, 1.06)
ax.set_xlabel("Mean participant-condition macro F1 on validation")
for i, value in enumerate(order.participant_condition_macro_f1):
    ax.text(value + 0.012, i, f"{value:.3f}", va="center", fontsize=9)
ax.set_title(
    "Every declared model configuration, including weaker candidates",
    loc="left",
    fontweight="bold",
    pad=18,
)
fig.text(
    0.01,
    0.01,
    "17 configurations · Participant separation · Same three sensor conditions for every candidate\nC=10 is the largest tested logistic value. This finite search does not establish a global optimum.",
    fontsize=9,
    color="#526273",
)
fig.tight_layout(rect=[0, 0.07, 1, 1])
save(fig, "optimization-evidence")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
people = pd.read_csv(R / "participant_metrics.csv")
reliability = pd.read_csv(R / "reliability.csv")
for condition, label, color in zip(conditions, labels, [TEAL, ORANGE, "#6578a4"]):
    v = people[(people.model == winner) & (people.condition == condition)]
    axes[0].scatter(
        v.coverage * 100,
        v.selective_accuracy * 100,
        label=label,
        color=color,
        s=48,
        alpha=0.8,
    )
    rr = reliability[
        (reliability.model == winner)
        & (reliability.condition == condition)
        & (reliability.n >= 20)
    ]
    axes[1].plot(
        rr.confidence * 100, rr.accuracy * 100, "o-", label=label, color=color, lw=1.4
    )
axes[0].axhline(90, color=GRAY, linestyle="--", lw=1)
axes[0].set(
    xlim=(0, 105),
    ylim=(0, 105),
    xlabel="Retained windows per participant (%)",
    ylabel="Retained-window accuracy (%)",
)
axes[0].set_title(
    "People vary beneath the pooled result", loc="left", fontweight="bold"
)
axes[1].plot([0, 100], [0, 100], "--", color=GRAY, lw=1)
axes[1].set(
    xlim=(0, 100),
    ylim=(0, 100),
    xlabel="Mean model confidence (%)",
    ylabel="Observed window accuracy (%)",
)
axes[1].set_title("Confidence needs a reliability check", loc="left", fontweight="bold")
axes[0].legend(frameon=False, fontsize=9, loc="lower left")
fig.suptitle(
    "Inspect participant variation and probability reliability",
    x=0.01,
    ha="left",
    fontsize=16,
    fontweight="bold",
    y=1.04,
)
fig.text(
    0.01,
    -0.10,
    "Nine test participants; overlapping windows are not independent people. Selected model only.\nReliability uses fixed 10-point confidence bins with at least 20 windows; omitted small bins remain in the exported table.",
    fontsize=9,
    color="#526273",
)
fig.tight_layout(w_pad=3)
save(fig, "reliability-and-participants")
print("Three evidence figures exported.")

fig, ax = plt.subplots(figsize=(8, 5))
for name, label, color in [
    (base, "Clean-only comparator", ORANGE),
    (winner, "Dropout-augmented model", TEAL),
]:
    frame = curve[curve.model == name].sort_values("coverage")
    ax.plot(
        frame.coverage * 100,
        frame.selective_accuracy * 100,
        "o-",
        color=color,
        label=label,
        lw=1.6,
    )
    chosen = frame[np.isclose(frame.threshold, selection["thresholds"][name])]
    ax.scatter(
        chosen.coverage * 100,
        chosen.selective_accuracy * 100,
        marker="s",
        s=110,
        color=color,
        edgecolor=INK,
        zorder=3,
    )
ax.axhline(90, color=GRAY, ls="--", lw=1)
ax.set(
    xlim=(0, 105),
    ylim=(0, 105),
    xlabel="Validation windows retained (%)",
    ylabel="Accuracy among retained windows (%)",
)
ax.set_title(
    "The chosen cutoff depends on the validation mixture",
    loc="left",
    fontweight="bold",
    pad=17,
)
ax.legend(frameon=False, fontsize=10, loc="lower left")
fig.text(
    0.01,
    -0.06,
    "Nine prespecified cutoffs · Six validation participants · Three equally represented sensor conditions\nSquares identify frozen choices. Validation-only selection; no guarantee for a new participant or failure pattern.",
    fontsize=9,
    color="#526273",
)
fig.tight_layout()
save(fig, "abstention-frontier")
print("Validation coverage-accuracy frontier exported.")
