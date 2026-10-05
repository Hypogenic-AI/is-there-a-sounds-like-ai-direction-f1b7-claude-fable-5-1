"""Figures for the report. usage: python src/figures.py  -> figures/*.png"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("figures", exist_ok=True)
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e0"
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
         green="#008300", violet="#4a3aa7", red="#e34948", gray="#898781")
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.titlesize": 10, "lines.linewidth": 2, "lines.markersize": 5,
                     "legend.frameon": False, "figure.dpi": 150})
# colour follows the entity (direction) in every figure
DIRS = [("ai_read", "AI readout direction (diff-of-means)", C["blue"]),
        ("write_chat", "own-writing direction (self vs human, chat)", C["orange"]),
        ("chat_register", "chat register (instruct text - base text)", C["aqua"]),
        ("formality", "formality (Pavlick)", C["yellow"]),
        ("assistant_axis", "assistant-axis proxy", C["magenta"]),
        ("fluency", "fluency (low NLL)", C["green"]),
        ("ai_read_perp_all", "AI direction, confounds projected out", C["violet"]),
        ("random", "random, equal norm (mean of 2)", C["gray"])]


def load_table(run):
    T = pd.read_csv(f"results/table_{run}.csv")
    r = T[T.direction.isin(["random0", "random1"])].groupby("coef", as_index=False).mean(numeric_only=True)
    r["direction"] = "random"
    return pd.concat([T, r], ignore_index=True), T[T.cond == "none"].iloc[0], T[T.cond == "human_ref"].iloc[0]


# ------------------------------------------------------------------ Fig 1: readout by layer
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, k in zip(axes, ("instruct", "base")):
    R = json.load(open(f"results/readout_{k}.json"))["layers"]
    ls = sorted(int(l) for l in R)
    for key, lab, col in [("hape_gen_gpt-4o-2024-08-06", "HAP-E: GPT-4o text", C["blue"]),
                          ("hape_gen_llama-3-8B-Instruct", "HAP-E: Llama-3-8B-Instruct text", C["orange"]),
                          ("hape_gen_llama-3-8B", "HAP-E: Llama-3-8B base text", C["aqua"]),
                          ("hc3_dm", "HC3 (ChatGPT)", C["yellow"]),
                          ("raid_chat_dm", "RAID chat generators", C["magenta"]),
                          ("raid_nonchat_dm", "RAID non-chat generators", C["green"]),
                          ("mage_dm", "MAGE (mostly pre-chat generators)", C["violet"])]:
        ax.plot(ls, [R[str(l)][key] for l in ls], color=col, label=lab)
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.set_title(f"Reader: Llama-3.1-8B{'-Instruct' if k == 'instruct' else ' (base)'}")
    ax.set_xlabel("layer")
axes[0].set_ylabel("AUROC vs human text\n(projection on HAP-E diff-of-means direction)")
axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.suptitle("Readout: the direction separates chat-model text from human text far better than base-model text", x=0.02, ha="left")
fig.tight_layout(); fig.savefig("figures/fig1_readout_by_layer.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Fig 2: cosines at layer 12
G = json.load(open("results/geometry.json"))
fig, ax = plt.subplots(figsize=(7, 3.6))
names = [("chat_register", "chat register (Llama instruct - base text)"), ("write_chat", "own writing vs human (chat)"),
         ("ai_raid", "RAID chat-vs-human direction"), ("ai_hc3", "HC3 direction"),
         ("ai_base_gens", "base-LLM text vs human"), ("write_raw", "own writing vs human (raw continuation)"),
         ("ai_lr", "logistic-regression probe"), ("fluency", "fluency (low NLL)"),
         ("formality", "formality (Pavlick)"), ("length", "length (256 vs 48 tokens)"),
         ("assistant_axis", "assistant-axis proxy")]
Ri = json.load(open("results/readout_instruct.json"))["layers"]["12"]["cos_with_ai_read"]
Rb = json.load(open("results/readout_base.json"))["layers"]["12"]["cos_with_ai_read"]
y = np.arange(len(names))[::-1]
ax.barh(y + 0.19, [Ri.get(n, np.nan) for n, _ in names], height=0.34, color=C["blue"], label="Instruct model")
ax.barh(y - 0.19, [Rb.get(n, np.nan) for n, _ in names], height=0.34, color=C["orange"], label="Base model")
ax.set_yticks(y); ax.set_yticklabels([l for _, l in names], color=INK2)
ax.axvline(0, color=MUTED, lw=1); ax.set_xlim(-0.3, 1.0); ax.grid(axis="y", visible=False)
ax.set_xlabel("cosine with the AI readout direction (layer 12)")
ax.set_title("What the AI readout direction is aligned with", loc="left"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig("figures/fig2_cosines.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Fig 3: dose-response (main run)
T, none, human = load_table("main_test")
panels = [("desklib", "desklib detector P(AI)"), ("bino", "Binoculars score (higher = more AI)"),
          ("radar", "RADAR P(AI)"), ("ai_likelihood", "LLM judge AI-likelihood (0-100)"),
          ("coherence", "LLM judge coherence (0-100)"), ("on_task", "LLM judge on-task (0-100)")]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True)
for ax, (m, title) in zip(axes.ravel(), panels):
    for d, lab, col in DIRS:
        g = T[T.direction == d].sort_values("coef")
        if len(g) == 0:
            continue
        x = np.r_[g.coef[g.coef < 0], 0, g.coef[g.coef > 0]]
        yv = np.r_[g[m][g.coef < 0], none[m], g[m][g.coef > 0]]
        ax.plot(x, yv, color=col, marker="o", label=lab, lw=2.4 if d == "ai_read" else 1.6)
    ax.axhline(human[m], color=INK2, lw=1, ls="--")
    ax.text(ax.get_xlim()[1], human[m], " human refs", va="center", ha="left", fontsize=7, color=INK2)
    ax.set_title(title, loc="left")
for ax in axes[1]:
    ax.set_xlabel("steering coefficient (x residual norm)\n<- human side | AI side ->")
axes[0, 2].legend(loc="upper left", bbox_to_anchor=(1.12, 1.0))
fig.suptitle("Steering Llama-3.1-8B-Instruct: detector scores, coherence and task adherence (208 test prompts)", x=0.02, ha="left")
fig.tight_layout(); fig.savefig("figures/fig3_dose_response.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Fig 4: detector change vs coherence frontier
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
for ax, (m, title) in zip(axes, [("desklib_flag", "desklib: fraction flagged AI"),
                                 ("bino_flag", "Binoculars: fraction flagged AI"),
                                 ("ai_likelihood", "LLM judge AI-likelihood")]):
    for d, lab, col in DIRS:
        g = T[(T.direction == d) & (T.coef < 0)].sort_values("coef", ascending=False)
        if len(g) == 0:
            continue
        ax.plot(np.r_[none.coherence, g.coherence], np.r_[none[m], g[m]], color=col, marker="o", label=lab,
                lw=2.4 if d == "ai_read" else 1.6)
    for nm, mk in [("prompt_human", "*")]:
        p = T[T.cond == nm]
        if len(p):
            ax.scatter(p.coherence, p[m], marker=mk, s=160, color=C["red"], zorder=5, label='prompt: "write like a human"',
                       edgecolor=SURF, linewidth=1.5)
    ax.axhline(human[m], color=INK2, lw=1, ls="--")
    ax.axvline(none.coherence - 10, color=MUTED, lw=1, ls=":")
    ax.set_title(title, loc="left"); ax.set_xlabel("LLM judge coherence (0-100)"); ax.invert_xaxis()
axes[2].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig.suptitle("Steering toward the human side: detector score against coherence (flag threshold = 5% FPR on human refs; dashed = human refs; dotted = matched-coherence limit)", x=0.02, ha="left")
fig.tight_layout(); fig.savefig("figures/fig4_frontier.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Fig 5: base model raw continuation
try:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharex=True)
    runs = [("base_cont", "base model", "-"), ("instruct_cont", "instruct model (raw continuation)", "--")]
    for ax, (m, title) in zip(axes, [("desklib", "desklib P(AI)"), ("bino", "Binoculars (higher = AI)"),
                                     ("ai_likelihood", "judge AI-likelihood"), ("coherence", "judge coherence")]):
        for run, rl, ls in runs:
            try:
                Tb, nb, hb = load_table(run)
            except FileNotFoundError:
                continue
            for d, lab, col in [("ai_read", "AI readout direction", C["blue"]), ("chat_register", "chat register", C["aqua"]),
                                ("write_raw", "own-writing direction (raw)", C["orange"]), ("random", "random", C["gray"])]:
                g = Tb[Tb.direction == d].sort_values("coef")
                if len(g) == 0:
                    continue
                x = np.r_[g.coef[g.coef < 0], 0, g.coef[g.coef > 0]]
                yv = np.r_[g[m][g.coef < 0], nb[m], g[m][g.coef > 0]]
                ax.plot(x, yv, color=col, marker="o", ls=ls, label=f"{lab} | {rl}")
            ax.axhline(hb[m], color=INK2, lw=1, ls="--")
        ax.set_title(title, loc="left"); ax.set_xlabel("steering coefficient\n<- human side | AI side ->")
    axes[3].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
    fig.suptitle("Raw continuation (no chat template): steering the base model toward the AI side", x=0.02, ha="left")
    fig.tight_layout(); fig.savefig("figures/fig5_base_model.png", bbox_inches="tight"); plt.close(fig)
except FileNotFoundError as e:
    print("skip fig5", e)
print("figures written")
