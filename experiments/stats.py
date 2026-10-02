"""
Statistics for the report. Reads the saved results and writes Markdown tables
(ready to paste into README.md) to results/tables/, plus two CSV files that
plots.py uses.

Run from the repository root (after experiments/run.py has produced the results):
    python experiments/stats.py

Tests used (all non-parametric, because line counts are very skewed):
  * Wilcoxon signed-rank (paired): two players on the SAME 20 test games.
  * Mann-Whitney U (unpaired): two groups of values, e.g. 10 runs vs 10 runs.
  * Kruskal-Wallis: are 3-4 groups different at all?
  * Holm correction when several pairwise tests are made at once.
  * Effect size A12 (Vargha-Delaney): chance that a value from the first group
    beats a value from the second (0.5 = no difference).
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr, wilcoxon

import analysis as A


def summary(lines):
    lines = np.asarray(lines)
    q1, med, q3 = np.percentile(lines, [25, 50, 75])
    return {"mean": lines.mean(), "median": med, "q1": q1, "q3": q3, "max": lines.max()}


def md_table(df, floatfmt="{:,.1f}"):
    """DataFrame -> Markdown table (numbers formatted, strings kept)."""
    def cell(v):
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return "-"
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def write(name, title, parts):
    A.TABLES.mkdir(parents=True, exist_ok=True)
    text = f"# {title}\n\n" + "\n\n".join(parts) + "\n"
    (A.TABLES / name).write_text(text, encoding="utf-8")
    print(f"\n{text}")


def games_of(df, player):
    """Lines per game of one player, sorted by game seed (so two players line up game by game)."""
    return df[df.player == player].sort_values("game_seed").lines.values


# ---------------------------------------------------------------- RQ1 / RQ4
def player_table(df, players, reference="Hand-tuned"):
    """One row per player: summary of its test games + paired comparison with the reference."""
    ref = games_of(df, reference)
    rows, pvals = [], []
    for label, names in players:
        lines = np.concatenate([games_of(df, n) for n in names])
        row = {"player": label, **summary(lines)}
        if label == reference:
            row.update({"wins vs hand": "-", "p vs hand": "-", "A12 vs hand": "-"})
        elif len(names) == 1:                                  # paired: same 20 games
            x = games_of(df, names[0])
            p = wilcoxon(x, ref).pvalue if np.any(x != ref) else 1.0
            row.update({"wins vs hand": f"{(x > ref).sum()}/20", "p vs hand": p, "A12 vs hand": a12_str(x, ref)})
            pvals.append((len(rows), p))
        else:                                                  # several players pooled: unpaired
            p = mannwhitneyu(lines, ref).pvalue
            row.update({"wins vs hand": "pooled", "p vs hand": p, "A12 vs hand": a12_str(lines, ref)})
            pvals.append((len(rows), p))
        if "reached_cap" in df and df[df.player.isin(names)].reached_cap.any():
            row["player"] += f" ({int(df[df.player.isin(names)].reached_cap.sum())} game(s) hit the cap)"
        rows.append(row)
    # Holm correction over all comparisons with the reference
    adjusted = A.holm([p for _, p in pvals])
    for (i, _), p in zip(pvals, adjusted):
        rows[i]["p vs hand"] = A.fmt_p(p)
    out = pd.DataFrame(rows)
    return out.rename(columns={"q1": "25%", "q3": "75%", "p vs hand": "p vs hand (Holm)"})


def a12_str(x, y):
    return f"{A.a12(x, y):.2f}"


def rq1_tables():
    e1 = A.per_game("E1_baselines.csv")
    default_players = sorted(p for p in e1.player.unique() if p.startswith("GA (default"))
    t1 = player_table(e1, [
        ("Random", ["Random"]), ("Literature", ["Literature"]), ("Hand-tuned", ["Hand-tuned"]),
        ("GA best of default runs", ["GA best (default)"]),
        ("GA, all 10 default runs", default_players),
    ])
    write("rq1_training_board.md", "RQ1 - lines cleared on the 10x10 training board (E1, 20 test games)", [
        md_table(t1),
        "Paired Wilcoxon test (same 20 games) for single players, Mann-Whitney U for the pooled "
        "10 runs (200 games); p-values Holm-corrected over the 4 comparisons. "
        "A12 = probability that the player beats hand-tuned in a random game pair.",
    ])

    e5 = A.per_game("E5_generalization.csv")
    top = sorted(p for p in e5.player.unique() if p.startswith("GA top"))
    default_players = sorted(p for p in e5.player.unique() if p.startswith("GA (default"))
    t5 = player_table(e5, [
        ("Random", ["Random"]), ("Literature", ["Literature"]), ("Hand-tuned", ["Hand-tuned"]),
        ("GA best of default runs", ["GA best (default)"]),
        ("GA, all 10 default runs", default_players),
        *[(p.split(":")[0] + ": " + p.split("GA (")[1].rstrip(")"), [p]) for p in top],
    ])
    hand_median = summary(games_of(e5, "Hand-tuned"))["median"]
    t5["median / hand"] = [f"{m / hand_median:.1f}x" for m in t5["median"]]
    write("rq1_rq4_standard_board.md", "RQ1 + RQ4 - lines cleared on the standard 10x20 board (E5, 20 test games)", [
        md_table(t5, "{:,.0f}"),
        "Same players as on the training board plus the 3 champions with the best VALIDATION "
        "score out of all 80 GA runs (chosen without looking at test games). Cap: 250,000 pieces. "
        "Tests as above (Holm over 6 comparisons).",
    ])
    return e1, e5


def rq4_table(e1, e5):
    small = pd.concat([A.per_game("E2-E4_champions.csv"), e1[e1.player.isin(["Hand-tuned", "Literature"])]])
    rows = []
    for player in e5.player.unique():
        key = player.split(": ", 1)[1] if player.startswith("GA top") else player
        if key == "GA best (default)" or key == "Random" or key not in set(small.player):
            continue
        s, b = games_of(small, key), games_of(e5, player)
        rows.append({"player": player.replace("GA top-", "top-"), "10x10 mean": s.mean(), "10x10 median": np.median(s),
                     "10x20 mean": b.mean(), "10x20 median": np.median(b), "10x20 / 10x10 (means)": b.mean() / s.mean()})
    t = pd.DataFrame(rows).sort_values("10x20 median", ascending=False)
    rho, p = spearmanr(t["10x10 mean"], t["10x20 mean"])
    rho_m, p_m = spearmanr(t["10x10 median"], t["10x20 median"])
    write("rq4_generalization.md", "RQ4 - the same players on the training board (10x10) and the standard board (10x20)", [
        md_table(t, "{:,.1f}"),
        f"Rank correlation between the two boards over these {len(t)} players: Spearman rho = {rho:.2f} "
        f"(p = {A.fmt_p(p)}) for means, {rho_m:.2f} (p = {A.fmt_p(p_m)}) for medians.",
    ])
    return t


# ------------------------------------------------------------- RQ2 / RQ3
def run_metrics(group):
    """One row per run with the numbers compared in E2-E4."""
    h = A.load_histories(group)
    pop = A.POPULATION.get(group, 50)
    rows = []
    for seed, run in h.groupby("seed"):
        champ = run.champion_validation.values
        rows.append({
            "seed": seed,
            "final_pop_mean": run.mean_fitness.tail(10).mean(),
            "gens_to_90": int(np.argmax(champ >= 0.9 * champ[-1])),
            "late_diversity": run.diversity.iloc[25:].mean(),
            "minutes": run.elapsed_sec.iloc[-1] / 60,
            "training_games": pop * A.GAMES_PER_EVAL * len(run),
        })
    return pd.DataFrame(rows)


def variants_tables(champions):
    parts = []
    for exp, variants in A.EXPERIMENTS.items():
        per_variant, rows = {}, []
        for group, label in variants:
            c = champions[champions.group == group].sort_values("seed").reset_index(drop=True)
            m = run_metrics(group)
            per_variant[label] = c.join(m.drop(columns="seed"))
            v = per_variant[label]
            rows.append({
                "variant": label + (" (default)" if group == "default" else ""),
                "test mean": v.test_mean.mean(), "SD over runs": v.test_mean.std(),
                "median of runs": v.test_mean.median(), "validation": v.validation.mean(),
                "pop. mean (last 10 gens)": v.final_pop_mean.mean(), "gens to 90%": v.gens_to_90.mean(),
                "diversity (gens 25-49)": v.late_diversity.mean(), "min / run": v.minutes.mean(),
            })
        table = pd.DataFrame(rows)

        # Kruskal-Wallis across all variants of this experiment, for several measures
        kw = {}
        for col, name in [("test_mean", "test score"), ("validation", "validation"),
                          ("final_pop_mean", "population mean"), ("gens_to_90", "gens to 90%"),
                          ("late_diversity", "diversity")]:
            kw[name] = kruskal(*[v[col] for v in per_variant.values()]).pvalue
        kw_line = "Kruskal-Wallis across variants: " + ", ".join(f"{k} p = {A.fmt_p(p)}" for k, p in kw.items())

        # every variant vs the default (paired by seed: same initial population and games)
        default_label = [lab for g, lab in variants if g == "default"][0]
        base = per_variant[default_label]
        comp, pvals = [], []
        for label, v in per_variant.items():
            if label == default_label:
                continue
            p_mwu = mannwhitneyu(v.test_mean, base.test_mean).pvalue
            p_wil = wilcoxon(v.test_mean, base.test_mean).pvalue if np.any(v.test_mean != base.test_mean) else 1.0
            pvals.append(p_mwu)
            comp.append({"comparison": f"{label} vs {default_label}", "test-score A12": f"{A.a12(v.test_mean, base.test_mean):.2f}",
                         "Mann-Whitney p": p_mwu, "paired Wilcoxon p": A.fmt_p(p_wil),
                         "diversity A12": f"{A.a12(v.late_diversity, base.late_diversity):.2f}",
                         "diversity p": A.fmt_p(mannwhitneyu(v.late_diversity, base.late_diversity).pvalue)})
        for row, p in zip(comp, A.holm(pvals)):
            row["Mann-Whitney p"] = f"{A.fmt_p(row['Mann-Whitney p'])} (Holm {A.fmt_p(p)})"
        parts += [f"## {exp}", md_table(table, "{:,.2f}"), kw_line, md_table(pd.DataFrame(comp))]

        if exp == "E2":   # equal-compute view: champion validation after the same number of training games
            eq = []
            for games in (3_000, 7_500):
                row = {"training games": f"{games:,}"}
                for group, label in variants:
                    gen = games // (A.POPULATION.get(group, 50) * A.GAMES_PER_EVAL) - 1
                    h = A.load_histories(group)
                    row[label] = h[h.generation == gen].champion_validation.mean() if gen < 50 else float("nan")
                eq.append(row)
            parts += ["Equal compute - average champion validation score after the same number of training games:",
                      md_table(pd.DataFrame(eq), "{:,.1f}")]

    parts.append("Test score = each run's champion on the 20 test games (10x10), averaged per run; "
                 "10 runs per variant. Runs with the same seed share their initial population and "
                 "training games, so the Wilcoxon test pairs them.")
    write("rq2_rq3_variants.md", "RQ2 + RQ3 - GA settings (E2 population size, E3 mutation rate, E4 operators)", parts)


# --------------------------------------------------------- weights & styles
def weights_and_styles(champions):
    eff = pd.DataFrame([A.effective_weights(w) for w in champions.weights])
    champions = champions.join(eff)
    cols = ["line_value", "bumpiness", "max_height", "wells"]
    res = [spearmanr(champions[c], champions.test_mean) for c in cols]
    adjusted = A.holm([r.pvalue for r in res])
    corr = pd.DataFrame({"effective weight": cols, "Spearman rho with test score": [f"{r.statistic:.2f}" for r in res],
                         "p (Holm)": [A.fmt_p(p) for p in adjusted]})

    e5 = A.per_game("E5_generalization.csv")
    rows = []
    for name, weights, source in A.style_players():
        w = np.asarray(weights) / np.linalg.norm(weights)
        prof = A.style_profile(weights)
        key = {"Hand-tuned": "Hand-tuned"}.get(name) or next(p for p in e5.player.unique() if source in p)
        rows.append({"player": name, "run": source,
                     **{f"w {f}": round(x, 2) for f, x in zip(A.FEATURES, w)},
                     **{f"eff. {k}": round(v, 2) for k, v in A.effective_weights(weights).items()},
                     **{k: round(v, 2) for k, v in prof.items()},
                     "10x20 median lines": np.median(games_of(e5, key))})
    styles = pd.DataFrame(rows)
    styles.to_csv(A.TABLES / "style_profiles.csv", index=False)
    champions.drop(columns="weights").assign(weights=[json.dumps([round(float(x), 6) for x in w])
                                                      for w in champions.weights]) \
             .to_csv(A.TABLES / "champions.csv", index=False)

    write("weights_and_styles.md", "Evolved weights and playing styles", [
        "## Raw weights (normalised to length 1) and effective weights",
        "Effective weights remove the redundant 'holes' weight and are measured in cells of stack "
        "height (see analysis.effective_weights): line_value = net reward for clearing one line.",
        md_table(styles[["player", "run", *[f"w {f}" for f in A.FEATURES], "eff. line_value", "eff. bumpiness", "eff. max_height", "eff. wells"]], "{:,.2f}"),
        "## How they play (10x20 board, first 1,000 pieces of 10 test games, averaged after every move)",
        md_table(styles[["player", "avg_height", "max_height", "holes", "bumpiness", "multi_line_share",
                         "lines_per_piece", "10x20 median lines"]], "{:,.2f}"),
        "## Which effective weights go with a good champion? (all 80 champions, test score on 10x10)",
        md_table(corr),
    ])


def main():
    e1, e5 = rq1_tables()
    rq4_table(e1, e5)
    champions = A.champion_table()
    variants_tables(champions)
    weights_and_styles(champions)
    print(f"Tables written to {A.TABLES}")


if __name__ == "__main__":
    main()
