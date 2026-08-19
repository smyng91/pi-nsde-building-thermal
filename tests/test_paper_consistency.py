"""Guard: manuscript macros, figure CSVs, and seed-0 ensemble must agree."""

from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TEX = PAPER / "generated_numbers.tex"
FIGS = PAPER / "figures"


def _macros() -> dict[str, str]:
    text = TEX.read_text(encoding="utf-8")
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z][A-Za-z0-9]*)\}\{([^}]*)\}", text))


def test_figure_csv_matches_heating_and_cooling_macros():
    macros = _macros()
    heat = np.loadtxt(FIGS / "fig2_params.csv", delimiter=",", skiprows=1)
    cool = np.loadtxt(FIGS / "fig4_params_cool.csv", delimiter=",", skiprows=1)
    assert abs(heat[0, 1] - float(macros["KnownC"])) < 5e-3
    assert abs(heat[0, 3] - float(macros["UnknownC"])) < 5e-3
    assert abs(heat[1, 1] - float(macros["KnownR"])) < 5e-2
    assert abs(heat[1, 3] - float(macros["UnknownR"])) < 5e-2
    assert abs(heat[2, 3] - float(macros["UnknownQ"])) < 5e-2
    assert abs(cool[0, 1] - float(macros["CoolKnownC"])) < 5e-3
    assert abs(cool[0, 3] - float(macros["CoolUnknownC"])) < 5e-3
    assert abs(cool[1, 1] - float(macros["CoolKnownR"])) < 5e-2
    assert abs(cool[1, 3] - float(macros["CoolUnknownR"])) < 5e-2
    assert abs(cool[2, 3] - float(macros["CoolUnknownQ"])) < 5e-2


def test_seed0_matches_unknown_tables():
    import json

    payload = json.loads((FIGS / "seed_study.json").read_text(encoding="utf-8"))
    macros = _macros()
    heat = next(r for r in payload["rows"] if r["season"] == "heating" and r["seed"] == 0)
    cool = next(r for r in payload["rows"] if r["season"] == "cooling" and r["seed"] == 0)
    assert abs(heat["C"] - float(macros["UnknownC"])) < 0.05
    assert abs(cool["C"] - float(macros["CoolUnknownC"])) < 0.05


def test_stale_learned_beta_macros_are_gone():
    macros = _macros()
    for key in (
        "KnownBeta",
        "UnknownBeta",
        "KnownBetarel",
        "UnknownBetarel",
        "CoolKnownBeta",
        "CoolUnknownBeta",
    ):
        assert key not in macros
    assert macros["TrueBeta"] == "120"
    assert macros["FixedBeta"] == "120"


def test_zscore_macros_match_figure_csv():
    macros = _macros()
    heat = np.loadtxt(FIGS / "fig2_params.csv", delimiter=",", skiprows=1)
    cool = np.loadtxt(FIGS / "fig4_params_cool.csv", delimiter=",", skiprows=1)
    true_c, true_r, true_q = heat[0, 0], heat[1, 0], heat[2, 0]
    pairs = (
        (heat[0, 1], heat[0, 2], true_c, "KnownCz"),
        (heat[1, 1], heat[1, 2], true_r, "KnownRz"),
        (heat[0, 3], heat[0, 4], true_c, "UnknownCz"),
        (heat[1, 3], heat[1, 4], true_r, "UnknownRz"),
        (heat[2, 3], heat[2, 4], true_q, "UnknownQz"),
        (cool[0, 1], cool[0, 2], true_c, "CoolKnownCz"),
        (cool[1, 1], cool[1, 2], true_r, "CoolKnownRz"),
        (cool[0, 3], cool[0, 4], true_c, "CoolUnknownCz"),
        (cool[1, 3], cool[1, 4], true_r, "CoolUnknownRz"),
        (cool[2, 3], cool[2, 4], true_q, "CoolUnknownQz"),
    )
    for mean, sd, truth, key in pairs:
        z = abs(mean - truth) / sd
        assert abs(z - float(macros[key])) < 0.08, (key, z, macros[key])


def test_manuscript_laplace_coverage_matches_csv():
    """Prose must not claim 95% coverage the figure CSVs contradict."""
    heat = np.loadtxt(FIGS / "fig2_params.csv", delimiter=",", skiprows=1)
    cool = np.loadtxt(FIGS / "fig4_params_cool.csv", delimiter=",", skiprows=1)
    text = (PAPER / "main.tex").read_text(encoding="utf-8")

    def covers(mean, sd, truth) -> bool:
        return abs(mean - truth) <= 1.96 * sd

    assert not covers(heat[0, 1], heat[0, 2], heat[0, 0])
    assert not covers(heat[1, 1], heat[1, 2], heat[1, 0])
    assert not covers(heat[0, 3], heat[0, 4], heat[0, 0])
    assert not covers(heat[1, 3], heat[1, 4], heat[1, 0])
    assert not covers(heat[2, 3], heat[2, 4], heat[2, 0])
    assert covers(cool[0, 1], cool[0, 2], cool[0, 0])
    assert not covers(cool[1, 1], cool[1, 2], cool[1, 0])
    assert not covers(cool[0, 3], cool[0, 4], cool[0, 0])
    assert covers(cool[1, 3], cool[1, 4], cool[1, 0])
    assert not covers(cool[2, 3], cool[2, 4], cool[2, 0])

    assert r"interval for $R$ contains plant $R^*$" not in text
    assert r"while $R$ is covered" not in text
    assert "happen to contain plant" not in text
    assert r"wide $R$ interval still contains" not in text
    assert r"\UnknownCz" in text
    assert r"\CoolUnknownQz" in text
    # Prose that previously contradicted the code, CSVs, or the paper's own UQ caveats.
    for phrase in (
        "does not cancel on this winter",
        "in-phase with daytime solar",
        r"once $\beta=\TrueBeta$",
        "weak prior can slide",
        r"95\% interval for $R$",
        "mathematically straightforward",
        r"\mathcal{R}_\mathrm{id}=0$ at the start",
        r"driven to \KnownRhoDT",
        "structural identifiability of $R$",
        r"collinear $UA",
        r"over 30\% of global",
        "known a priori",
        "better on winter known HVAC",
        "typical thermostat week",
        r"to $\KnownCrel$\% and $\KnownRrel$\% of plant truth",
        r"brings MAP $C$ to $\KnownCrel$\% (heating)",
    ):
        assert phrase not in text, phrase
    assert r"to within $\KnownCrel$" in text
    assert "better on summer known HVAC" in text
    assert r"diag(0.6^2,1.0^2)" in text


def test_ablation_rmse_macros_match_prose():
    """PIN-SDE has lower winter-known holdout RMSE; gray-box is lower only in summer known."""
    macros = _macros()
    assert float(macros["GbKnownHoldRMSE"]) > float(macros["KnownHoldRMSE"])
    assert float(macros["CoolGbKnownHoldRMSE"]) < float(macros["CoolKnownHoldRMSE"])
    assert float(macros["GbUnknownHoldRMSE"]) > float(macros["UnknownHoldRMSE"])
    assert float(macros["CoolGbUnknownHoldRMSE"]) > float(macros["CoolUnknownHoldRMSE"])


def test_seed_study_rmse_below_one_kelvin():
    import json

    payload = json.loads((FIGS / "seed_study.json").read_text(encoding="utf-8"))
    rmses = [float(r["holdout_rmse"]) for r in payload["rows"]]
    assert rmses and max(rmses) < 1.0
    text = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert r"below $1\,\mathrm{K}$ on every seed" in text


def test_holdout_csv_rmse_matches_macros():
    macros = _macros()
    pairs = (
        (FIGS / "fig1_holdout.csv", "KnownHoldRMSE", "UnknownHoldRMSE"),
        (FIGS / "fig5_holdout_cool.csv", "CoolKnownHoldRMSE", "CoolUnknownHoldRMSE"),
    )
    for path, kn_key, un_key in pairs:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        t_in, kn, un = data[:, 1], data[:, 2], data[:, 3]
        mask = np.isfinite(kn)
        kn_rmse = float(np.sqrt(np.mean((kn[mask] - t_in[mask]) ** 2)))
        un_rmse = float(np.sqrt(np.mean((un[mask] - t_in[mask]) ** 2)))
        assert abs(kn_rmse - float(macros[kn_key])) < 5e-3, (path, kn_rmse, macros[kn_key])
        assert abs(un_rmse - float(macros[un_key])) < 5e-3, (path, un_rmse, macros[un_key])


def test_weather_macros_match_seed0_twins():
    from pi_nsde_building_thermal.synthetic import SyntheticConfig, generate_synthetic_building

    macros = _macros()
    for season, prefix in (("heating", ""), ("cooling", "Cool")):
        arrays = generate_synthetic_building(
            SyntheticConfig(days=7, seed=0, hvac_mode=season)
        ).arrays
        ta = np.asarray(arrays.t_out_c)
        tin = np.asarray(arrays.t_in_c)
        ghi = np.asarray(arrays.ghi_w_m2)
        dT = float(np.mean(ta - tin))
        keys = {
            f"{prefix}TaMin" if prefix else "TaMin": (float(np.min(ta)), 1),
            f"{prefix}TaMax" if prefix else "TaMax": (float(np.max(ta)), 1),
            f"{prefix}TaMean" if prefix else "TaMean": (float(np.mean(ta)), 1),
            f"{prefix}GhiMean" if prefix else "GhiMean": (float(np.mean(ghi)), 0),
        }
        for key, (val, nd) in keys.items():
            assert abs(float(f"{val:.{nd}f}") - float(macros[key])) < 1e-9, (key, val, macros[key])
        mean_key = f"{prefix}MeanDT" if prefix else "MeanDT"
        nd = 2 if abs(dT) < 10 else 1
        assert abs(float(f"{dT:.{nd}f}") - float(macros[mean_key])) < 1e-9, (mean_key, dT, macros[mean_key])


def test_train_csv_uses_symmetric_delta_method_intervals():
    src = (ROOT / "examples" / "train_csv.py").read_text(encoding="utf-8")
    assert "est - 1.96 * sd" in src
    assert "math.exp(-1.96" not in src
