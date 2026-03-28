import numpy as np
import pandas as pd


FEATURE_COLS = [
    "elo_diff",
    "adjEM_diff",
    "eFG_diff",
    "TO_diff",
    "OR_diff",
    "FTRate_diff",
    "opp_eFG_diff",
    "opp_TO_diff",
    "opp_OR_diff",
    "massey_diff",
    "seed_diff",
]


def test_encode_matchups_uses_prior_season_elo_and_seed_diff_sign() -> None:
    season = 2023
    team1 = 1101
    team2 = 1102

    matchups_df = pd.DataFrame({"Season": [season], "Team1ID": [team1], "Team2ID": [team2]})

    # Prior-season Elo only.
    elo_dict = {
        (season - 1, team1): 1600.0,
        (season - 1, team2): 1500.0,
    }

    efficiency_df = pd.DataFrame(
        [
            {"Season": season, "TeamID": team1, "AdjEM": 10.0},
            {"Season": season, "TeamID": team2, "AdjEM": 2.0},
        ]
    )

    four_factors_df = pd.DataFrame(
        [
            {
                "Season": season,
                "TeamID": team1,
                "eFG_pct": 0.60,
                "TO_pct": 0.18,
                "OR_pct": 0.30,
                "FTRate": 0.20,
                "opp_eFG_pct": 0.55,
                "opp_TO_pct": 0.19,
                "opp_OR_pct": 0.28,
            },
            {
                "Season": season,
                "TeamID": team2,
                "eFG_pct": 0.50,
                "TO_pct": 0.22,
                "OR_pct": 0.25,
                "FTRate": 0.18,
                "opp_eFG_pct": 0.52,
                "opp_TO_pct": 0.20,
                "opp_OR_pct": 0.26,
            },
        ]
    )

    massey_df = pd.DataFrame(
        [
            {"Season": season, "TeamID": team1, "massey_consensus": 1.5},
            {"Season": season, "TeamID": team2, "massey_consensus": -0.5},
        ]
    )

    # Lower seed is better. seed_diff = seed[Team2] - seed[Team1]
    seeds_df = pd.DataFrame(
        [
            {"Season": season, "TeamID": team1, "Seed": 5},
            {"Season": season, "TeamID": team2, "Seed": 10},
        ]
    )

    from src.features import encode_matchups

    out = encode_matchups(
        matchups_df=matchups_df,
        elo_dict=elo_dict,
        efficiency_df=efficiency_df,
        four_factors_df=four_factors_df,
        massey_df=massey_df,
        seeds_df=seeds_df,
        label_col=None,
    )

    assert len(out) == 1
    assert list(out.columns) == [
        "elo_diff",
        "adjEM_diff",
        "eFG_diff",
        "TO_diff",
        "OR_diff",
        "FTRate_diff",
        "opp_eFG_diff",
        "opp_TO_diff",
        "opp_OR_diff",
        "massey_diff",
        "seed_diff",
        "label",
    ]

    # Elo diff uses prior-season.
    assert out.loc[0, "elo_diff"] == 100.0
    assert out.loc[0, "adjEM_diff"] == 8.0
    assert out.loc[0, "seed_diff"] == 5.0  # 10 - 5

    # Feature columns should be NaN-free.
    assert not out[FEATURE_COLS].isna().any().any()
    assert out["label"].isna().all()


def test_encode_matchups_symmetric_flips_signs_and_label_and_doubles_rows() -> None:
    season = 2024
    team1 = 1101
    team2 = 1102

    matchups_df = pd.DataFrame({"Season": [season], "Team1ID": [team1], "Team2ID": [team2]})
    label_col = pd.Series([1])  # Team1 won

    # Elo prior season.
    elo_dict = {
        (season - 1, team1): 1500.0,
        (season - 1, team2): 1400.0,
    }

    # Provide efficiency + massey only for team1; team2 missing to exercise 0-fill.
    efficiency_df = pd.DataFrame(
        [
            {"Season": season, "TeamID": team1, "AdjEM": 5.0},
        ]
    )
    massey_df = pd.DataFrame(
        [
            {"Season": season, "TeamID": team1, "massey_consensus": 2.0},
        ]
    )

    # Provide four_factors only for team1; team2 missing to exercise 0-fill.
    four_factors_df = pd.DataFrame(
        [
            {
                "Season": season,
                "TeamID": team1,
                "eFG_pct": 0.60,
                "TO_pct": 0.18,
                "OR_pct": 0.30,
                "FTRate": 0.20,
                "opp_eFG_pct": 0.55,
                "opp_TO_pct": 0.19,
                "opp_OR_pct": 0.28,
            },
        ]
    )

    # Seed missing for team2 => seed_diff must be 0.
    seeds_df = pd.DataFrame(
        [
            {"Season": season, "TeamID": team1, "Seed": 7},
        ]
    )

    from src.features import encode_matchups, encode_matchups_symmetric

    original = encode_matchups(
        matchups_df=matchups_df,
        elo_dict=elo_dict,
        efficiency_df=efficiency_df,
        four_factors_df=four_factors_df,
        massey_df=massey_df,
        seeds_df=seeds_df,
        label_col=label_col,
    )
    sym = encode_matchups_symmetric(
        matchups_df=matchups_df,
        elo_dict=elo_dict,
        efficiency_df=efficiency_df,
        four_factors_df=four_factors_df,
        massey_df=massey_df,
        seeds_df=seeds_df,
        label_col=label_col,
    )

    assert len(sym) == 2
    assert not sym[FEATURE_COLS].isna().any().any()

    # One row should equal original diffs, the other should have diffs negated.
    orig_vals = original.loc[0, FEATURE_COLS].to_numpy(dtype=float)

    sym_sorted = sym.sort_values("elo_diff").reset_index(drop=True)
    row_neg = sym_sorted.iloc[0]
    row_pos = sym_sorted.iloc[1]

    if original.loc[0, "elo_diff"] > 0:
        assert row_pos["elo_diff"] == original.loc[0, "elo_diff"]
        assert row_neg["elo_diff"] == -original.loc[0, "elo_diff"]
    else:
        assert row_neg["elo_diff"] == original.loc[0, "elo_diff"]
        assert row_pos["elo_diff"] == -original.loc[0, "elo_diff"]

    # All diff features must flip sign between the two rows.
    for col in FEATURE_COLS:
        assert math_is_close(row_pos[col], orig_vals[FEATURE_COLS.index(col)])
        assert math_is_close(row_neg[col], -orig_vals[FEATURE_COLS.index(col)])

    # Label must flip: 1 -> 0 for swapped ordering.
    assert set(sym["label"].tolist()) == {0, 1}


def math_is_close(a: float, b: float, *, rel_tol: float = 1e-12, abs_tol: float = 1e-12) -> bool:
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

