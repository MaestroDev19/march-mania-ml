import math

import pandas as pd


def test_compute_four_factors_single_game_two_teams() -> None:
    # One neutral-site game, with enough stats to compute all factors.
    detailed = pd.DataFrame(
        [
            {
                "Season": 2026,
                "DayNum": 10,
                "WTeamID": 1101,
                "WScore": 80,
                "LTeamID": 1102,
                "LScore": 70,
                "WLoc": "N",
                # Winner offensive box score
                "WFGM": 28,
                "WFGA": 60,
                "WFGM3": 8,
                "WFGA3": 20,
                "WFTA": 20,
                "WOR": 10,
                "WDR": 22,
                "WTO": 12,
                # Loser offensive box score
                "LFGM": 25,
                "LFGA": 58,
                "LFGM3": 6,
                "LFGA3": 18,
                "LFTA": 18,
                "LOR": 8,
                "LDR": 20,
                "LTO": 14,
            }
        ]
    )

    from src.features import compute_four_factors

    out = compute_four_factors(detailed)
    expected_cols = {
        "Season",
        "TeamID",
        "eFG_pct",
        "TO_pct",
        "OR_pct",
        "FTRate",
        "opp_eFG_pct",
        "opp_TO_pct",
        "opp_OR_pct",
        "opp_FTRate",
    }
    assert set(out.columns) == expected_cols
    assert len(out) == 2

    # Possessions:
    poss_1101 = 60 - 10 + 12 + 0.475 * 20
    poss_1102 = 58 - 8 + 14 + 0.475 * 18

    # Team 1101
    row_1101 = out[(out["Season"] == 2026) & (out["TeamID"] == 1101)].iloc[0]
    efg_1101 = (28 + 0.5 * 8) / 60
    to_1101 = 12 / poss_1101
    or_1101 = 10 / (10 + 20)  # OppDefReb = LDR
    ft_1101 = 20 / 60

    assert math.isclose(row_1101["eFG_pct"], efg_1101, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_1101["TO_pct"], to_1101, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_1101["OR_pct"], or_1101, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_1101["FTRate"], ft_1101, rel_tol=1e-12, abs_tol=1e-12)

    # Opponent-side against 1101 is 1102's offense and rebounding vs 1101's defense.
    opp_efg_against_1101 = (25 + 0.5 * 6) / 58
    opp_to_against_1101 = 14 / poss_1102
    opp_or_against_1101 = 8 / (8 + 22)  # OppOffReb / (OppOffReb + YourDefReb)
    opp_ft_against_1101 = 18 / 58

    assert math.isclose(row_1101["opp_eFG_pct"], opp_efg_against_1101, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_1101["opp_TO_pct"], opp_to_against_1101, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_1101["opp_OR_pct"], opp_or_against_1101, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_1101["opp_FTRate"], opp_ft_against_1101, rel_tol=1e-12, abs_tol=1e-12)

    # Team 1102
    row_1102 = out[(out["Season"] == 2026) & (out["TeamID"] == 1102)].iloc[0]
    efg_1102 = (25 + 0.5 * 6) / 58
    to_1102 = 14 / poss_1102
    or_1102 = 8 / (8 + 22)  # OppDefReb = WDR
    ft_1102 = 18 / 58

    assert math.isclose(row_1102["eFG_pct"], efg_1102, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_1102["TO_pct"], to_1102, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_1102["OR_pct"], or_1102, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_1102["FTRate"], ft_1102, rel_tol=1e-12, abs_tol=1e-12)

    # Opponent-side against 1102 is 1101's offense and rebounding vs 1102's defense.
    opp_efg_against_1102 = (28 + 0.5 * 8) / 60
    opp_to_against_1102 = 12 / poss_1101
    opp_or_against_1102 = 10 / (10 + 20)  # OppOffReb / (OppOffReb + YourDefReb)
    opp_ft_against_1102 = 20 / 60

    assert math.isclose(row_1102["opp_eFG_pct"], opp_efg_against_1102, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_1102["opp_TO_pct"], opp_to_against_1102, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_1102["opp_OR_pct"], opp_or_against_1102, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_1102["opp_FTRate"], opp_ft_against_1102, rel_tol=1e-12, abs_tol=1e-12)


def test_compute_four_factors_clips_efg_pct_bounds() -> None:
    # Make winner eFG > 0.8 and ensure it is clipped to 0.8 after aggregation.
    detailed = pd.DataFrame(
        [
            {
                "Season": 2026,
                "DayNum": 10,
                "WTeamID": 1101,
                "WScore": 80,
                "LTeamID": 1102,
                "LScore": 70,
                "WLoc": "N",
                # Winner: eFG = (50 + 0.5*0)/50 = 1.0 -> clip to 0.8
                "WFGM": 50,
                "WFGA": 50,
                "WFGM3": 0,
                "WFGA3": 0,
                "WFTA": 0,
                "WOR": 0,
                "WDR": 10,
                "WTO": 0,
                # Loser: keep reasonable
                "LFGM": 25,
                "LFGA": 50,
                "LFGM3": 0,
                "LFGA3": 0,
                "LFTA": 0,
                "LOR": 0,
                "LDR": 10,
                "LTO": 0,
            }
        ]
    )

    from src.features import compute_four_factors

    out = compute_four_factors(detailed)
    row_1101 = out[(out["Season"] == 2026) & (out["TeamID"] == 1101)].iloc[0]
    assert row_1101["eFG_pct"] == 0.8

