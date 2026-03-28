import math

import pandas as pd


def test_compute_efficiency_single_game_two_teams(tmp_path) -> None:
    # One game, neutral site.
    # Team 1101 (winner) beats 1102 (loser).
    # We expect one row per (Season, TeamID) with Poss, OffEff, DefEff, AdjEM.
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
                # Winner stats
                "WFGA": 60,
                "WOR": 10,
                "WTO": 12,
                "WFTA": 20,
                # Loser stats
                "LFGA": 58,
                "LOR": 8,
                "LTO": 14,
                "LFTA": 18,
            }
        ]
    )

    from src.features import compute_efficiency

    out = compute_efficiency(detailed)
    assert set(out.columns) == {"Season", "TeamID", "Poss", "OffEff", "DefEff", "AdjEM"}
    assert len(out) == 2

    # Possessions per team-game:
    # Poss = FGA - OffReb + TO + 0.475 * FTA
    poss_w = 60 - 10 + 12 + 0.475 * 20
    poss_l = 58 - 8 + 14 + 0.475 * 18

    row_w = out[(out["Season"] == 2026) & (out["TeamID"] == 1101)].iloc[0]
    row_l = out[(out["Season"] == 2026) & (out["TeamID"] == 1102)].iloc[0]

    assert math.isclose(row_w["Poss"], poss_w, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_l["Poss"], poss_l, rel_tol=1e-12, abs_tol=1e-12)

    offeff_w = (80 / poss_w) * 100.0
    defeff_w = (70 / poss_w) * 100.0
    offeff_l = (70 / poss_l) * 100.0
    defeff_l = (80 / poss_l) * 100.0

    assert math.isclose(row_w["OffEff"], offeff_w, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_w["DefEff"], defeff_w, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_w["AdjEM"], offeff_w - defeff_w, rel_tol=1e-12, abs_tol=1e-12)

    assert math.isclose(row_l["OffEff"], offeff_l, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_l["DefEff"], defeff_l, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_l["AdjEM"], offeff_l - defeff_l, rel_tol=1e-12, abs_tol=1e-12)


def test_compute_efficiency_drops_possessions_below_40() -> None:
    # Construct a team-game with Poss < 40 and ensure it is dropped from averages.
    # Game A: valid possessions
    # Game B: invalid possessions (Poss < 40) for both teams
    detailed = pd.DataFrame(
        [
            {
                "Season": 2026,
                "DayNum": 1,
                "WTeamID": 1101,
                "WScore": 80,
                "LTeamID": 1102,
                "LScore": 70,
                "WLoc": "N",
                "WFGA": 60,
                "WOR": 10,
                "WTO": 12,
                "WFTA": 20,
                "LFGA": 58,
                "LOR": 8,
                "LTO": 14,
                "LFTA": 18,
            },
            {
                "Season": 2026,
                "DayNum": 2,
                "WTeamID": 1101,
                "WScore": 10,
                "LTeamID": 1102,
                "LScore": 9,
                "WLoc": "N",
                # Force Poss < 40:
                "WFGA": 30,
                "WOR": 0,
                "WTO": 0,
                "WFTA": 0,
                "LFGA": 30,
                "LOR": 0,
                "LTO": 0,
                "LFTA": 0,
            },
        ]
    )

    from src.features import compute_efficiency

    out = compute_efficiency(detailed)

    poss_w_valid = 60 - 10 + 12 + 0.475 * 20
    poss_l_valid = 58 - 8 + 14 + 0.475 * 18

    row_w = out[(out["Season"] == 2026) & (out["TeamID"] == 1101)].iloc[0]
    row_l = out[(out["Season"] == 2026) & (out["TeamID"] == 1102)].iloc[0]

    # If the invalid game were included, Poss would be averaged down; we expect it unchanged.
    assert math.isclose(row_w["Poss"], poss_w_valid, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(row_l["Poss"], poss_l_valid, rel_tol=1e-12, abs_tol=1e-12)

