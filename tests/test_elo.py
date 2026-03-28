import math
from pathlib import Path

import pandas as pd


COLUMN_LIST = [
    "Season",
    "DayNum",
    "WTeamID",
    "WScore",
    "LTeamID",
    "LScore",
    "WLoc",
    "NumOT",
]


def _write_csvs(
    tmp_path: Path,
    *,
    gender: str,
    regular_rows: list[dict],
    tourney_rows: list[dict],
) -> None:
    if gender == "men":
        regular_name = "MRegularSeasonCompactResults.csv"
        tourney_name = "MNCAATourneyCompactResults.csv"
    elif gender == "women":
        regular_name = "WRegularSeasonCompactResults.csv"
        tourney_name = "WNCAATourneyCompactResults.csv"
    else:
        raise ValueError("gender")

    pd.DataFrame(regular_rows, columns=COLUMN_LIST).to_csv(tmp_path / regular_name, index=False)
    pd.DataFrame(tourney_rows, columns=COLUMN_LIST).to_csv(tmp_path / tourney_name, index=False)


def test_compute_elo_hca_home_adjustment_and_snapshot(tmp_path: Path) -> None:
    # One game in season 2026:
    # winner at home (WLoc='H') => winner's rating adjusted +ELO_HCA only for expected-score.
    regular_rows = [
        {
            "Season": 2026,
            "DayNum": 0,
            "WTeamID": 1101,
            "WScore": 80,
            "LTeamID": 1102,
            "LScore": 70,
            "WLoc": "H",
            "NumOT": 0,
        }
    ]
    _write_csvs(tmp_path, gender="men", regular_rows=regular_rows, tourney_rows=[])
    _write_csvs(tmp_path, gender="women", regular_rows=[], tourney_rows=[])

    from src.elo import compute_elo_men

    elo = compute_elo_men(str(tmp_path))
    assert (2026, 1101) in elo
    assert (2026, 1102) in elo

    ELO_INIT = 1500.0
    ELO_HCA = 100.0
    K = 20.0

    w_adj = ELO_INIT + ELO_HCA
    l = ELO_INIT
    exp_w = 1.0 / (1.0 + 10 ** ((l - w_adj) / 400.0))

    expected_w = ELO_INIT + K * (1.0 - exp_w)
    expected_l = ELO_INIT + K * (0.0 - (1.0 - exp_w))

    assert math.isclose(elo[(2026, 1101)], expected_w, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(elo[(2026, 1102)], expected_l, rel_tol=1e-12, abs_tol=1e-12)


def test_compute_elo_hca_away_adjustment_and_snapshot(tmp_path: Path) -> None:
    # One game at away court (WLoc='A') => winner's rating adjusted -ELO_HCA for expected-score.
    regular_rows = [
        {
            "Season": 2026,
            "DayNum": 0,
            "WTeamID": 1101,
            "WScore": 80,
            "LTeamID": 1102,
            "LScore": 70,
            "WLoc": "A",
            "NumOT": 0,
        }
    ]
    _write_csvs(tmp_path, gender="men", regular_rows=regular_rows, tourney_rows=[])
    _write_csvs(tmp_path, gender="women", regular_rows=[], tourney_rows=[])

    from src.elo import compute_elo_men

    elo = compute_elo_men(str(tmp_path))

    ELO_INIT = 1500.0
    ELO_HCA = 100.0
    K = 20.0

    w_adj = ELO_INIT - ELO_HCA
    l = ELO_INIT
    exp_w = 1.0 / (1.0 + 10 ** ((l - w_adj) / 400.0))

    expected_w = ELO_INIT + K * (1.0 - exp_w)
    expected_l = ELO_INIT + K * (0.0 - (1.0 - exp_w))

    assert math.isclose(elo[(2026, 1101)], expected_w, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(elo[(2026, 1102)], expected_l, rel_tol=1e-12, abs_tol=1e-12)


def test_compute_elo_between_seasons_regression(tmp_path: Path) -> None:
    # Season regression: for season 2026 start, ratings should be
    #   R_new = 0.75 * R_end_prev + 0.25 * 1500
    regular_rows = [
        {
            "Season": 2025,
            "DayNum": 0,
            "WTeamID": 1101,
            "WScore": 75,
            "LTeamID": 1102,
            "LScore": 70,
            "WLoc": "N",
            "NumOT": 0,
        },
        {
            "Season": 2026,
            "DayNum": 0,
            "WTeamID": 1101,
            "WScore": 75,
            "LTeamID": 1102,
            "LScore": 70,
            "WLoc": "N",
            "NumOT": 0,
        },
    ]
    _write_csvs(tmp_path, gender="men", regular_rows=regular_rows, tourney_rows=[])
    _write_csvs(tmp_path, gender="women", regular_rows=[], tourney_rows=[])

    from src.elo import compute_elo_men

    elo = compute_elo_men(str(tmp_path))

    assert (2025, 1101) in elo
    assert (2025, 1102) in elo
    assert (2026, 1101) in elo
    assert (2026, 1102) in elo

    ELO_INIT = 1500.0
    K = 20.0

    # 2025 expected win prob starting from 1500/1500
    exp_2025 = 1.0 / (1.0 + 10 ** ((ELO_INIT - ELO_INIT) / 400.0))
    w_end_2025 = ELO_INIT + K * (1.0 - exp_2025)
    l_end_2025 = ELO_INIT + K * (0.0 - (1.0 - exp_2025))

    # Regression before processing 2026
    w_start_2026 = 0.75 * w_end_2025 + 0.25 * ELO_INIT
    l_start_2026 = 0.75 * l_end_2025 + 0.25 * ELO_INIT

    exp_2026 = 1.0 / (1.0 + 10 ** ((l_start_2026 - w_start_2026) / 400.0))
    expected_w_end_2026 = w_start_2026 + K * (1.0 - exp_2026)
    expected_l_end_2026 = l_start_2026 + K * (0.0 - (1.0 - exp_2026))

    assert math.isclose(elo[(2026, 1101)], expected_w_end_2026, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(elo[(2026, 1102)], expected_l_end_2026, rel_tol=1e-12, abs_tol=1e-12)


def test_compute_elo_sorts_by_season_then_daynum(tmp_path: Path) -> None:
    # Must sort by (Season, DayNum), not just DayNum.
    # Provide rows in reverse season order with conflicting DayNum values.
    # Correct order => Season 2025 first, then Season 2026.
    regular_rows_reverse_season = [
        {
            "Season": 2026,
            "DayNum": 0,
            "WTeamID": 1101,
            "WScore": 75,
            "LTeamID": 1102,
            "LScore": 70,
            "WLoc": "N",
            "NumOT": 0,
        },
        {
            "Season": 2025,
            "DayNum": 10,
            "WTeamID": 1101,
            "WScore": 75,
            "LTeamID": 1102,
            "LScore": 70,
            "WLoc": "N",
            "NumOT": 0,
        },
    ]
    _write_csvs(tmp_path, gender="men", regular_rows=regular_rows_reverse_season, tourney_rows=[])
    _write_csvs(tmp_path, gender="women", regular_rows=[], tourney_rows=[])

    from src.elo import compute_elo_men

    elo = compute_elo_men(str(tmp_path))

    ELO_INIT = 1500.0
    K = 20.0

    # Correct processing order:
    #  - Season 2025: day 10, 1101 wins from 1500/1500 => 1510/1490
    exp_2025 = 1.0 / (1.0 + 10 ** ((ELO_INIT - ELO_INIT) / 400.0))
    r_1101_end_2025 = ELO_INIT + K * (1.0 - exp_2025)
    r_1102_end_2025 = ELO_INIT + K * (0.0 - (1.0 - exp_2025))

    #  - Regression before 2026
    r_1101_start_2026 = 0.75 * r_1101_end_2025 + 0.25 * ELO_INIT
    r_1102_start_2026 = 0.75 * r_1102_end_2025 + 0.25 * ELO_INIT

    #  - Season 2026: day 0, 1101 wins at neutral
    exp_2026 = 1.0 / (1.0 + 10 ** ((r_1102_start_2026 - r_1101_start_2026) / 400.0))
    expected_r_1101_end_2026 = r_1101_start_2026 + K * (1.0 - exp_2026)
    expected_r_1102_end_2026 = r_1102_start_2026 + K * (0.0 - (1.0 - exp_2026))

    assert math.isclose(elo[(2026, 1101)], expected_r_1101_end_2026, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(elo[(2026, 1102)], expected_r_1102_end_2026, rel_tol=1e-12, abs_tol=1e-12)


def test_compute_elo_tourney_games_included_in_season_snapshot(tmp_path: Path) -> None:
    # Snapshot must be end-of-season after applying both regular and tourney games.
    # Season 2026 only, with a regular game *after* the tourney by DayNum.
    # This catches incorrect implementations that process all regular games
    # first and all tourney games afterwards (ignoring DayNum ordering).
    #
    # Expected DayNum order:
    #  - regular (DayNum 0): 1101 beats 1102
    #  - tourney (DayNum 140): 1102 beats 1101
    #  - regular (DayNum 200): 1101 beats 1102
    regular_rows = [
        {
            "Season": 2026,
            "DayNum": 0,
            "WTeamID": 1101,
            "WScore": 80,
            "LTeamID": 1102,
            "LScore": 70,
            "WLoc": "N",
            "NumOT": 0,
        },
        {
            "Season": 2026,
            "DayNum": 200,
            "WTeamID": 1101,
            "WScore": 85,
            "LTeamID": 1102,
            "LScore": 75,
            "WLoc": "N",
            "NumOT": 0,
        },
    ]
    tourney_rows = [
        {
            "Season": 2026,
            "DayNum": 140,
            "WTeamID": 1102,
            "WScore": 75,
            "LTeamID": 1101,
            "LScore": 70,
            "WLoc": "N",
            "NumOT": 0,
        }
    ]
    _write_csvs(tmp_path, gender="men", regular_rows=regular_rows, tourney_rows=tourney_rows)
    _write_csvs(tmp_path, gender="women", regular_rows=[], tourney_rows=[])

    from src.elo import compute_elo_men

    elo = compute_elo_men(str(tmp_path))

    ELO_INIT = 1500.0
    K = 20.0

    # Game 1 (DayNum 0): 1101 wins at neutral
    exp_g1 = 1.0 / (1.0 + 10 ** ((ELO_INIT - ELO_INIT) / 400.0))
    r_1101_after_g1 = ELO_INIT + K * (1.0 - exp_g1)
    r_1102_after_g1 = ELO_INIT + K * (0.0 - (1.0 - exp_g1))

    # Game 2 (DayNum 140): 1102 wins at neutral (using ratings after game 1)
    exp_g2 = 1.0 / (1.0 + 10 ** ((r_1101_after_g1 - r_1102_after_g1) / 400.0))
    r_1102_after_g2 = r_1102_after_g1 + K * (1.0 - exp_g2)
    r_1101_after_g2 = r_1101_after_g1 + K * (0.0 - (1.0 - exp_g2))

    # Game 3 (DayNum 200): 1101 wins at neutral (using ratings after game 2)
    exp_g3 = 1.0 / (1.0 + 10 ** ((r_1102_after_g2 - r_1101_after_g2) / 400.0))
    r_1101_after_g3 = r_1101_after_g2 + K * (1.0 - exp_g3)
    r_1102_after_g3 = r_1102_after_g2 + K * (0.0 - (1.0 - exp_g3))

    assert math.isclose(elo[(2026, 1101)], r_1101_after_g3, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(elo[(2026, 1102)], r_1102_after_g3, rel_tol=1e-12, abs_tol=1e-12)


def test_compute_elo_processed_by_season_daynum_order(tmp_path: Path) -> None:
    # Must sort by (Season, DayNum). Provide input in reverse DayNum order and assert
    # the result matches the sorted outcome.
    regular_rows_reverse = [
        {
            "Season": 2026,
            "DayNum": 1,
            "WTeamID": 1102,
            "WScore": 70,
            "LTeamID": 1101,
            "LScore": 65,
            "WLoc": "N",
            "NumOT": 0,
        },
        {
            "Season": 2026,
            "DayNum": 0,
            "WTeamID": 1101,
            "WScore": 80,
            "LTeamID": 1102,
            "LScore": 70,
            "WLoc": "N",
            "NumOT": 0,
        },
    ]
    _write_csvs(tmp_path, gender="men", regular_rows=regular_rows_reverse, tourney_rows=[])
    _write_csvs(tmp_path, gender="women", regular_rows=[], tourney_rows=[])

    from src.elo import compute_elo_men

    elo = compute_elo_men(str(tmp_path))

    ELO_INIT = 1500.0
    K = 20.0

    # Correct order:
    #  - DayNum 0: 1101 beats 1102
    #  - DayNum 1: 1102 beats 1101
    exp_g0 = 1.0 / (1.0 + 10 ** ((ELO_INIT - ELO_INIT) / 400.0))
    r1_after_g0 = ELO_INIT + K * (1.0 - exp_g0)  # 1101
    r2_after_g0 = ELO_INIT + K * (0.0 - (1.0 - exp_g0))  # 1102

    exp_g1 = 1.0 / (1.0 + 10 ** ((r1_after_g0 - r2_after_g0) / 400.0))
    r2_final = r2_after_g0 + K * (1.0 - exp_g1)  # 1102 wins
    r1_final = r1_after_g0 + K * (0.0 - (1.0 - exp_g1))  # 1101 loses

    assert math.isclose(elo[(2026, 1101)], r1_final, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(elo[(2026, 1102)], r2_final, rel_tol=1e-12, abs_tol=1e-12)


def test_compute_elo_men_and_women_are_independent(tmp_path: Path) -> None:
    # Men's TeamIDs don't overlap women's TeamIDs.
    _write_csvs(
        tmp_path,
        gender="men",
        regular_rows=[
            {
                "Season": 2026,
                "DayNum": 0,
                "WTeamID": 1101,
                "WScore": 80,
                "LTeamID": 1102,
                "LScore": 70,
                "WLoc": "N",
                "NumOT": 0,
            }
        ],
        tourney_rows=[],
    )
    _write_csvs(
        tmp_path,
        gender="women",
        regular_rows=[
            {
                "Season": 2026,
                "DayNum": 0,
                "WTeamID": 3101,
                "WScore": 80,
                "LTeamID": 3102,
                "LScore": 70,
                "WLoc": "N",
                "NumOT": 0,
            }
        ],
        tourney_rows=[],
    )

    from src.elo import compute_elo_men, compute_elo_women

    men = compute_elo_men(str(tmp_path))
    women = compute_elo_women(str(tmp_path))

    assert (2026, 1101) in men
    assert (2026, 1102) in men
    assert (2026, 3101) not in men
    assert (2026, 3102) not in men

    assert (2026, 3101) in women
    assert (2026, 3102) in women
    assert (2026, 1101) not in women
    assert (2026, 1102) not in women

