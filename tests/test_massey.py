import math
from pathlib import Path

import pandas as pd


def _write_massey_csv(tmp_path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(tmp_path / "MMasseyOrdinals.csv", index=False)


def _write_tourney_compact(tmp_path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(tmp_path / "MNCAATourneyCompactResults.csv", index=False)


def test_list_stable_systems_coverage_filter(tmp_path: Path) -> None:
    # Seasons: 2024, 2025. min_coverage=1.0 => must appear in both seasons.
    ord_rows = [
        # Season 2024 snapshot
        {"Season": 2024, "RankingDayNum": 133, "SystemName": "SYS_A", "TeamID": 1101, "OrdinalRank": 1},
        {"Season": 2024, "RankingDayNum": 133, "SystemName": "SYS_A", "TeamID": 1102, "OrdinalRank": 2},
        {"Season": 2024, "RankingDayNum": 133, "SystemName": "SYS_B", "TeamID": 1101, "OrdinalRank": 2},
        {"Season": 2024, "RankingDayNum": 133, "SystemName": "SYS_B", "TeamID": 1102, "OrdinalRank": 1},
        {"Season": 2024, "RankingDayNum": 133, "SystemName": "SYS_C", "TeamID": 1101, "OrdinalRank": 1},
        {"Season": 2024, "RankingDayNum": 133, "SystemName": "SYS_C", "TeamID": 1102, "OrdinalRank": 2},
        # Season 2025 snapshot
        {"Season": 2025, "RankingDayNum": 133, "SystemName": "SYS_A", "TeamID": 1101, "OrdinalRank": 1},
        {"Season": 2025, "RankingDayNum": 133, "SystemName": "SYS_A", "TeamID": 1102, "OrdinalRank": 2},
        {"Season": 2025, "RankingDayNum": 133, "SystemName": "SYS_B", "TeamID": 1101, "OrdinalRank": 2},
        {"Season": 2025, "RankingDayNum": 133, "SystemName": "SYS_B", "TeamID": 1102, "OrdinalRank": 1},
        # SYS_C missing in 2025 => should be filtered out at min_coverage=1.0
    ]
    _write_massey_csv(tmp_path, ord_rows)
    _write_tourney_compact(
        tmp_path,
        [
            {
                "Season": 2024,
                "DayNum": 140,
                "WTeamID": 1101,
                "WScore": 70,
                "LTeamID": 1102,
                "LScore": 60,
                "WLoc": "N",
                "NumOT": 0,
            }
        ],
    )

    from src.massey import list_stable_systems

    systems = list_stable_systems(str(tmp_path), min_coverage=1.0)
    assert systems == ["SYS_A", "SYS_B"]


def test_load_massey_features_consensus_and_top_systems(tmp_path: Path) -> None:
    # Build 2 seasons, 2 systems, 4 teams.
    # SYS_A: ranks lower TeamID better (predicts TeamID lower tends to win)
    # SYS_B: ranks higher TeamID better (anti-predictive for our synthetic tourney outcomes)
    teams = [1101, 1102, 1103, 1104]

    def season_rows(season: int) -> list[dict]:
        rows: list[dict] = []
        # SYS_A ranks in TeamID order
        for rank, tid in enumerate(teams, start=1):
            rows.append(
                {
                    "Season": season,
                    "RankingDayNum": 133,
                    "SystemName": "SYS_A",
                    "TeamID": tid,
                    "OrdinalRank": rank,
                }
            )
        # SYS_B reverse
        for rank, tid in enumerate(reversed(teams), start=1):
            rows.append(
                {
                    "Season": season,
                    "RankingDayNum": 133,
                    "SystemName": "SYS_B",
                    "TeamID": tid,
                    "OrdinalRank": rank,
                }
            )
        return rows

    ord_rows = season_rows(2024) + season_rows(2025)
    _write_massey_csv(tmp_path, ord_rows)

    # Tournament games: lower TeamID always wins (by construction).
    tourney_rows = [
        {"Season": 2024, "DayNum": 140, "WTeamID": 1101, "WScore": 70, "LTeamID": 1104, "LScore": 60, "WLoc": "N", "NumOT": 0},
        {"Season": 2024, "DayNum": 141, "WTeamID": 1102, "WScore": 70, "LTeamID": 1103, "LScore": 60, "WLoc": "N", "NumOT": 0},
        {"Season": 2025, "DayNum": 140, "WTeamID": 1101, "WScore": 70, "LTeamID": 1104, "LScore": 60, "WLoc": "N", "NumOT": 0},
        {"Season": 2025, "DayNum": 141, "WTeamID": 1102, "WScore": 70, "LTeamID": 1103, "LScore": 60, "WLoc": "N", "NumOT": 0},
    ]
    _write_tourney_compact(tmp_path, tourney_rows)

    from src.massey import load_massey_features

    out = load_massey_features(str(tmp_path), min_coverage=1.0)
    assert {"Season", "TeamID", "massey_consensus", "massey_n_systems"} <= set(out.columns)
    assert (out["massey_n_systems"] == 2).all()

    # Top system should include SYS_A and not include filtered systems.
    top_cols = [c for c in out.columns if c.startswith("massey_top3_")]
    assert "massey_top3_SYS_A" in top_cols
    assert "massey_top3_SYS_B" in top_cols

    # Sign check: in SYS_A, best team (rank 1) should have highest z after sign flip.
    row_1101_2024 = out[(out["Season"] == 2024) & (out["TeamID"] == 1101)].iloc[0]
    row_1104_2024 = out[(out["Season"] == 2024) & (out["TeamID"] == 1104)].iloc[0]
    assert row_1101_2024["massey_top3_SYS_A"] > row_1104_2024["massey_top3_SYS_A"]

    # Consensus should be bounded and centered-ish (basic sanity, not strict distribution test).
    assert out["massey_consensus"].between(-10, 10).all()

