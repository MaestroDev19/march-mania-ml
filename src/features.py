from __future__ import annotations

from typing import Final, Optional

import pandas as pd
import numpy as np

import logging


_logger = logging.getLogger(__name__)


_POSSESSIONS_FTA_COEF: Final[float] = 0.475
_MIN_POSSESSIONS: Final[float] = 40.0

_EFG_CLIP_MIN: Final[float] = 0.2
_EFG_CLIP_MAX: Final[float] = 0.8


def compute_efficiency(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team, per-season efficiency metrics from Detailed Results.

    Args:
        detailed_df: DataFrame in the Kaggle "RegularSeasonDetailedResults" shape,
            containing one row per game with winner (W*) and loser (L*) stats.

    Returns:
        DataFrame with one row per (Season, TeamID) and columns:
        Season, TeamID, Poss, OffEff, DefEff, AdjEM
    """
    required_cols = [
        "Season",
        "WTeamID",
        "LTeamID",
        "WScore",
        "LScore",
        "WLoc",
        "WFGA",
        "WOR",
        "WTO",
        "WFTA",
        "LFGA",
        "LOR",
        "LTO",
        "LFTA",
    ]
    missing = [c for c in required_cols if c not in detailed_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = detailed_df[required_cols].copy()

    is_neutral = df["WLoc"] == "N"

    winners = pd.DataFrame(
        {
            "Season": df["Season"].astype(int),
            "TeamID": df["WTeamID"].astype(int),
            "PointsScored": df["WScore"].astype(float),
            "PointsAllowed": df["LScore"].astype(float),
            "FGA": df["WFGA"].astype(float),
            "OffReb": df["WOR"].astype(float),
            "TO": df["WTO"].astype(float),
            "FTA": df["WFTA"].astype(float),
            "is_neutral": is_neutral.astype(bool),
        }
    )

    losers = pd.DataFrame(
        {
            "Season": df["Season"].astype(int),
            "TeamID": df["LTeamID"].astype(int),
            "PointsScored": df["LScore"].astype(float),
            "PointsAllowed": df["WScore"].astype(float),
            "FGA": df["LFGA"].astype(float),
            "OffReb": df["LOR"].astype(float),
            "TO": df["LTO"].astype(float),
            "FTA": df["LFTA"].astype(float),
            "is_neutral": is_neutral.astype(bool),
        }
    )

    team_games = pd.concat([winners, losers], ignore_index=True)
    team_games["Poss"] = (
        team_games["FGA"]
        - team_games["OffReb"]
        + team_games["TO"]
        + _POSSESSIONS_FTA_COEF * team_games["FTA"]
    )

    team_games = team_games[team_games["Poss"] >= _MIN_POSSESSIONS].copy()

    grouped = team_games.groupby(["Season", "TeamID"], as_index=False)
    agg = grouped.agg(
        Poss=("Poss", "mean"),
        PointsScored=("PointsScored", "sum"),
        PointsAllowed=("PointsAllowed", "sum"),
        PossSum=("Poss", "sum"),
    )

    agg["OffEff"] = (agg["PointsScored"] / agg["PossSum"]) * 100.0
    agg["DefEff"] = (agg["PointsAllowed"] / agg["PossSum"]) * 100.0
    agg["AdjEM"] = agg["OffEff"] - agg["DefEff"]

    out = agg[["Season", "TeamID", "Poss", "OffEff", "DefEff", "AdjEM"]].copy()
    out = out.sort_values(["Season", "TeamID"], kind="mergesort").reset_index(drop=True)
    return out


def compute_four_factors(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team, per-season Four Factors from Detailed Results.

    Args:
        detailed_df: DataFrame containing one row per game with winner (W*) and
            loser (L*) stats.

    Returns:
        DataFrame with one row per (Season, TeamID) and columns:
        Season, TeamID, eFG_pct, TO_pct, OR_pct, FTRate,
        opp_eFG_pct, opp_TO_pct, opp_OR_pct, opp_FTRate
    """
    required_cols = [
        "Season",
        "WTeamID",
        "LTeamID",
        "WLoc",
        "WFGA",
        "WFGM",
        "WFGM3",
        "WFTA",
        "WOR",
        "WDR",
        "WTO",
        "LFGA",
        "LFGM",
        "LFGM3",
        "LFTA",
        "LOR",
        "LDR",
        "LTO",
    ]
    missing = [c for c in required_cols if c not in detailed_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = detailed_df[required_cols].copy()
    is_neutral = df["WLoc"] == "N"

    winners = pd.DataFrame(
        {
            "Season": df["Season"].astype(int),
            "TeamID": df["WTeamID"].astype(int),
            "OppTeamID": df["LTeamID"].astype(int),
            "FGM": df["WFGM"].astype(float),
            "FGA": df["WFGA"].astype(float),
            "FGM3": df["WFGM3"].astype(float),
            "FTA": df["WFTA"].astype(float),
            "OffReb": df["WOR"].astype(float),
            "DefReb": df["WDR"].astype(float),
            "TO": df["WTO"].astype(float),
            "OppFGM": df["LFGM"].astype(float),
            "OppFGA": df["LFGA"].astype(float),
            "OppFGM3": df["LFGM3"].astype(float),
            "OppFTA": df["LFTA"].astype(float),
            "OppOffReb": df["LOR"].astype(float),
            "OppDefReb": df["LDR"].astype(float),
            "OppTO": df["LTO"].astype(float),
            "is_neutral": is_neutral.astype(bool),
        }
    )

    losers = pd.DataFrame(
        {
            "Season": df["Season"].astype(int),
            "TeamID": df["LTeamID"].astype(int),
            "OppTeamID": df["WTeamID"].astype(int),
            "FGM": df["LFGM"].astype(float),
            "FGA": df["LFGA"].astype(float),
            "FGM3": df["LFGM3"].astype(float),
            "FTA": df["LFTA"].astype(float),
            "OffReb": df["LOR"].astype(float),
            "DefReb": df["LDR"].astype(float),
            "TO": df["LTO"].astype(float),
            "OppFGM": df["WFGM"].astype(float),
            "OppFGA": df["WFGA"].astype(float),
            "OppFGM3": df["WFGM3"].astype(float),
            "OppFTA": df["WFTA"].astype(float),
            "OppOffReb": df["WOR"].astype(float),
            "OppDefReb": df["WDR"].astype(float),
            "OppTO": df["WTO"].astype(float),
            "is_neutral": is_neutral.astype(bool),
        }
    )

    team_games = pd.concat([winners, losers], ignore_index=True)
    team_games["Poss"] = (
        team_games["FGA"]
        - team_games["OffReb"]
        + team_games["TO"]
        + _POSSESSIONS_FTA_COEF * team_games["FTA"]
    )
    team_games["OppPoss"] = (
        team_games["OppFGA"]
        - team_games["OppOffReb"]
        + team_games["OppTO"]
        + _POSSESSIONS_FTA_COEF * team_games["OppFTA"]
    )

    # Per-game Four Factors.
    team_games["eFG_pct"] = (team_games["FGM"] + 0.5 * team_games["FGM3"]) / team_games["FGA"]
    team_games["TO_pct"] = team_games["TO"] / team_games["Poss"]
    team_games["OR_pct"] = team_games["OffReb"] / (team_games["OffReb"] + team_games["OppDefReb"])
    team_games["FTRate"] = team_games["FTA"] / team_games["FGA"]

    team_games["opp_eFG_pct"] = (team_games["OppFGM"] + 0.5 * team_games["OppFGM3"]) / team_games["OppFGA"]
    team_games["opp_TO_pct"] = team_games["OppTO"] / team_games["OppPoss"]
    team_games["opp_OR_pct"] = team_games["OppOffReb"] / (team_games["OppOffReb"] + team_games["DefReb"])
    team_games["opp_FTRate"] = team_games["OppFTA"] / team_games["OppFGA"]

    # Avoid infs from divide-by-zero by turning them into NaN (ignored by mean).
    metric_cols = [
        "eFG_pct",
        "TO_pct",
        "OR_pct",
        "FTRate",
        "opp_eFG_pct",
        "opp_TO_pct",
        "opp_OR_pct",
        "opp_FTRate",
    ]
    team_games[metric_cols] = team_games[metric_cols].replace([float("inf"), float("-inf")], pd.NA)

    grouped = team_games.groupby(["Season", "TeamID"], as_index=False)
    out = grouped[metric_cols].mean(numeric_only=True)

    # Clip eFG_pct bounds after aggregation.
    out["eFG_pct"] = out["eFG_pct"].clip(lower=_EFG_CLIP_MIN, upper=_EFG_CLIP_MAX)

    out = out[
        [
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
        ]
    ].copy()
    out = out.sort_values(["Season", "TeamID"], kind="mergesort").reset_index(drop=True)
    return out


def encode_matchups(
    matchups_df: pd.DataFrame,
    elo_dict: dict[tuple[int, int], float],
    efficiency_df: pd.DataFrame,
    four_factors_df: pd.DataFrame,
    massey_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    label_col: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Encode matchup rows into model-ready difference features.

    Sign convention (Team1 - Team2):
      - `elo_diff = elo[Season-1, Team1] - elo[Season-1, Team2]`
      - `seed_diff = seed[Team2] - seed[Team1]` (so positive means Team1 has better seed).
    """
    required_matchup_cols = {"Season", "Team1ID", "Team2ID"}
    missing = required_matchup_cols.difference(matchups_df.columns)
    if missing:
        raise ValueError(f"matchups_df missing columns: {sorted(missing)}")

    base = matchups_df[["Season", "Team1ID", "Team2ID"]].copy()

    # Elo lookup uses pre-tournament ratings (END of previous season).
    def _elo_value(season: int, team_id: int) -> float:
        return float(elo_dict.get((season, team_id), 0.0))

    seasons_prev = (base["Season"].astype(int) - 1).to_numpy(dtype=int)
    team1_ids = base["Team1ID"].astype(int).to_numpy(dtype=int)
    team2_ids = base["Team2ID"].astype(int).to_numpy(dtype=int)

    elo1 = np.array([_elo_value(int(s), int(t)) for s, t in zip(seasons_prev, team1_ids)], dtype=float)
    elo2 = np.array([_elo_value(int(s), int(t)) for s, t in zip(seasons_prev, team2_ids)], dtype=float)
    elo_diff = elo1 - elo2

    # Merge per-team features for Team1.
    eff_needed = {"Season", "TeamID", "AdjEM"}
    if not eff_needed.issubset(efficiency_df.columns):
        missing_eff = sorted(eff_needed.difference(efficiency_df.columns))
        raise ValueError(f"efficiency_df missing columns: {missing_eff}")
    eff = efficiency_df[["Season", "TeamID", "AdjEM"]].copy()

    base = base.merge(
        eff.rename(columns={"TeamID": "Team1ID", "AdjEM": "AdjEM_T1"}),
        on=["Season", "Team1ID"],
        how="left",
    ).merge(
        eff.rename(columns={"TeamID": "Team2ID", "AdjEM": "AdjEM_T2"}),
        on=["Season", "Team2ID"],
        how="left",
    )
    base[["AdjEM_T1", "AdjEM_T2"]] = base[["AdjEM_T1", "AdjEM_T2"]].fillna(0.0)

    ff_needed = {
        "Season",
        "TeamID",
        "eFG_pct",
        "TO_pct",
        "OR_pct",
        "FTRate",
        "opp_eFG_pct",
        "opp_TO_pct",
        "opp_OR_pct",
    }
    if not ff_needed.issubset(four_factors_df.columns):
        missing_ff = sorted(ff_needed.difference(four_factors_df.columns))
        raise ValueError(f"four_factors_df missing columns: {missing_ff}")
    ff = four_factors_df[list(ff_needed)].copy()

    base = base.merge(
        ff.rename(
            columns={
                "TeamID": "Team1ID",
                "eFG_pct": "eFG_pct_T1",
                "TO_pct": "TO_pct_T1",
                "OR_pct": "OR_pct_T1",
                "FTRate": "FTRate_T1",
                "opp_eFG_pct": "opp_eFG_pct_T1",
                "opp_TO_pct": "opp_TO_pct_T1",
                "opp_OR_pct": "opp_OR_pct_T1",
            }
        ),
        on=["Season", "Team1ID"],
        how="left",
    ).merge(
        ff.rename(
            columns={
                "TeamID": "Team2ID",
                "eFG_pct": "eFG_pct_T2",
                "TO_pct": "TO_pct_T2",
                "OR_pct": "OR_pct_T2",
                "FTRate": "FTRate_T2",
                "opp_eFG_pct": "opp_eFG_pct_T2",
                "opp_TO_pct": "opp_TO_pct_T2",
                "opp_OR_pct": "opp_OR_pct_T2",
            }
        ),
        on=["Season", "Team2ID"],
        how="left",
    )

    ff_value_cols = [
        "eFG_pct_T1",
        "eFG_pct_T2",
        "TO_pct_T1",
        "TO_pct_T2",
        "OR_pct_T1",
        "OR_pct_T2",
        "FTRate_T1",
        "FTRate_T2",
        "opp_eFG_pct_T1",
        "opp_eFG_pct_T2",
        "opp_TO_pct_T1",
        "opp_TO_pct_T2",
        "opp_OR_pct_T1",
        "opp_OR_pct_T2",
    ]
    base[ff_value_cols] = base[ff_value_cols].fillna(0.0)

    massey_needed = {"Season", "TeamID", "massey_consensus"}
    if not massey_needed.issubset(massey_df.columns):
        missing_m = sorted(massey_needed.difference(massey_df.columns))
        raise ValueError(f"massey_df missing columns: {missing_m}")
    ms = massey_df[["Season", "TeamID", "massey_consensus"]].copy()
    base = base.merge(
        ms.rename(columns={"TeamID": "Team1ID", "massey_consensus": "massey_T1"}),
        on=["Season", "Team1ID"],
        how="left",
    ).merge(
        ms.rename(columns={"TeamID": "Team2ID", "massey_consensus": "massey_T2"}),
        on=["Season", "Team2ID"],
        how="left",
    )
    base[["massey_T1", "massey_T2"]] = base[["massey_T1", "massey_T2"]].fillna(0.0)

    seeds_needed = {"Season", "TeamID", "Seed"}
    if not seeds_needed.issubset(seeds_df.columns):
        missing_s = sorted(seeds_needed.difference(seeds_df.columns))
        raise ValueError(f"seeds_df missing columns: {missing_s}")
    sd = seeds_df[["Season", "TeamID", "Seed"]].copy()
    base = base.merge(
        sd.rename(columns={"TeamID": "Team1ID", "Seed": "Seed_T1"}),
        on=["Season", "Team1ID"],
        how="left",
    ).merge(
        sd.rename(columns={"TeamID": "Team2ID", "Seed": "Seed_T2"}),
        on=["Season", "Team2ID"],
        how="left",
    )

    seed1 = base["Seed_T1"]
    seed2 = base["Seed_T2"]
    seed_diff = np.where(seed1.notna() & seed2.notna(), seed2.to_numpy() - seed1.to_numpy(), 0.0).astype(float)
    # Drop seed columns after we compute differences.
    base["seed_diff"] = seed_diff

    out = pd.DataFrame(
        {
            "elo_diff": elo_diff,
            "adjEM_diff": base["AdjEM_T1"] - base["AdjEM_T2"],
            "eFG_diff": base["eFG_pct_T1"] - base["eFG_pct_T2"],
            "TO_diff": base["TO_pct_T1"] - base["TO_pct_T2"],
            "OR_diff": base["OR_pct_T1"] - base["OR_pct_T2"],
            "FTRate_diff": base["FTRate_T1"] - base["FTRate_T2"],
            "opp_eFG_diff": base["opp_eFG_pct_T1"] - base["opp_eFG_pct_T2"],
            "opp_TO_diff": base["opp_TO_pct_T1"] - base["opp_TO_pct_T2"],
            "opp_OR_diff": base["opp_OR_pct_T1"] - base["opp_OR_pct_T2"],
            "massey_diff": base["massey_T1"] - base["massey_T2"],
            "seed_diff": base["seed_diff"],
        }
    )

    if label_col is None:
        out["label"] = np.nan
    else:
        out["label"] = pd.to_numeric(label_col, errors="coerce").astype(float).to_numpy()

    # Guarantee no NaNs in feature columns.
    feature_cols = [c for c in out.columns if c != "label"]
    if out[feature_cols].isna().any().any():
        out[feature_cols] = out[feature_cols].fillna(0.0)
        _logger.warning("Some feature values were missing; filled with 0.")

    return out


def encode_matchups_symmetric(
    matchups_df: pd.DataFrame,
    elo_dict: dict[tuple[int, int], float],
    efficiency_df: pd.DataFrame,
    four_factors_df: pd.DataFrame,
    massey_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    label_col: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Return both match orders with sign/label flips."""
    swapped = matchups_df.copy()
    swapped_team1 = swapped["Team1ID"].copy()
    swapped["Team1ID"] = swapped["Team2ID"]
    swapped["Team2ID"] = swapped_team1

    if label_col is None:
        labels_swapped = None
    else:
        # label=1 means Team1 won; swapped ordering flips the outcome.
        labels_swapped = 1.0 - pd.to_numeric(label_col, errors="coerce").astype(float)

    df1 = encode_matchups(
        matchups_df=matchups_df,
        elo_dict=elo_dict,
        efficiency_df=efficiency_df,
        four_factors_df=four_factors_df,
        massey_df=massey_df,
        seeds_df=seeds_df,
        label_col=label_col,
    )
    df2 = encode_matchups(
        matchups_df=swapped,
        elo_dict=elo_dict,
        efficiency_df=efficiency_df,
        four_factors_df=four_factors_df,
        massey_df=massey_df,
        seeds_df=seeds_df,
        label_col=labels_swapped,
    )
    return pd.concat([df1, df2], ignore_index=True)

