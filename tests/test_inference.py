from src.inference import normalize_team_pair


def test_normalize_team_pair_orders_ids() -> None:
    assert normalize_team_pair(1200, 1100) == (1100, 1200)
    assert normalize_team_pair(1100, 1200) == (1100, 1200)


def test_normalize_team_pair_rejects_equal() -> None:
    try:
        normalize_team_pair(5, 5)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
