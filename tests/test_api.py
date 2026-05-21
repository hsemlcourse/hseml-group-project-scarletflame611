"""
Smoke-тесты для FastAPI. Проверяют что эндпоинты отвечают и возвращают валидный JSON.
Запуск: pytest tests/test_api.py -v
API должен быть запущен на localhost:8000.
"""

import httpx
import pytest

BASE = "http://localhost:8000"


@pytest.fixture(scope="session")
def client():
    return httpx.Client(base_url=BASE, timeout=30)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    data = r.json()
    assert "skaters" in data
    assert "goalies" in data
    assert data["skaters"]["test_r2"] > 0.5


def test_predict_skater_default(client):
    r = client.post("/predict/skater", json={})
    assert r.status_code == 200
    data = r.json()
    assert "predicted_cap_hit" in data
    assert data["predicted_cap_hit"] > 0
    assert "salary_tier" in data
    assert len(data["shap_top5"]) == 5
    assert len(data["similar_players"]) > 0


def test_predict_goalie_default(client):
    r = client.post("/predict/goalie", json={})
    assert r.status_code == 200
    data = r.json()
    assert "predicted_cap_hit" in data
    assert data["predicted_cap_hit"] > 0
    assert "salary_tier" in data


def test_predict_skater_star(client):
    r = client.post(
        "/predict/skater",
        json={
            "positionCode": "C",
            "gamesPlayed": 82,
            "goals": 50,
            "assists": 70,
            "points": 120,
            "timeOnIcePerGame": 1320,
            "ppPoints": 40,
            "ppTimeOnIcePerGame": 300,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_cap_hit"] > 5_000_000
    assert data["salary_tier"] in ["Top-6", "Star", "Franchise"]


def test_predict_skater_entry_level(client):
    r = client.post(
        "/predict/skater",
        json={
            "positionCode": "L",
            "gamesPlayed": 20,
            "goals": 2,
            "assists": 3,
            "points": 5,
            "timeOnIcePerGame": 600,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_cap_hit"] < 5_000_000


def test_salary_tier_values(client):
    r = client.post("/predict/skater", json={})
    data = r.json()
    valid_tiers = {"Entry-level", "Bottom-6", "Middle-6", "Top-6", "Star", "Franchise"}
    assert data["salary_tier"] in valid_tiers


def test_shap_directions(client):
    r = client.post("/predict/skater", json={})
    data = r.json()
    for feat in data["shap_top5"]:
        assert feat["direction"] in ["up", "down"]
        assert feat["shap_value"] != 0


def test_search_found(client):
    r = client.get("/players/search", params={"name": "Pettersson", "type": "skater"})
    assert r.status_code == 200
    data = r.json()
    assert len(data) > 0
    assert any("Pettersson" in p["name"] for p in data)


def test_search_not_found(client):
    r = client.get("/players/search", params={"name": "zzzznotaplayer", "type": "skater"})
    assert r.status_code == 404


def test_search_goalie(client):
    r = client.get("/players/search", params={"name": "Vasilevskiy", "type": "goalie"})
    assert r.status_code == 200
    data = r.json()
    assert len(data) > 0


def test_predict_with_prev_season(client):
    r = client.post(
        "/predict/skater",
        json={
            "goals": 20,
            "assists": 30,
            "points": 50,
            "prev_season": {
                "goals": 15,
                "assists": 25,
                "points": 40,
            },
        },
    )
    assert r.status_code == 200
    assert r.json()["predicted_cap_hit"] > 0


def test_batch_upload(client):
    csv_content = (
        "skaterFullName,gamesPlayed,goals,assists,points,plusMinus,"
        "penaltyMinutes,shots,timeOnIcePerGame,ppGoals,ppPoints,"
        "ppTimeOnIcePerGame,hits,blockedShots,heightInInches,weightInPounds,"
        "birthCountry,draftYear\n"
        "Test Player,60,15,25,40,0,20,120,900,3,8,90,50,30,73,200,CAN,2018\n"
        "Test Player 2,70,20,30,50,5,15,150,1000,5,12,120,40,25,74,210,USA,2017\n"
    )
    r = client.post(
        "/predict/batch",
        files={"file": ("test.csv", csv_content.encode(), "text/csv")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert all(
        p["predicted_cap_hit"] > 0 for p in data["predictions"] if p["predicted_cap_hit"] != 0
    )


def test_static_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "NHL" in r.text
