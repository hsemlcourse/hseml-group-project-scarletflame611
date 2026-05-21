"""
Pydantic-схемы для FastAPI.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class PrevSeasonSkater(BaseModel):
    """Статистика прошлого сезона для лаговых фич. Всё опционально."""
    goals: Optional[float] = None
    assists: Optional[float] = None
    points: Optional[float] = None
    timeOnIcePerGame: Optional[float] = None
    points_per_60: Optional[float] = None
    goals_per_60: Optional[float] = None
    assists_per_60: Optional[float] = None
    hits: Optional[float] = None
    blockedShots: Optional[float] = None
    plusMinus: Optional[float] = None
    ppPoints: Optional[float] = None
    cap_hit: Optional[float] = None


class PrevSeasonGoalie(BaseModel):
    savePct: Optional[float] = None
    goalsAgainstAverage: Optional[float] = None
    wins: Optional[float] = None
    shutouts: Optional[float] = None
    gamesStarted: Optional[float] = None
    gsaa_proxy: Optional[float] = None
    win_rate: Optional[float] = None
    cap_hit: Optional[float] = None


class SkaterInput(BaseModel):
    positionCode: Literal["C", "L", "R", "D"] = Field("C", description="Позиция")
    season_year: int = Field(2026, description="Год сезона")

    gamesPlayed: int = Field(60, ge=1, le=82)
    goals: int = Field(0, ge=0)
    assists: int = Field(0, ge=0)
    points: int = Field(0, ge=0)
    plusMinus: int = Field(0)
    penaltyMinutes: int = Field(0, ge=0)
    shots: int = Field(0, ge=0)
    shootingPct: float = Field(0.0, ge=0.0, le=100.0)
    timeOnIcePerGame: float = Field(900.0, ge=0.0, description="Секунды")

    ppGoals: int = Field(0, ge=0)
    ppPoints: int = Field(0, ge=0)
    ppAssists: int = Field(0, ge=0)
    ppShots: int = Field(0, ge=0)
    ppTimeOnIcePerGame: float = Field(0.0, ge=0.0)
    ppGoalsPer60: float = Field(0.0, ge=0.0)
    ppPointsPer60: float = Field(0.0, ge=0.0)
    ppShotsPer60: float = Field(0.0, ge=0.0)
    ppShootingPct: float = Field(0.0, ge=0.0)
    ppPrimaryAssists: int = Field(0, ge=0)
    ppSecondaryAssists: int = Field(0, ge=0)
    ppGoalsForPer60: float = Field(0.0, ge=0.0)
    ppIndividualSatForPer60: float = Field(0.0, ge=0.0)
    ppPrimaryAssistsPer60: float = Field(0.0, ge=0.0)
    ppSecondaryAssistsPer60: float = Field(0.0, ge=0.0)

    hits: int = Field(0, ge=0)
    hitsPer60: float = Field(0.0, ge=0.0)
    blockedShots: int = Field(0, ge=0)
    blockedShotsPer60: float = Field(0.0, ge=0.0)
    giveaways: int = Field(0, ge=0)
    giveawaysPer60: float = Field(0.0, ge=0.0)
    takeaways: int = Field(0, ge=0)
    takeawaysPer60: float = Field(0.0, ge=0.0)
    totalShotAttempts: int = Field(0, ge=0)
    shotAttemptsBlocked: int = Field(0, ge=0)

    missedShots: int = Field(0, ge=0)
    missedShotCrossbar: int = Field(0, ge=0)
    missedShotGoalpost: int = Field(0, ge=0)
    missedShotOverNet: int = Field(0, ge=0)
    missedShotShort: int = Field(0, ge=0)
    missedShotFailedBankAttempt: int = Field(0, ge=0)

    emptyNetGoals: int = Field(0, ge=0)
    emptyNetAssists: int = Field(0, ge=0)
    emptyNetPoints: int = Field(0, ge=0)
    firstGoals: int = Field(0, ge=0)
    gameWinningGoals: int = Field(0, ge=0)
    otGoals: int = Field(0, ge=0)
    shGoals: int = Field(0, ge=0)
    shPoints: int = Field(0, ge=0)

    faceoffWinPct: float = Field(0.0, ge=0.0, le=100.0)
    totalFaceoffs: int = Field(0, ge=0)
    evFaceoffs: int = Field(0, ge=0)
    evFaceoffPct: float = Field(0.0, ge=0.0)
    offensiveZoneFaceoffs: int = Field(0, ge=0)
    offensiveZoneFaceoffPct: float = Field(0.0, ge=0.0)
    defensiveZoneFaceoffs: int = Field(0, ge=0)
    defensiveZoneFaceoffPct: float = Field(0.0, ge=0.0)
    neutralZoneFaceoffs: int = Field(0, ge=0)
    neutralZoneFaceoffPct: float = Field(0.0, ge=0.0)
    ppFaceoffs: int = Field(0, ge=0)
    ppFaceoffPct: float = Field(0.0, ge=0.0)
    shFaceoffs: int = Field(0, ge=0)
    shFaceoffPct: float = Field(0.0, ge=0.0)

    heightInInches: float = Field(73.0, ge=60.0, le=84.0)
    weightInPounds: float = Field(200.0, ge=150.0, le=280.0)
    birthCountry: str = Field("CAN")
    draftYear: Optional[int] = None
    draftRound: Optional[int] = None
    draftOverall: Optional[int] = None

    prev_season: Optional[PrevSeasonSkater] = None


class GoalieInput(BaseModel):
    season_year: int = Field(2026)

    gamesPlayed: int = Field(40, ge=1, le=82)
    gamesStarted: int = Field(35, ge=0)
    wins: int = Field(0, ge=0)
    losses: int = Field(0, ge=0)
    otLosses: int = Field(0, ge=0)
    savePct: float = Field(0.910, ge=0.0, le=1.0)
    goalsAgainst: int = Field(0, ge=0)
    goalsAgainstAverage: float = Field(2.8, ge=0.0)
    shotsAgainst: int = Field(0, ge=0)
    shutouts: int = Field(0, ge=0)
    qualityStart: int = Field(0, ge=0)
    qualityStartsPct: float = Field(0.0, ge=0.0, le=1.0)
    completeGamePct: float = Field(0.0, ge=0.0, le=1.0)
    goalsFor: int = Field(0, ge=0)
    goalsForAverage: float = Field(0.0, ge=0.0)
    shotsAgainstPer60: float = Field(0.0, ge=0.0)
    incompleteGames: int = Field(0, ge=0)

    heightInInches: float = Field(74.0, ge=60.0, le=84.0)
    weightInPounds: float = Field(195.0, ge=150.0, le=280.0)
    birthCountry: str = Field("CAN")
    draftYear: Optional[int] = None
    draftRound: Optional[int] = None
    draftOverall: Optional[int] = None

    prev_season: Optional[PrevSeasonGoalie] = None


class ShapFeature(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: Literal["up", "down"]


class SimilarPlayer(BaseModel):
    name: str
    position: str
    team: str
    cap_hit: float
    cap_hit_m: str
    key_stat: str
    key_stat_value: float


class PredictionOutput(BaseModel):
    predicted_cap_hit: float
    predicted_cap_hit_m: str
    salary_tier: str
    tier_range: str
    shap_top5: list[ShapFeature]
    similar_players: list[SimilarPlayer]


class PlayerSearchResult(BaseModel):
    player_id: int
    name: str
    position: str
    team: str
    season_year: int
    cap_hit: float
    cap_hit_m: str
    stats: dict


class BatchPredictionRow(BaseModel):
    name: str
    predicted_cap_hit: float
    predicted_cap_hit_m: str
    salary_tier: str


class BatchPredictionOutput(BaseModel):
    count: int
    predictions: list[BatchPredictionRow]
    tier_distribution: dict[str, int]