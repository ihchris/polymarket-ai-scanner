"""Local configuration. LIVE_DATA only reads public data."""
import os
from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(validate_default=True, validate_assignment=True, allow_inf_nan=False)
    mode: str = Field(default_factory=lambda: os.getenv('SCANNER_MODE', 'DEMO'), pattern='^(DEMO|LIVE_DATA)$')
    refresh_seconds: int = Field(60, ge=5)
    minimum_liquidity: float = Field(100, ge=0)
    minimum_edge: float = Field(.02, ge=0, le=1)
    simulations: int = Field(10000, ge=500, le=100000)
    estimated_fee_per_share: float | None = Field(None, ge=0, le=1)
    slippage_per_share: float = Field(.002, ge=0, le=1)
    maximum_age_seconds: int = Field(300, ge=1)
    order_size: float = Field(10, gt=0)
    database: str = Field(default_factory=lambda: os.getenv('SCANNER_DATABASE', 'data/scanner.sqlite3'))
