from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, field_validator
from src.utils.validation import utc


class ProbabilityEstimate(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    probability: float = Field(ge=0, le=1)
    lower: float = Field(ge=0, le=1)
    upper: float = Field(ge=0, le=1)
    timestamp: datetime
    model_version: str
    independent: bool = True

    @field_validator('timestamp')
    @classmethod
    def aware(cls, v):
        return utc(v)


def from_forecast(forecast, side='YES'):
    # Monte Carlo error alone is NOT a model credible interval for P(win).
    # Build a conservative robustness interval by +/- 1pp systematic vote-share
    # sensitivity across all posterior scenarios. This is a scenario range,
    # not a calibrated credible interval of win probability.
    import numpy as np
    p = forecast.win_probability
    lower = float(np.mean(forecast.samples > .51))
    upper = float(np.mean(forecast.samples > .49))
    if side == 'NO':
        p, lower, upper = 1-p, 1-upper, 1-lower
    return ProbabilityEstimate(probability=p, lower=min(lower, p), upper=max(upper, p),
        timestamp=forecast.timestamp, model_version=forecast.model_version)
