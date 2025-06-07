from dataclasses import dataclass
from typing import List, Optional, Protocol, Union
from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, validator


class MarketData(BaseModel):
    """Market data for Black-Scholes pricing."""
    
    spot_price: float
    volatility: float
    risk_free_rate: float
    dividend_yield: float = 0.0
    
    @validator('spot_price', 'volatility', 'risk_free_rate')
    def must_be_positive(cls, v: float) -> float:
        """Validate that critical parameters are positive."""
        if v <= 0:
            raise ValueError('Value must be positive')
        return v
    
    @validator('dividend_yield')
    def dividend_yield_non_negative(cls, v: float) -> float:
        """Validate that dividend yield is non-negative."""
        if v < 0:
            raise ValueError('Dividend yield must be non-negative')
        return v


class MonteCarloConfig(BaseModel):
    """Configuration for Monte Carlo simulation."""
    
    num_simulations: int = 100000
    num_time_steps: int = 252
    random_seed: Optional[int] = None
    
    @validator('num_simulations', 'num_time_steps')
    def must_be_positive_int(cls, v: int) -> int:
        """Validate that simulation parameters are positive integers."""
        if v <= 0:
            raise ValueError('Value must be positive integer')
        return v


@dataclass
class SimulationResult:
    """Result of Monte Carlo simulation."""
    
    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    num_simulations: int


class Payoff(Protocol):
    """Protocol for payoff functions."""
    
    def __call__(self, spot_paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Calculate payoff given spot price paths and time grid."""
        ...


class AbstractPayoff(ABC):
    """Abstract base class for payoff implementations."""
    
    @abstractmethod
    def calculate_payoff(self, spot_paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Calculate payoff given spot price paths and time grid.
        
        Args:
            spot_paths: Array of shape (num_simulations, num_time_steps) with spot prices
            times: Array of shape (num_time_steps,) with time points
            
        Returns:
            Array of shape (num_simulations,) with payoff values
        """
        pass
    
    def __call__(self, spot_paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Make payoff callable."""
        return self.calculate_payoff(spot_paths, times)
