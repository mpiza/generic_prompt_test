#!monte_carlo/models.py
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


#!monte_carlo/payoffs.py
import numpy as np
from typing import Union, List
from .models import AbstractPayoff


class EuropeanCallPayoff(AbstractPayoff):
    """European call option payoff."""
    
    def __init__(self, strike: float, maturity: float):
        """Initialize European call payoff.
        
        Args:
            strike: Strike price of the option
            maturity: Time to maturity in years
        """
        if strike <= 0:
            raise ValueError('Strike must be positive')
        if maturity <= 0:
            raise ValueError('Maturity must be positive')
            
        self._strike = strike
        self._maturity = maturity
    
    @property
    def strike(self) -> float:
        """Strike price of the option."""
        return self._strike
    
    @property
    def maturity(self) -> float:
        """Time to maturity in years."""
        return self._maturity
    
    def calculate_payoff(self, spot_paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Calculate European call payoff.
        
        Args:
            spot_paths: Array of shape (num_simulations, num_time_steps)
            times: Array of shape (num_time_steps,)
            
        Returns:
            Array of shape (num_simulations,) with call payoffs
        """
        maturity_idx = np.argmin(np.abs(times - self._maturity))
        final_spots = spot_paths[:, maturity_idx]
        return np.maximum(final_spots - self._strike, 0.0)


class EuropeanPutPayoff(AbstractPayoff):
    """European put option payoff."""
    
    def __init__(self, strike: float, maturity: float):
        """Initialize European put payoff.
        
        Args:
            strike: Strike price of the option
            maturity: Time to maturity in years
        """
        if strike <= 0:
            raise ValueError('Strike must be positive')
        if maturity <= 0:
            raise ValueError('Maturity must be positive')
            
        self._strike = strike
        self._maturity = maturity
    
    @property
    def strike(self) -> float:
        """Strike price of the option."""
        return self._strike
    
    @property
    def maturity(self) -> float:
        """Time to maturity in years."""
        return self._maturity
    
    def calculate_payoff(self, spot_paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Calculate European put payoff.
        
        Args:
            spot_paths: Array of shape (num_simulations, num_time_steps)
            times: Array of shape (num_time_steps,)
            
        Returns:
            Array of shape (num_simulations,) with put payoffs
        """
        maturity_idx = np.argmin(np.abs(times - self._maturity))
        final_spots = spot_paths[:, maturity_idx]
        return np.maximum(self._strike - final_spots, 0.0)


class AsianCallPayoff(AbstractPayoff):
    """Asian call option payoff (arithmetic average)."""
    
    def __init__(self, strike: float, maturity: float):
        """Initialize Asian call payoff.
        
        Args:
            strike: Strike price of the option
            maturity: Time to maturity in years
        """
        if strike <= 0:
            raise ValueError('Strike must be positive')
        if maturity <= 0:
            raise ValueError('Maturity must be positive')
            
        self._strike = strike
        self._maturity = maturity
    
    @property
    def strike(self) -> float:
        """Strike price of the option."""
        return self._strike
    
    @property
    def maturity(self) -> float:
        """Time to maturity in years."""
        return self._maturity
    
    def calculate_payoff(self, spot_paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Calculate Asian call payoff.
        
        Args:
            spot_paths: Array of shape (num_simulations, num_time_steps)
            times: Array of shape (num_time_steps,)
            
        Returns:
            Array of shape (num_simulations,) with Asian call payoffs
        """
        maturity_idx = np.argmin(np.abs(times - self._maturity))
        relevant_paths = spot_paths[:, :maturity_idx + 1]
        average_spots = np.mean(relevant_paths, axis=1)
        return np.maximum(average_spots - self._strike, 0.0)


class BarrierCallPayoff(AbstractPayoff):
    """Barrier call option payoff (knock-out)."""
    
    def __init__(self, strike: float, maturity: float, barrier: float):
        """Initialize barrier call payoff.
        
        Args:
            strike: Strike price of the option
            maturity: Time to maturity in years
            barrier: Barrier level (knock-out if spot hits this level)
        """
        if strike <= 0:
            raise ValueError('Strike must be positive')
        if maturity <= 0:
            raise ValueError('Maturity must be positive')
        if barrier <= 0:
            raise ValueError('Barrier must be positive')
            
        self._strike = strike
        self._maturity = maturity
        self._barrier = barrier
    
    @property
    def strike(self) -> float:
        """Strike price of the option."""
        return self._strike
    
    @property
    def maturity(self) -> float:
        """Time to maturity in years."""
        return self._maturity
    
    @property
    def barrier(self) -> float:
        """Barrier level."""
        return self._barrier
    
    def calculate_payoff(self, spot_paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Calculate barrier call payoff.
        
        Args:
            spot_paths: Array of shape (num_simulations, num_time_steps)
            times: Array of shape (num_time_steps,)
            
        Returns:
            Array of shape (num_simulations,) with barrier call payoffs
        """
        maturity_idx = np.argmin(np.abs(times - self._maturity))
        relevant_paths = spot_paths[:, :maturity_idx + 1]
        
        barrier_hit = np.any(relevant_paths >= self._barrier, axis=1)
        final_spots = spot_paths[:, maturity_idx]
        call_payoffs = np.maximum(final_spots - self._strike, 0.0)
        
        return np.where(barrier_hit, 0.0, call_payoffs)


#!monte_carlo/black_scholes.py
import numpy as np
import logging
from typing import Optional
from .models import MarketData, MonteCarloConfig

logger = logging.getLogger(__name__)


class BlackScholesPathGenerator:
    """Generates stock price paths using Black-Scholes model."""
    
    def __init__(self, market_data: MarketData, config: MonteCarloConfig):
        """Initialize Black-Scholes path generator.
        
        Args:
            market_data: Market parameters for the model
            config: Monte Carlo simulation configuration
        """
        self._market_data = market_data
        self._config = config
        self._rng = np.random.RandomState(config.random_seed)
        
        logger.info(f"Initialized Black-Scholes generator with {config.num_simulations} simulations")
    
    @property
    def market_data(self) -> MarketData:
        """Market data used for simulation."""
        return self._market_data
    
    @property
    def config(self) -> MonteCarloConfig:
        """Monte Carlo configuration."""
        return self._config
    
    def generate_paths(self, maturity: float) -> tuple[np.ndarray, np.ndarray]:
        """Generate stock price paths using geometric Brownian motion.
        
        Args:
            maturity: Maximum time to simulate (in years)
            
        Returns:
            Tuple of (spot_paths, time_grid) where:
            - spot_paths: Array of shape (num_simulations, num_time_steps)
            - time_grid: Array of shape (num_time_steps,) with time points
        """
        if maturity <= 0:
            raise ValueError('Maturity must be positive')
            
        dt = maturity / (self._config.num_time_steps - 1)
        time_grid = np.linspace(0, maturity, self._config.num_time_steps)
        
        # Pre-compute constants for efficiency
        drift = (self._market_data.risk_free_rate - 
                self._market_data.dividend_yield - 
                0.5 * self._market_data.volatility ** 2) * dt
        
        vol_sqrt_dt = self._market_data.volatility * np.sqrt(dt)
        
        # Generate random numbers
        random_increments = self._rng.normal(
            0, 1, (self._config.num_simulations, self._config.num_time_steps - 1)
        )
        
        # Calculate log returns
        log_returns = drift + vol_sqrt_dt * random_increments
        
        # Initialize paths
        log_spots = np.zeros((self._config.num_simulations, self._config.num_time_steps))
        log_spots[:, 0] = np.log(self._market_data.spot_price)
        
        # Generate paths using cumulative sum for efficiency
        log_spots[:, 1:] = log_spots[:, 0:1] + np.cumsum(log_returns, axis=1)
        spot_paths = np.exp(log_spots)
        
        logger.debug(f"Generated {self._config.num_simulations} paths with {self._config.num_time_steps} time steps")
        
        return spot_paths, time_grid


#!monte_carlo/engine.py
import numpy as np
import logging
from typing import Union
from scipy import stats
from .models import MarketData, MonteCarloConfig, SimulationResult, Payoff
from .black_scholes import BlackScholesPathGenerator

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    """Monte Carlo engine for pricing derivatives with arbitrary payoffs."""
    
    def __init__(self, market_data: MarketData, config: MonteCarloConfig):
        """Initialize Monte Carlo engine.
        
        Args:
            market_data: Market parameters
            config: Simulation configuration
        """
        self._market_data = market_data
        self._config = config
        self._path_generator = BlackScholesPathGenerator(market_data, config)
        
        logger.info("Initialized Monte Carlo engine")
    
    @property
    def market_data(self) -> MarketData:
        """Market data used for pricing."""
        return self._market_data
    
    @property
    def config(self) -> MonteCarloConfig:
        """Monte Carlo configuration."""
        return self._config
    
    def price(self, payoff: Payoff, maturity: float, confidence_level: float = 0.95) -> SimulationResult:
        """Price a derivative with given payoff function.
        
        Args:
            payoff: Function that calculates payoff given spot paths and times
            maturity: Time to maturity in years
            confidence_level: Confidence level for confidence interval
            
        Returns:
            SimulationResult with price, standard error, and confidence interval
            
        Raises:
            ValueError: If maturity is not positive or confidence level is invalid
        """
        if maturity <= 0:
            raise ValueError('Maturity must be positive')
        if not 0 < confidence_level < 1:
            raise ValueError('Confidence level must be between 0 and 1')
            
        try:
            logger.info(f"Starting Monte Carlo pricing with maturity {maturity:.4f}")
            
            # Generate paths
            spot_paths, time_grid = self._path_generator.generate_paths(maturity)
            
            # Calculate payoffs
            payoffs = payoff(spot_paths, time_grid)
            
            if not isinstance(payoffs, np.ndarray) or payoffs.shape != (self._config.num_simulations,):
                raise ValueError(f"Payoff function must return array of shape ({self._config.num_simulations},)")
            
            # Discount payoffs to present value
            discount_factor = np.exp(-self._market_data.risk_free_rate * maturity)
            discounted_payoffs = payoffs * discount_factor
            
            # Calculate statistics
            mean_payoff = np.mean(discounted_payoffs)
            std_payoff = np.std(discounted_payoffs, ddof=1)
            standard_error = std_payoff / np.sqrt(self._config.num_simulations)
            
            # Calculate confidence interval
            alpha = 1 - confidence_level
            z_score = stats.norm.ppf(1 - alpha / 2)
            margin_of_error = z_score * standard_error
            
            confidence_interval = (
                mean_payoff - margin_of_error,
                mean_payoff + margin_of_error
            )
            
            result = SimulationResult(
                price=mean_payoff,
                standard_error=standard_error,
                confidence_interval=confidence_interval,
                num_simulations=self._config.num_simulations
            )
            
            logger.info(f"Pricing completed: price={result.price:.6f}, std_error={result.standard_error:.6f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during Monte Carlo pricing: {e}")
            raise
    
    def price_multiple(self, payoffs: list[Payoff], maturity: float, 
                      confidence_level: float = 0.95) -> list[SimulationResult]:
        """Price multiple derivatives using the same underlying paths.
        
        Args:
            payoffs: List of payoff functions
            maturity: Time to maturity in years
            confidence_level: Confidence level for confidence intervals
            
        Returns:
            List of SimulationResult objects
        """
        if not payoffs:
            raise ValueError('Payoffs list cannot be empty')
            
        logger.info(f"Pricing {len(payoffs)} derivatives simultaneously")
        
        # Generate paths once for all payoffs
        spot_paths, time_grid = self._path_generator.generate_paths(maturity)
        discount_factor = np.exp(-self._market_data.risk_free_rate * maturity)
        
        results = []
        for i, payoff in enumerate(payoffs):
            try:
                payoff_values = payoff(spot_paths, time_grid)
                discounted_payoffs = payoff_values * discount_factor
                
                mean_payoff = np.mean(discounted_payoffs)
                std_payoff = np.std(discounted_payoffs, ddof=1)
                standard_error = std_payoff / np.sqrt(self._config.num_simulations)
                
                alpha = 1 - confidence_level
                z_score = stats.norm.ppf(1 - alpha / 2)
                margin_of_error = z_score * standard_error
                
                confidence_interval = (
                    mean_payoff - margin_of_error,
                    mean_payoff + margin_of_error
                )
                
                result = SimulationResult(
                    price=mean_payoff,
                    standard_error=standard_error,
                    confidence_interval=confidence_interval,
                    num_simulations=self._config.num_simulations
                )
                
                results.append(result)
                logger.debug(f"Payoff {i+1}/{len(payoffs)} priced: {result.price:.6f}")
                
            except Exception as e:
                logger.error(f"Error pricing payoff {i+1}: {e}")
                raise
        
        return results


#!monte_carlo/__init__.py
"""Monte Carlo engine for pricing derivatives using Black-Scholes models."""

from .models import MarketData, MonteCarloConfig, SimulationResult
from .payoffs import EuropeanCallPayoff, EuropeanPutPayoff, AsianCallPayoff, BarrierCallPayoff
from .engine import MonteCarloEngine
from .black_scholes import BlackScholesPathGenerator

__version__ = "1.0.0"
__all__ = [
    "MarketData",
    "MonteCarloConfig", 
    "SimulationResult",
    "EuropeanCallPayoff",
    "EuropeanPutPayoff",
    "AsianCallPayoff",
    "BarrierCallPayoff",
    "MonteCarloEngine",
    "BlackScholesPathGenerator",
]
