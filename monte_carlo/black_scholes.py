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
