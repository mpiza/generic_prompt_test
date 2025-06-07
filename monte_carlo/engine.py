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
