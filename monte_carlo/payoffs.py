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
