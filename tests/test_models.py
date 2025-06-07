import pytest
import numpy as np
from pydantic import ValidationError
from monte_carlo.models import MarketData, MonteCarloConfig, SimulationResult, AbstractPayoff


class TestMarketData:
    """Test MarketData model validation and functionality."""
    
    def test_valid_market_data(self):
        """Test creation of valid market data."""
        market_data = MarketData(
            spot_price=100.0,
            volatility=0.2,
            risk_free_rate=0.05,
            dividend_yield=0.02
        )
        
        assert market_data.spot_price == 100.0
        assert market_data.volatility == 0.2
        assert market_data.risk_free_rate == 0.05
        assert market_data.dividend_yield == 0.02
    
    def test_default_dividend_yield(self):
        """Test default dividend yield is zero."""
        market_data = MarketData(
            spot_price=100.0,
            volatility=0.2,
            risk_free_rate=0.05
        )
        assert market_data.dividend_yield == 0.0
    
    def test_negative_spot_price_raises_error(self):
        """Test that negative spot price raises validation error."""
        with pytest.raises(ValidationError, match="Value must be positive"):
            MarketData(
                spot_price=-100.0,
                volatility=0.2,
                risk_free_rate=0.05
            )
    
    def test_zero_spot_price_raises_error(self):
        """Test that zero spot price raises validation error."""
        with pytest.raises(ValidationError, match="Value must be positive"):
            MarketData(
                spot_price=0.0,
                volatility=0.2,
                risk_free_rate=0.05
            )
    
    def test_negative_volatility_raises_error(self):
        """Test that negative volatility raises validation error."""
        with pytest.raises(ValidationError, match="Value must be positive"):
            MarketData(
                spot_price=100.0,
                volatility=-0.2,
                risk_free_rate=0.05
            )
    
    def test_negative_risk_free_rate_raises_error(self):
        """Test that negative risk-free rate raises validation error."""
        with pytest.raises(ValidationError, match="Value must be positive"):
            MarketData(
                spot_price=100.0,
                volatility=0.2,
                risk_free_rate=-0.05
            )
    
    def test_negative_dividend_yield_raises_error(self):
        """Test that negative dividend yield raises validation error."""
        with pytest.raises(ValidationError, match="Dividend yield must be non-negative"):
            MarketData(
                spot_price=100.0,
                volatility=0.2,
                risk_free_rate=0.05,
                dividend_yield=-0.01
            )


class TestMonteCarloConfig:
    """Test MonteCarloConfig model validation and functionality."""
    
    def test_valid_config(self):
        """Test creation of valid Monte Carlo configuration."""
        config = MonteCarloConfig(
            num_simulations=50000,
            num_time_steps=100,
            random_seed=42
        )
        
        assert config.num_simulations == 50000
        assert config.num_time_steps == 100
        assert config.random_seed == 42
    
    def test_default_values(self):
        """Test default configuration values."""
        config = MonteCarloConfig()
        
        assert config.num_simulations == 100000
        assert config.num_time_steps == 252
        assert config.random_seed is None
    
    def test_negative_num_simulations_raises_error(self):
        """Test that negative number of simulations raises validation error."""
        with pytest.raises(ValidationError, match="Value must be positive integer"):
            MonteCarloConfig(num_simulations=-1000)
    
    def test_zero_num_simulations_raises_error(self):
        """Test that zero number of simulations raises validation error."""
        with pytest.raises(ValidationError, match="Value must be positive integer"):
            MonteCarloConfig(num_simulations=0)
    
    def test_negative_num_time_steps_raises_error(self):
        """Test that negative number of time steps raises validation error."""
        with pytest.raises(ValidationError, match="Value must be positive integer"):
            MonteCarloConfig(num_time_steps=-100)
    
    def test_zero_num_time_steps_raises_error(self):
        """Test that zero number of time steps raises validation error."""
        with pytest.raises(ValidationError, match="Value must be positive integer"):
            MonteCarloConfig(num_time_steps=0)


class TestSimulationResult:
    """Test SimulationResult dataclass."""
    
    def test_simulation_result_creation(self):
        """Test creation of simulation result."""
        result = SimulationResult(
            price=10.5,
            standard_error=0.05,
            confidence_interval=(10.4, 10.6),
            num_simulations=100000
        )
        
        assert result.price == 10.5
        assert result.standard_error == 0.05
        assert result.confidence_interval == (10.4, 10.6)
        assert result.num_simulations == 100000


class DummyPayoff(AbstractPayoff):
    """Dummy payoff implementation for testing."""
    
    def calculate_payoff(self, spot_paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Return simple payoff based on final spot prices."""
        return np.maximum(spot_paths[:, -1] - 100.0, 0.0)


class TestAbstractPayoff:
    """Test AbstractPayoff base class."""
    
    def test_abstract_payoff_implementation(self):
        """Test that AbstractPayoff can be implemented and called."""
        payoff = DummyPayoff()
        
        spot_paths = np.array([[90.0, 95.0, 105.0], [100.0, 110.0, 120.0]])
        times = np.array([0.0, 0.5, 1.0])
        
        result = payoff(spot_paths, times)
        expected = np.array([5.0, 20.0])  # max(105-100, 0), max(120-100, 0)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_calculate_payoff_directly(self):
        """Test calling calculate_payoff method directly."""
        payoff = DummyPayoff()
        
        spot_paths = np.array([[90.0, 95.0, 105.0]])
        times = np.array([0.0, 0.5, 1.0])
        
        result = payoff.calculate_payoff(spot_paths, times)
        expected = np.array([5.0])
        
        np.testing.assert_array_equal(result, expected)
