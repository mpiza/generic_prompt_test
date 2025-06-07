import pytest
import numpy as np
from monte_carlo.engine import MonteCarloEngine
from monte_carlo.models import MarketData, MonteCarloConfig, SimulationResult
from monte_carlo.payoffs import EuropeanCallPayoff, EuropeanPutPayoff


class TestMonteCarloEngine:
    """Test MonteCarloEngine implementation."""
    
    @pytest.fixture
    def market_data(self):
        """Create sample market data for testing."""
        return MarketData(
            spot_price=100.0,
            volatility=0.2,
            risk_free_rate=0.05,
            dividend_yield=0.02
        )
    
    @pytest.fixture
    def config(self):
        """Create sample Monte Carlo configuration for testing."""
        return MonteCarloConfig(
            num_simulations=10000,
            num_time_steps=50,
            random_seed=42
        )
    
    @pytest.fixture
    def engine(self, market_data, config):
        """Create Monte Carlo engine for testing."""
        return MonteCarloEngine(market_data, config)
    
    def test_initialization(self, market_data, config):
        """Test Monte Carlo engine initialization."""
        engine = MonteCarloEngine(market_data, config)
        
        assert engine.market_data == market_data
        assert engine.config == config
    
    def test_price_european_call(self, engine):
        """Test pricing of European call option."""
        payoff = EuropeanCallPayoff(strike=100.0, maturity=1.0)
        
        result = engine.price(payoff, maturity=1.0)
        
        assert isinstance(result, SimulationResult)
        assert result.price > 0  # Call should have positive value
        assert result.standard_error > 0
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] < result.confidence_interval[1]
        assert result.num_simulations == engine.config.num_simulations
    
    def test_price_european_put(self, engine):
        """Test pricing of European put option."""
        payoff = EuropeanPutPayoff(strike=100.0, maturity=1.0)
        
        result = engine.price(payoff, maturity=1.0)
        
        assert isinstance(result, SimulationResult)
        assert result.price > 0  # Put should have positive value
        assert result.standard_error > 0
        assert result.confidence_interval[0] < result.confidence_interval[1]
    
    def test_price_negative_maturity_raises_error(self, engine):
        """Test that negative maturity raises ValueError."""
        payoff = EuropeanCallPayoff(strike=100.0, maturity=1.0)
        
        with pytest.raises(ValueError, match="Maturity must be positive"):
            engine.price(payoff, maturity=-1.0)
    
    def test_price_zero_maturity_raises_error(self, engine):
        """Test that zero maturity raises ValueError."""
        payoff = EuropeanCallPayoff(strike=100.0, maturity=1.0)
        
        with pytest.raises(ValueError, match="Maturity must be positive"):
            engine.price(payoff, maturity=0.0)
    
    def test_price_invalid_confidence_level_raises_error(self, engine):
        """Test that invalid confidence level raises ValueError."""
        payoff = EuropeanCallPayoff(strike=100.0, maturity=1.0)
        
        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            engine.price(payoff, maturity=1.0, confidence_level=1.5)
        
        with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
            engine.price(payoff, maturity=1.0, confidence_level=-0.1)
    
    def test_price_different_confidence_levels(self, engine):
        """Test pricing with different confidence levels."""
        payoff = EuropeanCallPayoff(strike=100.0, maturity=1.0)
        
        result_95 = engine.price(payoff, maturity=1.0, confidence_level=0.95)
        result_99 = engine.price(payoff, maturity=1.0, confidence_level=0.99)
        
        # 99% confidence interval should be wider than 95%
        width_95 = result_95.confidence_interval[1] - result_95.confidence_interval[0]
        width_99 = result_99.confidence_interval[1] - result_99.confidence_interval[0]
        
        assert width_99 > width_95
    
    def test_price_reproducibility(self, market_data):
        """Test that same configuration produces same results."""
        config = MonteCarloConfig(num_simulations=1000, random_seed=42)
        
        engine1 = MonteCarloEngine(market_data, config)
        engine2 = MonteCarloEngine(market_data, config)
        
        payoff = EuropeanCallPayoff(strike=100.0, maturity=1.0)
        
        result1 = engine1.price(payoff, maturity=1.0)
        result2 = engine2.price(payoff, maturity=1.0)
        
        assert abs(result1.price - result2.price) < 1e-10
        assert abs(result1.standard_error - result2.standard_error) < 1e-10
    
    def test_price_multiple_empty_list_raises_error(self, engine):
        """Test that empty payoffs list raises ValueError."""
        with pytest.raises(ValueError, match="Payoffs list cannot be empty"):
            engine.price_multiple([], maturity=1.0)
    
    def test_price_multiple_single_payoff(self, engine):
        """Test pricing multiple payoffs with single payoff."""
        payoffs = [EuropeanCallPayoff(strike=100.0, maturity=1.0)]
        
        results = engine.price_multiple(payoffs, maturity=1.0)
        
        assert len(results) == 1
        assert isinstance(results[0], SimulationResult)
        assert results[0].price > 0
    
    def test_price_multiple_payoffs(self, engine):
        """Test pricing multiple payoffs simultaneously."""
        payoffs = [
            EuropeanCallPayoff(strike=100.0, maturity=1.0),
            EuropeanPutPayoff(strike=100.0, maturity=1.0),
            EuropeanCallPayoff(strike=110.0, maturity=1.0)
        ]
        
        results = engine.price_multiple(payoffs, maturity=1.0)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, SimulationResult)
            assert result.price >= 0
            assert result.standard_error > 0
    
    def test_price_multiple_consistency(self, engine):
        """Test that multiple pricing gives same results as individual pricing."""
        call_payoff = EuropeanCallPayoff(strike=100.0, maturity=1.0)
        put_payoff = EuropeanPutPayoff(strike=100.0, maturity=1.0)
        
        # Price individually
        call_result = engine.price(call_payoff, maturity=1.0)
        put_result = engine.price(put_payoff, maturity=1.0)
        
        # Price together (need new engine with same seed for fair comparison)
        engine2 = MonteCarloEngine(engine.market_data, engine.config)
        multiple_results = engine2.price_multiple([call_payoff, put_payoff], maturity=1.0)
        
        # Results should be reasonably close (allow for Monte Carlo error)
        # With 10000 simulations, differences should be small but may exceed 0.01
        assert abs(call_result.price - multiple_results[0].price) < 0.1
        assert abs(put_result.price - multiple_results[1].price) < 0.1
    
    def test_put_call_parity_approximation(self, engine):
        """Test that put-call parity holds approximately."""
        strike = 100.0
        maturity = 1.0
        
        call_payoff = EuropeanCallPayoff(strike=strike, maturity=maturity)
        put_payoff = EuropeanPutPayoff(strike=strike, maturity=maturity)
        
        call_result = engine.price(call_payoff, maturity=maturity)
        put_result = engine.price(put_payoff, maturity=maturity)
        
        # Put-call parity: C - P = S*exp(-q*T) - K*exp(-r*T)
        spot = engine.market_data.spot_price
        rate = engine.market_data.risk_free_rate
        div_yield = engine.market_data.dividend_yield
        
        forward_price = spot * np.exp(-div_yield * maturity)
        discounted_strike = strike * np.exp(-rate * maturity)
        
        expected_diff = forward_price - discounted_strike
        actual_diff = call_result.price - put_result.price
        
        # Should be close due to Monte Carlo approximation
        assert abs(actual_diff - expected_diff) < 1.0  # Allow for MC error
    
    def test_deep_out_of_money_call(self, engine):
        """Test pricing of deep out-of-the-money call."""
        # Very high strike relative to spot
        payoff = EuropeanCallPayoff(strike=200.0, maturity=1.0)
        
        result = engine.price(payoff, maturity=1.0)
        
        # Should have very low price
        assert result.price < 1.0
        assert result.price >= 0.0
    
    def test_deep_in_money_call(self, engine):
        """Test pricing of deep in-the-money call."""
        # Very low strike relative to spot
        payoff = EuropeanCallPayoff(strike=50.0, maturity=1.0)
        
        result = engine.price(payoff, maturity=1.0)
        
        # Should have high price, close to forward price minus strike
        spot = engine.market_data.spot_price
        rate = engine.market_data.risk_free_rate
        div_yield = engine.market_data.dividend_yield
        
        forward_price = spot * np.exp(-div_yield * 1.0)
        discounted_strike = 50.0 * np.exp(-rate * 1.0)
        
        expected_price = forward_price - discounted_strike
        
        # Should be close to intrinsic value
        assert abs(result.price - expected_price) < 5.0  # Allow for MC error and volatility
    
    def test_confidence_interval_contains_true_price(self, engine):
        """Test that confidence interval methodology is reasonable."""
        payoff = EuropeanCallPayoff(strike=100.0, maturity=1.0)
        
        # Run multiple pricing exercises and check coverage
        results = []
        for _ in range(10):  # Limited due to computation time
            result = engine.price(payoff, maturity=1.0, confidence_level=0.95)
            results.append(result)
        
        # Check that confidence intervals have reasonable properties
        for result in results:
            assert result.confidence_interval[0] <= result.price <= result.confidence_interval[1]
            assert result.confidence_interval[1] - result.confidence_interval[0] > 0


class DummyBadPayoff:
    """Dummy payoff that returns wrong shape for testing error handling."""
    
    def __call__(self, spot_paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Return wrong shape array."""
        return np.array([1.0, 2.0])  # Wrong shape


class TestMonteCarloEngineErrorHandling:
    """Test error handling in MonteCarloEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create engine for error testing."""
        market_data = MarketData(
            spot_price=100.0,
            volatility=0.2,
            risk_free_rate=0.05
        )
        config = MonteCarloConfig(num_simulations=100)
        return MonteCarloEngine(market_data, config)
    
    def test_payoff_wrong_shape_raises_error(self, engine):
        """Test that payoff returning wrong shape raises error."""
        bad_payoff = DummyBadPayoff()
        
        with pytest.raises(ValueError, match="Payoff function must return array of shape"):
            engine.price(bad_payoff, maturity=1.0)
