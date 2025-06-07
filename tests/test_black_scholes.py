import pytest
import numpy as np
from pydantic import ValidationError
from monte_carlo.black_scholes import BlackScholesPathGenerator
from monte_carlo.models import MarketData, MonteCarloConfig


class TestBlackScholesPathGenerator:
    """Test BlackScholesPathGenerator implementation."""
    
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
            num_simulations=1000,
            num_time_steps=50,
            random_seed=42
        )
    
    def test_initialization(self, market_data, config):
        """Test path generator initialization."""
        generator = BlackScholesPathGenerator(market_data, config)
        
        assert generator.market_data == market_data
        assert generator.config == config
    
    def test_generate_paths_shape(self, market_data, config):
        """Test that generated paths have correct shape."""
        generator = BlackScholesPathGenerator(market_data, config)
        
        maturity = 1.0
        spot_paths, time_grid = generator.generate_paths(maturity)
        
        assert spot_paths.shape == (config.num_simulations, config.num_time_steps)
        assert time_grid.shape == (config.num_time_steps,)
    
    def test_generate_paths_time_grid(self, market_data, config):
        """Test that time grid is correctly structured."""
        generator = BlackScholesPathGenerator(market_data, config)
        
        maturity = 1.0
        _, time_grid = generator.generate_paths(maturity)
        
        assert time_grid[0] == 0.0
        assert time_grid[-1] == maturity
        assert np.all(np.diff(time_grid) >= 0)  # Non-decreasing
    
    def test_generate_paths_initial_spot(self, market_data, config):
        """Test that all paths start at initial spot price."""
        generator = BlackScholesPathGenerator(market_data, config)
        
        maturity = 1.0
        spot_paths, _ = generator.generate_paths(maturity)
        
        initial_spots = spot_paths[:, 0]
        np.testing.assert_array_almost_equal(
            initial_spots, 
            market_data.spot_price, 
            decimal=10
        )
    
    def test_generate_paths_positive_prices(self, market_data, config):
        """Test that all generated prices are positive."""
        generator = BlackScholesPathGenerator(market_data, config)
        
        maturity = 1.0
        spot_paths, _ = generator.generate_paths(maturity)
        
        assert np.all(spot_paths > 0)
    
    def test_generate_paths_reproducibility(self, market_data):
        """Test that same seed produces same paths."""
        config1 = MonteCarloConfig(num_simulations=100, random_seed=42)
        config2 = MonteCarloConfig(num_simulations=100, random_seed=42)
        
        generator1 = BlackScholesPathGenerator(market_data, config1)
        generator2 = BlackScholesPathGenerator(market_data, config2)
        
        maturity = 1.0
        paths1, _ = generator1.generate_paths(maturity)
        paths2, _ = generator2.generate_paths(maturity)
        
        np.testing.assert_array_equal(paths1, paths2)
    
    def test_generate_paths_different_seeds(self, market_data):
        """Test that different seeds produce different paths."""
        config1 = MonteCarloConfig(num_simulations=100, random_seed=42)
        config2 = MonteCarloConfig(num_simulations=100, random_seed=43)
        
        generator1 = BlackScholesPathGenerator(market_data, config1)
        generator2 = BlackScholesPathGenerator(market_data, config2)
        
        maturity = 1.0
        paths1, _ = generator1.generate_paths(maturity)
        paths2, _ = generator2.generate_paths(maturity)
        
        # Paths should be different (very high probability)
        assert not np.array_equal(paths1, paths2)
    
    def test_generate_paths_zero_maturity_raises_error(self, market_data, config):
        """Test that zero maturity raises ValueError."""
        generator = BlackScholesPathGenerator(market_data, config)
        
        with pytest.raises(ValueError, match="Maturity must be positive"):
            generator.generate_paths(0.0)
    
    def test_generate_paths_negative_maturity_raises_error(self, market_data, config):
        """Test that negative maturity raises ValueError."""
        generator = BlackScholesPathGenerator(market_data, config)
        
        with pytest.raises(ValueError, match="Maturity must be positive"):
            generator.generate_paths(-1.0)
    
    def test_generate_paths_different_maturities(self, market_data, config):
        """Test path generation with different maturities."""
        generator = BlackScholesPathGenerator(market_data, config)
        
        maturity1 = 0.5
        maturity2 = 2.0
        
        _, time_grid1 = generator.generate_paths(maturity1)
        _, time_grid2 = generator.generate_paths(maturity2)
        
        assert time_grid1[-1] == maturity1
        assert time_grid2[-1] == maturity2
    
    def test_zero_volatility_paths(self, config):
        """Test path generation with zero volatility."""
        # This should fail validation since volatility must be positive
        with pytest.raises(ValidationError, match="Value must be positive"):
            MarketData(
                spot_price=100.0,
                volatility=0.0,
                risk_free_rate=0.05,
                dividend_yield=0.02
            )
    
    def test_high_volatility_paths(self, config):
        """Test path generation with high volatility."""
        market_data = MarketData(
            spot_price=100.0,
            volatility=1.0,  # 100% volatility
            risk_free_rate=0.05,
            dividend_yield=0.02
        )
        
        generator = BlackScholesPathGenerator(market_data, config)
        maturity = 1.0
        spot_paths, _ = generator.generate_paths(maturity)
        
        # High volatility should produce wider spread of final prices
        final_prices = spot_paths[:, -1]
        price_std = np.std(final_prices)
        
        # With 100% vol and 1000 sims, standard deviation should be substantial
        assert price_std > 50.0
    
    def test_drift_effect(self, config):
        """Test that drift affects path evolution correctly."""
        # High positive risk-free rate should tend to push paths up
        market_data_high_rate = MarketData(
            spot_price=100.0,
            volatility=0.1,  # Low vol to see drift effect
            risk_free_rate=0.20,  # High rate
            dividend_yield=0.0
        )
        
        # High dividend yield should tend to push paths down
        market_data_high_div = MarketData(
            spot_price=100.0,
            volatility=0.1,  # Low vol to see drift effect
            risk_free_rate=0.05,
            dividend_yield=0.15  # High dividend
        )
        
        generator_high_rate = BlackScholesPathGenerator(market_data_high_rate, config)
        generator_high_div = BlackScholesPathGenerator(market_data_high_div, config)
        
        maturity = 1.0
        paths_high_rate, _ = generator_high_rate.generate_paths(maturity)
        paths_high_div, _ = generator_high_div.generate_paths(maturity)
        
        mean_final_high_rate = np.mean(paths_high_rate[:, -1])
        mean_final_high_div = np.mean(paths_high_div[:, -1])
        
        # High risk-free rate should generally produce higher final prices
        assert mean_final_high_rate > mean_final_high_div
