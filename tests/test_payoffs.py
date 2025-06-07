import pytest
import numpy as np
from monte_carlo.payoffs import (
    EuropeanCallPayoff, EuropeanPutPayoff, AsianCallPayoff, BarrierCallPayoff
)


class TestEuropeanCallPayoff:
    """Test EuropeanCallPayoff implementation."""
    
    def test_initialization(self):
        """Test European call payoff initialization."""
        payoff = EuropeanCallPayoff(strike=100.0, maturity=1.0)
        
        assert payoff.strike == 100.0
        assert payoff.maturity == 1.0
    
    def test_negative_strike_raises_error(self):
        """Test that negative strike raises ValueError."""
        with pytest.raises(ValueError, match="Strike must be positive"):
            EuropeanCallPayoff(strike=-100.0, maturity=1.0)
    
    def test_zero_strike_raises_error(self):
        """Test that zero strike raises ValueError."""
        with pytest.raises(ValueError, match="Strike must be positive"):
            EuropeanCallPayoff(strike=0.0, maturity=1.0)
    
    def test_negative_maturity_raises_error(self):
        """Test that negative maturity raises ValueError."""
        with pytest.raises(ValueError, match="Maturity must be positive"):
            EuropeanCallPayoff(strike=100.0, maturity=-1.0)
    
    def test_zero_maturity_raises_error(self):
        """Test that zero maturity raises ValueError."""
        with pytest.raises(ValueError, match="Maturity must be positive"):
            EuropeanCallPayoff(strike=100.0, maturity=0.0)
    
    def test_in_the_money_payoff(self):
        """Test call payoff when option is in the money."""
        payoff = EuropeanCallPayoff(strike=100.0, maturity=1.0)
        
        spot_paths = np.array([[90.0, 95.0, 110.0], [100.0, 105.0, 120.0]])
        times = np.array([0.0, 0.5, 1.0])
        
        result = payoff.calculate_payoff(spot_paths, times)
        expected = np.array([10.0, 20.0])  # max(110-100, 0), max(120-100, 0)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_out_of_money_payoff(self):
        """Test call payoff when option is out of the money."""
        payoff = EuropeanCallPayoff(strike=100.0, maturity=1.0)
        
        spot_paths = np.array([[90.0, 85.0, 80.0], [95.0, 90.0, 85.0]])
        times = np.array([0.0, 0.5, 1.0])
        
        result = payoff.calculate_payoff(spot_paths, times)
        expected = np.array([0.0, 0.0])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_at_the_money_payoff(self):
        """Test call payoff when option is at the money."""
        payoff = EuropeanCallPayoff(strike=100.0, maturity=1.0)
        
        spot_paths = np.array([[90.0, 95.0, 100.0]])
        times = np.array([0.0, 0.5, 1.0])
        
        result = payoff.calculate_payoff(spot_paths, times)
        expected = np.array([0.0])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_maturity_matching(self):
        """Test that payoff uses correct maturity index."""
        payoff = EuropeanCallPayoff(strike=100.0, maturity=0.5)
        
        spot_paths = np.array([[90.0, 110.0, 80.0]])  # Middle value should be used
        times = np.array([0.0, 0.5, 1.0])
        
        result = payoff.calculate_payoff(spot_paths, times)
        expected = np.array([10.0])  # max(110-100, 0)
        
        np.testing.assert_array_equal(result, expected)


class TestEuropeanPutPayoff:
    """Test EuropeanPutPayoff implementation."""
    
    def test_initialization(self):
        """Test European put payoff initialization."""
        payoff = EuropeanPutPayoff(strike=100.0, maturity=1.0)
        
        assert payoff.strike == 100.0
        assert payoff.maturity == 1.0
    
    def test_negative_strike_raises_error(self):
        """Test that negative strike raises ValueError."""
        with pytest.raises(ValueError, match="Strike must be positive"):
            EuropeanPutPayoff(strike=-100.0, maturity=1.0)
    
    def test_negative_maturity_raises_error(self):
        """Test that negative maturity raises ValueError."""
        with pytest.raises(ValueError, match="Maturity must be positive"):
            EuropeanPutPayoff(strike=100.0, maturity=-1.0)
    
    def test_in_the_money_payoff(self):
        """Test put payoff when option is in the money."""
        payoff = EuropeanPutPayoff(strike=100.0, maturity=1.0)
        
        spot_paths = np.array([[110.0, 95.0, 80.0], [100.0, 95.0, 70.0]])
        times = np.array([0.0, 0.5, 1.0])
        
        result = payoff.calculate_payoff(spot_paths, times)
        expected = np.array([20.0, 30.0])  # max(100-80, 0), max(100-70, 0)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_out_of_money_payoff(self):
        """Test put payoff when option is out of the money."""
        payoff = EuropeanPutPayoff(strike=100.0, maturity=1.0)
        
        spot_paths = np.array([[90.0, 105.0, 120.0], [95.0, 110.0, 115.0]])
        times = np.array([0.0, 0.5, 1.0])
        
        result = payoff.calculate_payoff(spot_paths, times)
        expected = np.array([0.0, 0.0])
        
        np.testing.assert_array_equal(result, expected)


class TestAsianCallPayoff:
    """Test AsianCallPayoff implementation."""
    
    def test_initialization(self):
        """Test Asian call payoff initialization."""
        payoff = AsianCallPayoff(strike=100.0, maturity=1.0)
        
        assert payoff.strike == 100.0
        assert payoff.maturity == 1.0
    
    def test_negative_strike_raises_error(self):
        """Test that negative strike raises ValueError."""
        with pytest.raises(ValueError, match="Strike must be positive"):
            AsianCallPayoff(strike=-100.0, maturity=1.0)
    
    def test_negative_maturity_raises_error(self):
        """Test that negative maturity raises ValueError."""
        with pytest.raises(ValueError, match="Maturity must be positive"):
            AsianCallPayoff(strike=100.0, maturity=-1.0)
    
    def test_asian_call_payoff(self):
        """Test Asian call payoff calculation."""
        payoff = AsianCallPayoff(strike=100.0, maturity=1.0)
        
        # Path 1: average = (90+100+110)/3 = 100, payoff = max(100-100, 0) = 0
        # Path 2: average = (100+110+120)/3 = 110, payoff = max(110-100, 0) = 10
        spot_paths = np.array([[90.0, 100.0, 110.0], [100.0, 110.0, 120.0]])
        times = np.array([0.0, 0.5, 1.0])
        
        result = payoff.calculate_payoff(spot_paths, times)
        expected = np.array([0.0, 10.0])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_asian_call_partial_maturity(self):
        """Test Asian call with partial maturity."""
        payoff = AsianCallPayoff(strike=100.0, maturity=0.5)
        
        # Only first two points should be used for averaging
        # Path: average = (90+110)/2 = 100, payoff = max(100-100, 0) = 0
        spot_paths = np.array([[90.0, 110.0, 80.0]])
        times = np.array([0.0, 0.5, 1.0])
        
        result = payoff.calculate_payoff(spot_paths, times)
        expected = np.array([0.0])
        
        np.testing.assert_array_equal(result, expected)


class TestBarrierCallPayoff:
    """Test BarrierCallPayoff implementation."""
    
    def test_initialization(self):
        """Test barrier call payoff initialization."""
        payoff = BarrierCallPayoff(strike=100.0, maturity=1.0, barrier=120.0)
        
        assert payoff.strike == 100.0
        assert payoff.maturity == 1.0
        assert payoff.barrier == 120.0
    
    def test_negative_strike_raises_error(self):
        """Test that negative strike raises ValueError."""
        with pytest.raises(ValueError, match="Strike must be positive"):
            BarrierCallPayoff(strike=-100.0, maturity=1.0, barrier=120.0)
    
    def test_negative_maturity_raises_error(self):
        """Test that negative maturity raises ValueError."""
        with pytest.raises(ValueError, match="Maturity must be positive"):
            BarrierCallPayoff(strike=100.0, maturity=-1.0, barrier=120.0)
    
    def test_negative_barrier_raises_error(self):
        """Test that negative barrier raises ValueError."""
        with pytest.raises(ValueError, match="Barrier must be positive"):
            BarrierCallPayoff(strike=100.0, maturity=1.0, barrier=-120.0)
    
    def test_barrier_not_hit_payoff(self):
        """Test barrier call when barrier is not hit."""
        payoff = BarrierCallPayoff(strike=100.0, maturity=1.0, barrier=130.0)
        
        # Path stays below barrier, final spot = 110, payoff = max(110-100, 0) = 10
        spot_paths = np.array([[90.0, 105.0, 110.0]])
        times = np.array([0.0, 0.5, 1.0])
        
        result = payoff.calculate_payoff(spot_paths, times)
        expected = np.array([10.0])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_barrier_hit_payoff(self):
        """Test barrier call when barrier is hit."""
        payoff = BarrierCallPayoff(strike=100.0, maturity=1.0, barrier=115.0)
        
        # Path hits barrier at middle point, payoff = 0 (knocked out)
        spot_paths = np.array([[90.0, 120.0, 110.0]])
        times = np.array([0.0, 0.5, 1.0])
        
        result = payoff.calculate_payoff(spot_paths, times)
        expected = np.array([0.0])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_barrier_exactly_touched(self):
        """Test barrier call when barrier is exactly touched."""
        payoff = BarrierCallPayoff(strike=100.0, maturity=1.0, barrier=115.0)
        
        # Path exactly touches barrier, should be knocked out
        spot_paths = np.array([[90.0, 115.0, 110.0]])
        times = np.array([0.0, 0.5, 1.0])
        
        result = payoff.calculate_payoff(spot_paths, times)
        expected = np.array([0.0])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_mixed_barrier_scenarios(self):
        """Test multiple paths with different barrier scenarios."""
        payoff = BarrierCallPayoff(strike=100.0, maturity=1.0, barrier=115.0)
        
        # Path 1: hits barrier (120 > 115), knocked out
        # Path 2: doesn't hit barrier (max = 110 < 115), payoff = max(110-100, 0) = 10
        spot_paths = np.array([[90.0, 120.0, 105.0], [95.0, 110.0, 110.0]])
        times = np.array([0.0, 0.5, 1.0])
        
        result = payoff.calculate_payoff(spot_paths, times)
        expected = np.array([0.0, 10.0])
        
        np.testing.assert_array_equal(result, expected)
