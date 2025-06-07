"""
Example usage of the Monte Carlo engine for pricing derivatives.

This example demonstrates how to use the Monte Carlo engine to price
various types of derivatives using Black-Scholes models.
"""

import logging
import numpy as np
from monte_carlo import (
    MarketData, MonteCarloConfig, MonteCarloEngine,
    EuropeanCallPayoff, EuropeanPutPayoff, AsianCallPayoff, BarrierCallPayoff
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def create_custom_payoff() -> callable:
    """Create a custom payoff function for demonstration.
    
    Returns:
        Callable that implements a digital call payoff
    """
    def digital_call_payoff(spot_paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Digital call payoff: pays 1 if S_T > K, 0 otherwise."""
        strike = 105.0
        maturity_idx = len(times) - 1  # Use final time point
        final_spots = spot_paths[:, maturity_idx]
        return np.where(final_spots > strike, 1.0, 0.0)
    
    return digital_call_payoff


def demonstrate_basic_pricing():
    """Demonstrate basic option pricing functionality."""
    print("=== Basic Option Pricing ===")
    
    # Set up market data
    market_data = MarketData(
        spot_price=100.0,
        volatility=0.25,
        risk_free_rate=0.05,
        dividend_yield=0.02
    )
    
    # Configure Monte Carlo simulation
    config = MonteCarloConfig(
        num_simulations=500000,
        num_time_steps=252,
        random_seed=42
    )
    
    # Create Monte Carlo engine
    engine = MonteCarloEngine(market_data, config)
    
    # Price European call option
    call_payoff = EuropeanCallPayoff(strike=100.0, maturity=1.0)
    call_result = engine.price(call_payoff, maturity=1.0)
    
    print(f"European Call Option (K=100, T=1):")
    print(f"  Price: ${call_result.price:.4f}")
    print(f"  Standard Error: ${call_result.standard_error:.4f}")
    print(f"  95% Confidence Interval: [${call_result.confidence_interval[0]:.4f}, ${call_result.confidence_interval[1]:.4f}]")
    print(f"  Number of Simulations: {call_result.num_simulations:,}")
    
    # Price European put option
    put_payoff = EuropeanPutPayoff(strike=100.0, maturity=1.0)
    put_result = engine.price(put_payoff, maturity=1.0)
    
    print(f"\nEuropean Put Option (K=100, T=1):")
    print(f"  Price: ${put_result.price:.4f}")
    print(f"  Standard Error: ${put_result.standard_error:.4f}")
    print(f"  95% Confidence Interval: [${put_result.confidence_interval[0]:.4f}, ${put_result.confidence_interval[1]:.4f}]")
    
    # Verify put-call parity
    forward_price = market_data.spot_price * np.exp(-market_data.dividend_yield * 1.0)
    discounted_strike = 100.0 * np.exp(-market_data.risk_free_rate * 1.0)
    parity_diff = call_result.price - put_result.price
    expected_diff = forward_price - discounted_strike
    
    print(f"\nPut-Call Parity Check:")
    print(f"  Call - Put = ${parity_diff:.4f}")
    print(f"  Expected (S*e^(-qT) - K*e^(-rT)) = ${expected_diff:.4f}")
    print(f"  Difference: ${abs(parity_diff - expected_diff):.4f}")


def demonstrate_exotic_options():
    """Demonstrate pricing of exotic options."""
    print("\n=== Exotic Option Pricing ===")
    
    market_data = MarketData(
        spot_price=100.0,
        volatility=0.3,
        risk_free_rate=0.04,
        dividend_yield=0.01
    )
    
    config = MonteCarloConfig(
        num_simulations=200000,
        num_time_steps=100,
        random_seed=123
    )
    
    engine = MonteCarloEngine(market_data, config)
    
    # Asian call option
    asian_call = AsianCallPayoff(strike=100.0, maturity=1.0)
    asian_result = engine.price(asian_call, maturity=1.0)
    
    print(f"Asian Call Option (K=100, T=1):")
    print(f"  Price: ${asian_result.price:.4f}")
    print(f"  Standard Error: ${asian_result.standard_error:.4f}")
    
    # Barrier call option
    barrier_call = BarrierCallPayoff(strike=100.0, maturity=1.0, barrier=130.0)
    barrier_result = engine.price(barrier_call, maturity=1.0)
    
    print(f"\nBarrier Call Option (K=100, T=1, Barrier=130):")
    print(f"  Price: ${barrier_result.price:.4f}")
    print(f"  Standard Error: ${barrier_result.standard_error:.4f}")
    
    # Custom digital call option
    digital_payoff = create_custom_payoff()
    digital_result = engine.price(digital_payoff, maturity=1.0)
    
    print(f"\nDigital Call Option (K=105, T=1):")
    print(f"  Price: ${digital_result.price:.4f}")
    print(f"  Standard Error: ${digital_result.standard_error:.4f}")


def demonstrate_multiple_pricing():
    """Demonstrate pricing multiple derivatives simultaneously."""
    print("\n=== Multiple Derivative Pricing ===")
    
    market_data = MarketData(
        spot_price=100.0,
        volatility=0.2,
        risk_free_rate=0.05
    )
    
    config = MonteCarloConfig(
        num_simulations=100000,
        random_seed=456
    )
    
    engine = MonteCarloEngine(market_data, config)
    
    # Create multiple payoffs
    payoffs = [
        EuropeanCallPayoff(strike=95.0, maturity=1.0),
        EuropeanCallPayoff(strike=100.0, maturity=1.0),
        EuropeanCallPayoff(strike=105.0, maturity=1.0),
        EuropeanPutPayoff(strike=95.0, maturity=1.0),
        EuropeanPutPayoff(strike=100.0, maturity=1.0),
        EuropeanPutPayoff(strike=105.0, maturity=1.0),
    ]
    
    strikes = [95.0, 100.0, 105.0, 95.0, 100.0, 105.0]
    option_types = ['Call', 'Call', 'Call', 'Put', 'Put', 'Put']
    
    # Price all options using the same underlying paths
    results = engine.price_multiple(payoffs, maturity=1.0)
    
    print("Strike  Type  Price    Std Error")
    print("-" * 35)
    for i, (strike, opt_type, result) in enumerate(zip(strikes, option_types, results)):
        print(f"{strike:6.0f}  {opt_type:4s}  ${result.price:6.4f}  ${result.standard_error:.4f}")


def demonstrate_convergence_analysis():
    """Demonstrate convergence analysis with different simulation counts."""
    print("\n=== Convergence Analysis ===")
    
    market_data = MarketData(
        spot_price=100.0,
        volatility=0.2,
        risk_free_rate=0.05
    )
    
    call_payoff = EuropeanCallPayoff(strike=100.0, maturity=1.0)
    simulation_counts = [10000, 50000, 100000, 500000]
    
    print("Simulations  Price    Std Error  95% CI Width")
    print("-" * 45)
    
    for num_sims in simulation_counts:
        config = MonteCarloConfig(
            num_simulations=num_sims,
            random_seed=789
        )
        engine = MonteCarloEngine(market_data, config)
        result = engine.price(call_payoff, maturity=1.0)
        
        ci_width = result.confidence_interval[1] - result.confidence_interval[0]
        
        print(f"{num_sims:10,}  ${result.price:6.4f}  ${result.standard_error:.4f}     ${ci_width:.4f}")


def demonstrate_sensitivity_analysis():
    """Demonstrate sensitivity to market parameters."""
    print("\n=== Sensitivity Analysis ===")
    
    base_market_data = MarketData(
        spot_price=100.0,
        volatility=0.2,
        risk_free_rate=0.05
    )
    
    config = MonteCarloConfig(num_simulations=100000, random_seed=999)
    call_payoff = EuropeanCallPayoff(strike=100.0, maturity=1.0)
    
    # Base case
    base_engine = MonteCarloEngine(base_market_data, config)
    base_result = base_engine.price(call_payoff, maturity=1.0)
    print(f"Base Case: ${base_result.price:.4f}")
    
    # Volatility sensitivity
    vol_shocks = [-0.05, 0.05, 0.1]
    print(f"\nVolatility Sensitivity:")
    print(f"Vol Change  New Vol  Price    Change")
    print("-" * 35)
    
    for vol_shock in vol_shocks:
        new_vol = base_market_data.volatility + vol_shock
        shocked_data = MarketData(
            spot_price=base_market_data.spot_price,
            volatility=new_vol,
            risk_free_rate=base_market_data.risk_free_rate
        )
        shocked_engine = MonteCarloEngine(shocked_data, config)
        shocked_result = shocked_engine.price(call_payoff, maturity=1.0)
        
        price_change = shocked_result.price - base_result.price
        print(f"{vol_shock:+8.2f}    {new_vol:.2f}  ${shocked_result.price:6.4f}  ${price_change:+6.4f}")
    
    # Spot price sensitivity
    spot_shocks = [-5.0, 5.0, 10.0]
    print(f"\nSpot Price Sensitivity:")
    print(f"Spot Change  New Spot  Price    Change")
    print("-" * 37)
    
    for spot_shock in spot_shocks:
        new_spot = base_market_data.spot_price + spot_shock
        shocked_data = MarketData(
            spot_price=new_spot,
            volatility=base_market_data.volatility,
            risk_free_rate=base_market_data.risk_free_rate
        )
        shocked_engine = MonteCarloEngine(shocked_data, config)
        shocked_result = shocked_engine.price(call_payoff, maturity=1.0)
        
        price_change = shocked_result.price - base_result.price
        print(f"{spot_shock:+10.1f}    {new_spot:6.1f}  ${shocked_result.price:6.4f}  ${price_change:+6.4f}")


if __name__ == "__main__":
    print("Monte Carlo Derivatives Pricing Engine - Comprehensive Example")
    print("=" * 70)
    
    try:
        demonstrate_basic_pricing()
        demonstrate_exotic_options()
        demonstrate_multiple_pricing()
        demonstrate_convergence_analysis()
        demonstrate_sensitivity_analysis()
        
        print(f"\n{'=' * 70}")
        print("All demonstrations completed successfully!")
        print("The Monte Carlo engine is ready for production use.")
        
    except Exception as e:
        logging.error(f"Error during demonstration: {e}")
        raise
