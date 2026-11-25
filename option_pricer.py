import math
import numpy as np
import matplotlib.pyplot as plt

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black_scholes_call(S0, K, T, r, sigma):
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def monte_carlo_call(S0, K, T, r, sigma, num_paths, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(num_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0.0)
    price = math.exp(-r * T) * np.mean(payoffs)
    std_err = math.exp(-r * T) * np.std(payoffs, ddof=1) / math.sqrt(num_paths)
    return price, std_err

S0 = 100.0      # Spot price ($)
K = 105.0       # Strike price ($)
T = 1.0         # Time to maturity (1 year)
r = 0.05        # Risk-free rate (5%)
sigma = 0.20    # Volatility (20%)

bs_price = black_scholes_call(S0, K, T, r, sigma)
print(f"Exact Black-Scholes Call Price: ${bs_price:.4f}\n")

path_counts = [100, 500, 1000, 5000, 10000, 50000, 100000]
mc_prices = []
errors = []

print(f"{'Paths (N)':>10} | {'MC Price ($)':>12} | {'Abs Error ($)':>13} | {'Std Error':>10}")
print("-" * 55)

for n in path_counts:
    mc_p, se = monte_carlo_call(S0, K, T, r, sigma, n)
    err = abs(mc_p - bs_price)
    mc_prices.append(mc_p)
    errors.append(se)
    print(f"{n:>10,} | {mc_p:>12.4f} | {err:>13.4f} | {se:>10.4f}")

plt.figure(figsize=(10, 5))
plt.errorbar(path_counts, mc_prices, yerr=1.96 * np.array(errors), fmt='o-', capsize=4, label="Monte Carlo Estimate (±95% CI)")
plt.axhline(bs_price, color='red', linestyle='--', linewidth=1.5, label=f"Exact Black-Scholes (${bs_price:.4f})")
plt.xscale("log")
plt.title("Monte Carlo European Call Option Convergence vs. Black-Scholes")
plt.xlabel("Number of Simulated Paths (Log Scale)")
plt.ylabel("Option Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mc_convergence.png", dpi=150)