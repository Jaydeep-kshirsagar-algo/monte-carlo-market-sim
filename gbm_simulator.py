import numpy as np
import matplotlib.pyplot as plt

# 1. Simulation Parameters
S0 = 100.0          # Initial asset price ($)
mu = 0.08           # Expected annual drift (8%)
sigma = 0.20        # Annual volatility (20%)
T = 1.0             # Time horizon in years (1 year)
steps = 252         # Trading days in a year
num_paths = 10000   # Number of Monte Carlo simulations

dt = T / steps      # Size of each time step

# 2. Generate Random Normal Shocks (steps x num_paths)
np.random.seed(42)  # Reproducible results
Z = np.random.standard_normal((steps, num_paths))

# 3. Compute Daily Log-Returns & Price Paths
daily_returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
price_paths = np.zeros((steps + 1, num_paths))
price_paths[0] = S0

for t in range(1, steps + 1):
    price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]

# 4. Print Key Quant Summary Statistics
final_prices = price_paths[-1]
print(f"Simulated {num_paths:,} paths over {steps} trading days:")
print(f"Mean Final Price:   ${np.mean(final_prices):.2f}")
print(f"Median Final Price: ${np.median(final_prices):.2f}")
print(f"5th Percentile:     ${np.percentile(final_prices, 5):.2f}")
print(f"95th Percentile:    ${np.percentile(final_prices, 95):.2f}")

# 5. Plot the First 50 Sample Paths & Save Image
plt.figure(figsize=(10, 5))
plt.plot(price_paths[:, :50], lw=1, alpha=0.7)
plt.title("Monte Carlo Simulation: 50 Sample Asset Price Paths (GBM)")
plt.xlabel("Trading Days")
plt.ylabel("Asset Price ($)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gbm_paths.png", dpi=150)
plt.show()