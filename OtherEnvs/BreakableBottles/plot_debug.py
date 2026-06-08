import os
import pandas as pd
import matplotlib.pyplot as plt

THETA = 0.01
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bb_policies', 'debug.log')

df = pd.read_csv(LOG_FILE, sep='\t')
df = df[df['iteration'] >= 1100]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Delta vs Iteration
ax = axes[0]
ax.plot(df['iteration'], df['delta'], marker='o', markersize=4, linewidth=1.5, label='Delta')
ax.axhline(y=THETA, color='red', linestyle='--', linewidth=1.2, label=f'θ = {THETA}')
ax.set_xlabel('Iteration')
ax.set_ylabel('Delta')
ax.set_title('Delta vs Iteration')
ax.legend()
ax.grid(True, alpha=0.3)

# Avg Hull Size vs Iteration
ax = axes[1]
ax.plot(df['iteration'], df['avg_hull_size'], marker='o', markersize=4, linewidth=1.5, color='tab:orange')
ax.set_xlabel('Iteration')
ax.set_ylabel('Average Hull Size')
ax.set_title('Average Hull Size vs Iteration')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bb_policies', 'debug_plots_theta0.001.png')
plt.savefig(out_path, dpi=150)
plt.show()
print(f"Saved to {out_path}")
