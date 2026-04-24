"""Regenerate the key README figures from the committed analysis output.

Run: python generate_figures.py
Reads: customer_segments_analysis.csv
Writes: figures/*.png (overwrites)

No raw dataset needed — everything is derived from the labeled customer CSV
plus the elbow/silhouette scores extracted from the notebook.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 160,
    'font.family': 'DejaVu Sans',
    'axes.titleweight': 'bold',
})

PALETTE = {'Champions': '#2E8B57', 'Promising': '#4682B4', 'At Risk': '#C44536'}
SEG_ORDER = ['Champions', 'Promising', 'At Risk']

FIG_DIR = Path('figures')
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv('customer_segments_analysis.csv')
colors = [PALETTE[s] for s in SEG_ORDER]

# ---------- Figure 1: Elbow + silhouette (justifies k=3) ----------
k_vals = list(range(2, 11))
sse = [10713.81, 7682.47, 6295.84, 5689.79, 5094.77, 4606.19, 4266.32, 3967.49, 3735.43]
sil = [0.335, 0.351, 0.298, 0.286, 0.252, 0.255, 0.258, 0.250, 0.244]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
ax1.plot(k_vals, sse, 'o-', color='#1f77b4', linewidth=2, markersize=8)
ax1.axvline(3, color='#C44536', linestyle='--', alpha=0.7, label='Chosen k=3')
ax1.set_xlabel('Number of clusters (k)'); ax1.set_ylabel('SSE (inertia)')
ax1.set_title('Elbow method')
ax1.legend()

ax2.plot(k_vals, sil, 'o-', color='#2ca02c', linewidth=2, markersize=8)
ax2.axvline(3, color='#C44536', linestyle='--', alpha=0.7, label='Chosen k=3')
ax2.scatter([3], [0.351], s=200, color='#C44536', zorder=5, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Number of clusters (k)'); ax2.set_ylabel('Silhouette score')
ax2.set_title('Silhouette score (higher = better separation)')
ax2.legend()

plt.suptitle('Choosing k: SSE elbow + silhouette peak at k=3', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'elbow-silhouette.png', bbox_inches='tight')
plt.close()

# ---------- Figure 2: Segment sizes + revenue share ----------
sizes = df['Segment_Name'].value_counts().reindex(SEG_ORDER)
pct_sizes = sizes / sizes.sum() * 100
revenue = df.groupby('Segment_Name')['MonetaryValue'].sum().reindex(SEG_ORDER)
pct_rev = revenue / revenue.sum() * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
bars = ax1.bar(SEG_ORDER, sizes, color=colors, edgecolor='white', linewidth=2)
for b, n, p in zip(bars, sizes, pct_sizes):
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height(),
             f'{n:,}\n({p:.1f}%)', ha='center', va='bottom', fontsize=11)
ax1.set_ylabel('Customers')
ax1.set_title('Customer count by segment')
ax1.set_ylim(0, sizes.max() * 1.2)

bars = ax2.bar(SEG_ORDER, revenue / 1000, color=colors, edgecolor='white', linewidth=2)
for b, r, p in zip(bars, revenue, pct_rev):
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height(),
             f'${r/1000:,.0f}K\n({p:.1f}%)', ha='center', va='bottom', fontsize=11)
ax2.set_ylabel('Revenue ($K)')
ax2.set_title('Total revenue by segment')
ax2.set_ylim(0, revenue.max() / 1000 * 1.2)

plt.tight_layout()
plt.savefig(FIG_DIR / 'segment-overview.png', bbox_inches='tight')
plt.close()

# ---------- Figure 3: RFMT fingerprint (mean per segment) ----------
metrics = ['Recency', 'Frequency', 'MonetaryValue', 'Tenure']
units = [' (days)', '', ' ($)', ' (days)']
profile = df.groupby('Segment_Name')[metrics].mean().reindex(SEG_ORDER)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, (m, u) in enumerate(zip(metrics, units)):
    vals = profile[m]
    axes[i].bar(SEG_ORDER, vals, color=colors, edgecolor='white', linewidth=2)
    for j, v in enumerate(vals):
        axes[i].text(j, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=10)
    axes[i].set_title(m + u)
    axes[i].tick_params(axis='x', rotation=15)
    axes[i].set_ylim(0, vals.max() * 1.18)

plt.suptitle('RFMT fingerprint — mean per segment', fontsize=13, y=1.03)
plt.tight_layout()
plt.savefig(FIG_DIR / 'rfmt-fingerprint.png', bbox_inches='tight')
plt.close()

# ---------- Figure 4: 3D scatter of segments in RFM space ----------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
for seg in SEG_ORDER:
    sub = df[df['Segment_Name'] == seg]
    ax.scatter(sub['Recency'], sub['Frequency'], sub['MonetaryValue'],
               c=PALETTE[seg], label=seg, alpha=0.55, s=18,
               edgecolor='white', linewidth=0.3)
ax.set_xlabel('Recency (days)')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary ($)')
ax.set_title('Customer segments in RFM space', fontsize=13)
ax.legend(loc='upper left', fontsize=10)
ax.view_init(elev=22, azim=-55)
plt.tight_layout()
plt.savefig(FIG_DIR / 'segments-3d.png', bbox_inches='tight')
plt.close()

# ---------- Figure 5: Churn risk stacked bar ----------
RISK_ORDER = ['Low', 'Medium', 'High', 'Critical']
RISK_COLORS = ['#52B788', '#F4A261', '#E76F51', '#7F1D1D']
ct = pd.crosstab(df['Segment_Name'], df['Churn_Risk']).reindex(SEG_ORDER)[RISK_ORDER]

fig, ax = plt.subplots(figsize=(11, 4.2))
ct.plot(kind='barh', stacked=True, ax=ax, color=RISK_COLORS,
        edgecolor='white', linewidth=1)
ax.set_xlabel('Customers')
ax.set_ylabel('')
ax.set_title('Churn risk distribution by segment')
ax.legend(title='Churn risk', loc='center left', bbox_to_anchor=(1.01, 0.5))
plt.tight_layout()
plt.savefig(FIG_DIR / 'churn-risk.png', bbox_inches='tight')
plt.close()

print('Generated figures:')
for f in sorted(FIG_DIR.glob('*.png')):
    print(f'  {f.name}  ({f.stat().st_size/1024:.0f} KB)')
