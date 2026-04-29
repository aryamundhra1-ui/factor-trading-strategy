import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════
# STEP 1 — DEFINE YOUR STOCK UNIVERSE
# 100 large-cap S&P 500 stocks across all sectors
# ════════════════════════════════════════════════════════════

UNIVERSE = [
    # Technology
    'AAPL','MSFT','NVDA','GOOGL','META','AVGO','ORCL','CRM','AMD','INTC',
    # Financials
    'JPM','BAC','WFC','GS','MS','BLK','SCHW','AXP','COF','USB',
    # Healthcare
    'JNJ','UNH','LLY','PFE','ABBV','MRK','TMO','ABT','DHR','BMY',
    # Consumer
    'AMZN','TSLA','HD','MCD','NKE','SBUX','TGT','COST','WMT','LOW',
    # Industrials
    'CAT','DE','BA','HON','UPS','RTX','LMT','GE','MMM','FDX',
    # Energy
    'XOM','CVX','COP','EOG','SLB','PSX','VLO','MPC','OXY','HAL',
    # Communication
    'NFLX','DIS','CMCSA','VZ','T','TMUS','CHTR','EA','TTWO','OMC',
    # Real Estate & Utilities
    'NEE','DUK','SO','D','AEP','PLD','AMT','CCI','EQIX','SPG',
    # Materials & Consumer Staples
    'PG','KO','PEP','PM','MO','CL','MDLZ','GIS','K','SYY',
    # Additional large caps
    'V','MA','PYPL','BRK-B','CVS','CI','HUM','MCK','ELV','DVA'
]

print(f"Universe: {len(UNIVERSE)} stocks")
print("Downloading price data... (this takes 3-5 minutes)")

# ════════════════════════════════════════════════════════════
# STEP 2 — DOWNLOAD PRICE DATA (2 years)
# ════════════════════════════════════════════════════════════

raw = yf.download(UNIVERSE, period='2y', auto_adjust=True)
prices = raw['Close'].dropna(axis=1, thresh=int(0.9 * len(raw)))
valid_tickers = list(prices.columns)
print(f"Successfully downloaded: {len(valid_tickers)} stocks")

# Daily log returns
returns = np.log(prices / prices.shift(1)).dropna()

# ════════════════════════════════════════════════════════════
# STEP 3 — CALCULATE THE 4 FACTOR SCORES
# ════════════════════════════════════════════════════════════

print("\nCalculating factor scores...")

# ── FACTOR 1: MOMENTUM ──────────────────────────────────────
# 12-month return, skipping the most recent month
# "Skip-1-month" removes short-term reversal noise
# This is the standard momentum definition used by AQR
t     = len(prices) - 1          # today
t_1m  = max(0, t - 21)           # 1 month ago  (skip last month)
t_12m = max(0, t - 252)          # 12 months ago

momentum_score = (
    (prices.iloc[t_1m] - prices.iloc[t_12m]) / prices.iloc[t_12m]
)

print(f"  ✓ Momentum calculated for {len(momentum_score)} stocks")

# ── FACTOR 2: LOW VOLATILITY ────────────────────────────────
# Annualized volatility over past 12 months
# We INVERT it — lower vol = higher score
annual_vol  = returns.std() * np.sqrt(252)
lowvol_score = -annual_vol      # negative because low vol is good

print(f"  ✓ Low Volatility calculated for {len(lowvol_score)} stocks")

# ── FACTOR 3 & 4: VALUE & QUALITY (from fundamentals) ───────
# Pull P/E ratio and Return on Equity from yfinance
print("  Downloading fundamental data...")

pe_ratios = {}
roe_values = {}

for i, ticker in enumerate(valid_tickers):
    try:
        info = yf.Ticker(ticker).info
        pe  = info.get('trailingPE', None)
        roe = info.get('returnOnEquity', None)

        # Value = earnings yield = 1/PE (higher = cheaper = better)
        if pe and pe > 0 and pe < 200:
            pe_ratios[ticker] = 1 / pe
        else:
            pe_ratios[ticker] = np.nan

        # Quality = ROE (higher = more profitable = better)
        if roe and not np.isnan(roe):
            roe_values[ticker] = roe
        else:
            roe_values[ticker] = np.nan

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(valid_tickers)} stocks processed...")

    except Exception:
        pe_ratios[ticker]  = np.nan
        roe_values[ticker] = np.nan

value_score   = pd.Series(pe_ratios)
quality_score = pd.Series(roe_values)

print(f"  ✓ Value calculated ({value_score.notna().sum()} stocks)")
print(f"  ✓ Quality calculated ({quality_score.notna().sum()} stocks)")

# ════════════════════════════════════════════════════════════
# STEP 4 — NORMALIZE TO Z-SCORES
# Z-score = (value - mean) / std
# Makes all 4 factors comparable on the same scale
# A Z-score of +2.0 means 2 standard deviations above average
# ════════════════════════════════════════════════════════════

def zscore(series):
    """Normalize a series to Z-scores, ignoring NaN values."""
    return (series - series.mean()) / series.std()

# Align all factors to the same stocks
common = (
    momentum_score.index
    .intersection(lowvol_score.index)
    .intersection(value_score.dropna().index)
    .intersection(quality_score.dropna().index)
)

mom_z  = zscore(momentum_score[common])
vol_z  = zscore(lowvol_score[common])
val_z  = zscore(value_score[common])
qua_z  = zscore(quality_score[common])

print(f"\nStocks with all 4 factors: {len(common)}")

# ════════════════════════════════════════════════════════════
# STEP 5 — BUILD COMPOSITE SCORE & RANK
# Equal weight all 4 factors (25% each)
# This is how AQR's core equity strategies work
# ════════════════════════════════════════════════════════════

composite = (mom_z + vol_z + val_z + qua_z) / 4

# Build the full factor table
factor_table = pd.DataFrame({
    'Momentum':    mom_z,
    'Value':       val_z,
    'Quality':     qua_z,
    'Low-Vol':     vol_z,
    'Composite':   composite
}).sort_values('Composite', ascending=False)

# ── TOP 20 STOCKS ───────────────────────────────────────────
top20 = factor_table.head(20)
bottom20 = factor_table.tail(20)

# ── PRINT FINAL REPORT ──────────────────────────────────────
print("\n" + "═"*72)
print("  MULTI-FACTOR STOCK SCREENER — MONTHLY REPORT")
print("═"*72)
print(f"\n{'Rank':<6}{'Ticker':<8}{'Momentum':>10}{'Value':>10}"
      f"{'Quality':>10}{'Low-Vol':>10}{'COMPOSITE':>12}")
print("─"*72)

for rank, (ticker, row) in enumerate(top20.iterrows(), 1):
    print(f"{rank:<6}{ticker:<8}"
          f"{row['Momentum']:>+10.2f}"
          f"{row['Value']:>+10.2f}"
          f"{row['Quality']:>+10.2f}"
          f"{row['Low-Vol']:>+10.2f}"
          f"{row['Composite']:>+12.2f}")

print("\n" + "═"*72)
print(f"  → These are your TOP 20 stocks for this month's portfolio")
print("═"*72)

# Print selected tickers cleanly
print(f"\nSelected tickers:")
print("  " + "  ".join(top20.index.tolist()))

# ════════════════════════════════════════════════════════════
# STEP 6 — VISUALIZE FACTOR SCORES
# ════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.patch.set_facecolor('#0F1117')
fig.suptitle('Multi-Factor Stock Screener — S&P 500 Universe',
             fontsize=16, fontweight='bold', color='white', y=0.98)

colors_bar = ['#E63946', '#1D9E75', '#2196F3', '#FF9800']
factor_names = ['Momentum', 'Value', 'Quality', 'Low-Vol']

for idx, (ax, factor, color) in enumerate(
        zip(axes.flat, factor_names, colors_bar)):
    ax.set_facecolor('#1A1D27')

    # Top 15 for each factor
    top15 = factor_table[factor].nlargest(15)
    bars  = ax.barh(range(len(top15)), top15.values,
                    color=color, alpha=0.85)

    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15.index, fontsize=9, color='white')
    ax.set_title(f'Top 15 — {factor} Factor',
                 color='white', fontsize=11, pad=8)
    ax.set_xlabel('Z-Score', color='white', fontsize=9)
    ax.tick_params(colors='white', labelsize=8)
    ax.axvline(0, color='white', linewidth=0.5, alpha=0.3)
    ax.grid(axis='x', alpha=0.15, color='white')
    for sp in ax.spines.values():
        sp.set_color('#333')

plt.tight_layout()
plt.savefig('factor_scores.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.show()
print("\nFactor chart saved as factor_scores.png")

# ════════════════════════════════════════════════════════════
# STEP 7 — COMPOSITE RANKING CHART
# ════════════════════════════════════════════════════════════

fig2, ax = plt.subplots(figsize=(14, 9))
fig2.patch.set_facecolor('#0F1117')
ax.set_facecolor('#1A1D27')

top25 = factor_table.head(25)

bar_colors = ['#E63946' if i < 20 else '#555555'
              for i in range(len(top25))]
bars = ax.barh(range(len(top25)),
               top25['Composite'].values,
               color=bar_colors, alpha=0.9)

ax.set_yticks(range(len(top25)))
ax.set_yticklabels(top25.index, fontsize=10, color='white')
ax.invert_yaxis()
ax.set_title('Composite Factor Ranking — Top 25 Stocks\n'
             '(Red = Selected for portfolio, Grey = Just missed)',
             color='white', fontsize=13, pad=12)
ax.set_xlabel('Composite Z-Score (higher = better)',
              color='white', fontsize=10)
ax.tick_params(colors='white')
ax.axvline(0, color='white', linewidth=0.8, alpha=0.4)
ax.axhline(19.5, color='#E63946', linewidth=1.5,
           linestyle='--', alpha=0.7,
           label='Top 20 cutoff')
ax.legend(fontsize=10, facecolor='#1A1D27',
          labelcolor='white', loc='lower right')
ax.grid(axis='x', alpha=0.15, color='white')
for sp in ax.spines.values():
    sp.set_color('#333')

plt.tight_layout()
plt.savefig('composite_ranking.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.show()
print("Composite ranking chart saved as composite_ranking.png")

# Save factor table for Phase 2
factor_table.to_csv('factor_scores.csv')
top20.to_csv('top20_stocks.csv')
print("\nFactor data saved:")
print("  factor_scores.csv — full universe rankings")
print("  top20_stocks.csv  — selected portfolio stocks")
print("\n✓ Phase 1 complete. Run Phase 2 to backtest this strategy.")

