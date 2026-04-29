import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
from datetime import date
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════
UNIVERSE = [
    'AAPL','MSFT','NVDA','GOOGL','META','AVGO','ORCL','CRM',
    'JPM','BAC','WFC','GS','MS','BLK','AXP','USB',
    'JNJ','UNH','LLY','PFE','ABBV','MRK','TMO','ABT','DHR',
    'AMZN','HD','MCD','NKE','COST','WMT','LOW',
    'CAT','HON','UPS','LMT','GE','MMM',
    'XOM','CVX','COP','EOG','PSX','VLO','MPC',
    'NFLX','DIS','CMCSA','VZ','T',
    'NEE','DUK','SO','AEP','SPG','AMT',
    'PG','KO','PEP','PM','CL','GIS',
    'V','MA','CVS','MCK'
]
UNIVERSE       = list(dict.fromkeys(UNIVERSE))
INITIAL_CAP    = 10000
TOP_N          = 20
COST_PER_TRADE = 0.001
TRADING_DAYS   = 252
START_DATE     = '2015-01-01'
END_DATE       = date.today().isoformat()
CURRENT_YEAR   = date.today().year

print("="*60)
print("  WALK-FORWARD BACKTESTING ENGINE")
print("="*60)
print(f"\nUniverse:  {len(UNIVERSE)} stocks")
print(f"Period:    {START_DATE} → {END_DATE}")
print(f"\nDownloading data...")

# ════════════════════════════════════════════════════════════
# DOWNLOAD ALL DATA UPFRONT
# ════════════════════════════════════════════════════════════
all_tickers = list(set(UNIVERSE + ['SPY']))
raw         = yf.download(all_tickers, start=START_DATE,
                          end=END_DATE, auto_adjust=True)
prices      = raw['Close'].dropna(axis=1, thresh=int(0.7*len(raw)))
valid       = [t for t in UNIVERSE if t in prices.columns]
spy_prices  = prices['SPY']
stock_prices = prices[valid]

print(f"Downloaded: {len(valid)} stocks × {len(prices)} trading days")

# ════════════════════════════════════════════════════════════
# WALK-FORWARD LOOP
# ════════════════════════════════════════════════════════════
print("\nRunning walk-forward backtest...")
print("Each dot = 1 month:")

rebal_dates = prices.resample('MS').first().index
rebal_dates = rebal_dates[rebal_dates >= '2015-06-01']

records    = []
prev_top20 = []

for i in range(len(rebal_dates) - 1):
    t0 = rebal_dates[i]
    t1 = rebal_dates[i + 1]

    hist   = stock_prices[stock_prices.index <= t0]
    hist   = hist.dropna(axis=1, thresh=int(0.7*len(hist)))
    avail  = list(hist.columns)

    if len(hist) < 63:
        continue

    log_ret = np.log(hist / hist.shift(1)).dropna()

    n      = len(hist)
    t_1m   = max(0, n - 22)
    t_12m  = max(0, n - 253)
    if t_12m >= t_1m - 1:
        continue

    mom = (hist.iloc[t_1m] - hist.iloc[t_12m]) / \
           hist.iloc[t_12m].replace(0, np.nan)

    lookback  = min(252, len(log_ret))
    ann_vol   = log_ret.tail(lookback).std() * np.sqrt(TRADING_DAYS)
    lowvol    = -ann_vol

    def z(s):
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        return (s - s.mean()) / (s.std() + 1e-9)

    common    = mom.dropna().index.intersection(lowvol.dropna().index)
    composite = (z(mom[common]) + z(lowvol[common])) / 2
    composite = composite.dropna()

    if len(composite) < TOP_N:
        continue

    top20 = composite.nlargest(TOP_N).index.tolist()

    mask     = (prices.index >= t0) & (prices.index <= t1)
    p_window = prices[mask]

    if len(p_window) < 2:
        continue

    avail_top20 = [s for s in top20 if s in p_window.columns]
    if len(avail_top20) < 5:
        continue

    p_start    = p_window[avail_top20].iloc[0]
    p_end      = p_window[avail_top20].iloc[-1]
    stock_rets = (p_end - p_start) / p_start
    port_ret   = stock_rets.mean()

    new_stocks = len(set(top20) - set(prev_top20))
    turnover   = new_stocks / TOP_N
    cost       = turnover * COST_PER_TRADE * 2
    port_ret  -= cost
    prev_top20 = top20

    if 'SPY' in p_window.columns:
        spy_ret = (p_window['SPY'].iloc[-1] /
                   p_window['SPY'].iloc[0]) - 1
    else:
        spy_ret = 0.0

    eq_stocks = [s for s in valid if s in p_window.columns]
    eq_start  = p_window[eq_stocks].iloc[0]
    eq_end    = p_window[eq_stocks].iloc[-1]
    eq_ret    = ((eq_end - eq_start) / eq_start).mean()

    records.append({
        'date':     t0,
        'strategy': float(port_ret),
        'spy':      float(spy_ret),
        'equal':    float(eq_ret),
        'top20':    ','.join(top20)
    })

    print('.', end='', flush=True)

print(f"\n\nMonths simulated: {len(records)}")

# ════════════════════════════════════════════════════════════
# BUILD RESULTS
# ════════════════════════════════════════════════════════════
df = pd.DataFrame(records).set_index('date')
df.index = pd.to_datetime(df.index)

print(f"\nMonthly return range check:")
print(f"  Strategy: {df['strategy'].min():.1%} to {df['strategy'].max():.1%}")
print(f"  SPY:      {df['spy'].min():.1%} to {df['spy'].max():.1%}")

df['strategy'] = df['strategy'].clip(-0.40, 0.40)

strat_cum = (1 + df['strategy']).cumprod() * INITIAL_CAP
spy_cum   = (1 + df['spy']).cumprod()      * INITIAL_CAP
equal_cum = (1 + df['equal']).cumprod()    * INITIAL_CAP

# ════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ════════════════════════════════════════════════════════════
def get_metrics(monthly_rets, name):
    r        = monthly_rets.dropna()
    ann_ret  = r.mean() * 12
    ann_vol  = r.std()  * np.sqrt(12)
    sharpe   = (ann_ret - 0.05) / ann_vol if ann_vol > 0 else 0
    cum      = (1 + r).cumprod()
    mdd      = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar   = ann_ret / abs(mdd) if mdd != 0 else 0
    down_vol = r[r < 0].std() * np.sqrt(12)
    sortino  = (ann_ret - 0.05) / down_vol if down_vol > 0 else 0
    win_rate = (r > 0).mean()
    total    = cum.iloc[-1] - 1
    final    = INITIAL_CAP * cum.iloc[-1]
    return dict(name=name, ann_ret=ann_ret, ann_vol=ann_vol,
                sharpe=sharpe, sortino=sortino, mdd=mdd,
                calmar=calmar, win_rate=win_rate,
                total=total, final=final)

ms = get_metrics(df['strategy'], 'Factor Strategy')
mb = get_metrics(df['spy'],      'SPY Benchmark')
me = get_metrics(df['equal'],    'Equal Weight')

# ── PRINT REPORT ───────────────────────────────────────────
print("\n" + "═"*65)
print(f"  FINAL PERFORMANCE REPORT (2015–{CURRENT_YEAR})")
print("═"*65)
print(f"\n{'Metric':<24}{'Factor Strategy':>17}{'SPY':>13}{'Equal Wt':>13}")
print("─"*65)
for label, key, pct in [
    ('Annual Return',     'ann_ret', True),
    ('Annual Volatility', 'ann_vol', True),
    ('Total Return',      'total',   True),
    ('Sharpe Ratio',      'sharpe',  False),
    ('Sortino Ratio',     'sortino', False),
    ('Max Drawdown',      'mdd',     True),
    ('Calmar Ratio',      'calmar',  False),
    ('Win Rate',          'win_rate',True),
]:
    vals = [ms[key], mb[key], me[key]]
    fmt  = (lambda v: f'{v:.2%}') if pct else (lambda v: f'{v:.3f}')
    print(f"{label:<24}" + "".join(f"{fmt(v):>13}" for v in vals))

print("─"*65)
for name, m in [('Factor', ms), ('SPY', mb), ('Equal', me)]:
    print(f"  {name} $10k → ${m['final']:>10,.0f}")

alpha = ms['ann_ret'] - mb['ann_ret']
te    = (df['strategy'] - df['spy']).std() * np.sqrt(12)
print(f"\n  Alpha vs SPY:       {alpha:+.2%}/year")
if te > 0:
    print(f"  Information Ratio:  {alpha/te:.3f}")
print("═"*65)

# ════════════════════════════════════════════════════════════
# 4-PANEL TEARSHEET
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 13))
fig.patch.set_facecolor('#0F1117')
fig.suptitle(f'Factor Strategy — Walk-Forward Backtest (2015–{CURRENT_YEAR})',
             fontsize=16, fontweight='bold', color='white', y=0.98)
gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.3,
                         top=0.93, bottom=0.05)
pnl = dict(facecolor='#1A1D27')
tc  = 'white'
gkw = dict(alpha=0.15, color='white')
C   = {'F':'#E63946', 'S':'#2196F3', 'E':'#FF9800'}

# ── Growth ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :], **pnl)
ax1.plot(strat_cum, color=C['F'], lw=2.5,
         label=f"Factor   ${ms['final']:,.0f}")
ax1.plot(spy_cum,   color=C['S'], lw=1.8, ls='--',
         label=f"SPY      ${mb['final']:,.0f}")
ax1.plot(equal_cum, color=C['E'], lw=1.8, ls=':',
         label=f"Equal    ${me['final']:,.0f}")
ax1.axhline(INITIAL_CAP, color='white', lw=0.6, alpha=0.3)
ax1.set_title(f'Portfolio Growth — $10,000 Initial Investment (2015–{CURRENT_YEAR})',
              color=tc, fontsize=12)
ax1.set_ylabel('Value ($)', color=tc)
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))
ax1.legend(fontsize=10, facecolor='#1A1D27', labelcolor=tc)
ax1.tick_params(colors=tc)
ax1.grid(**gkw)
for sp in ax1.spines.values(): sp.set_color('#333')

# ── Drawdown ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0], **pnl)
for nm, ret, col in [('Factor', df['strategy'], C['F']),
                      ('SPY',    df['spy'],      C['S']),
                      ('Equal',  df['equal'],    C['E'])]:
    c  = (1+ret).cumprod()
    dd = (c - c.cummax()) / c.cummax()
    ax2.fill_between(dd.index, dd, 0, alpha=0.3, color=col)
    ax2.plot(dd.index, dd, color=col, lw=0.9, label=nm)
ax2.set_title('Drawdown Over Time', color=tc, fontsize=11)
ax2.set_ylabel('Drawdown', color=tc)
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
ax2.legend(fontsize=9, facecolor='#1A1D27', labelcolor=tc)
ax2.tick_params(colors=tc, labelsize=8)
ax2.grid(**gkw)
for sp in ax2.spines.values(): sp.set_color('#333')

# ── Rolling Sharpe ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1], **pnl)
for nm, ret, col in [('Factor', df['strategy'], C['F']),
                      ('SPY',    df['spy'],      C['S'])]:
    rs = ((ret.rolling(12).mean()*12 - 0.05) /
          (ret.rolling(12).std()*np.sqrt(12)))
    ax3.plot(rs.index, rs, color=col, lw=1.5, label=nm)
ax3.axhline(0, color='white', lw=0.8, ls='--', alpha=0.4)
ax3.axhline(1, color='#1D9E75', lw=1, ls=':', alpha=0.6)
ax3.set_title('Rolling 12-Month Sharpe Ratio', color=tc, fontsize=11)
ax3.set_ylabel('Sharpe Ratio', color=tc)
ax3.legend(fontsize=9, facecolor='#1A1D27', labelcolor=tc)
ax3.tick_params(colors=tc, labelsize=8)
ax3.grid(**gkw)
for sp in ax3.spines.values(): sp.set_color('#333')

# ── Monthly Heatmap ────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, :], **pnl)
hm          = df['strategy'].to_frame('r')
hm['Year']  = hm.index.year
hm['Month'] = hm.index.month
pivot       = hm.pivot_table('r', 'Year', 'Month')
pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                  'Jul','Aug','Sep','Oct','Nov','Dec']
sns.heatmap(pivot, annot=True, fmt='.1%',
            cmap='RdYlGn', center=0,
            linewidths=0.4, ax=ax4,
            cbar_kws={'label': 'Monthly Return'},
            annot_kws={'size': 8})
ax4.set_title(f'Monthly Returns Heatmap — Factor Strategy (2015–{CURRENT_YEAR})',
              color=tc, fontsize=11)
ax4.tick_params(colors=tc, labelsize=9)

plt.savefig('backtest_tearsheet.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.show()
print("\nTearsheet saved!")

df.to_csv('monthly_returns.csv')
print("Monthly returns saved!")
print(f"\n✓ Phase 2 complete — data current to {END_DATE}")