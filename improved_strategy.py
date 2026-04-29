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
UNIVERSE     = list(dict.fromkeys(UNIVERSE))
INITIAL_CAP  = 10000
TOP_N        = 20
SHORT_N      = 10
START_DATE   = '2015-01-01'
END_DATE     = date.today().isoformat()
CURRENT_YEAR = date.today().year
TRADING_DAYS = 252
RISK_FREE    = 0.05

# ── IMPROVEMENT 1: New factor weights ───────────────────────
# Momentum dominant — captures trend premium
FACTOR_WEIGHTS = {
    'momentum': 0.50,    # was 0.35 — increased to capture tech trend
    'low_vol':  0.15,    # was 0.25 — reduced but kept for crash protection
    'trend':    0.25,    # unchanged — 52-week trend strength
    'reversal': 0.10,    # was 0.15 — slightly reduced
}

# ── IMPROVEMENT 2: QQQ core allocation ─────────────────────
QQQ_ALLOCATION    = 0.10    # permanent 10% in QQQ
CASH_BUFFER       = 0.05    # keep 5% in cash always
FACTOR_ALLOCATION = 1.0 - QQQ_ALLOCATION - CASH_BUFFER  # 85%

# ── Execution costs ─────────────────────────────────────────
MEGA_CAPS = {'AAPL','MSFT','NVDA','GOOGL','META','AMZN','V','MA','JPM'}
SPREAD    = {'mega': 0.0005, 'large': 0.0010}
SLIPPAGE  = 0.0005

print("="*65)
print("  IMPROVED FACTOR STRATEGY — 3 UPGRADES APPLIED")
print("="*65)
print(f"\nFactor weights: {FACTOR_WEIGHTS}")
print(f"QQQ allocation: {QQQ_ALLOCATION:.0%} permanent")
print(f"Factor stocks:  {FACTOR_ALLOCATION:.0%} of portfolio")
print(f"Period: {START_DATE} → {END_DATE}")
print(f"\nDownloading data...")

# ════════════════════════════════════════════════════════════
# DOWNLOAD DATA
# ════════════════════════════════════════════════════════════
tickers_needed = list(set(UNIVERSE + ['SPY','QQQ']))
raw    = yf.download(tickers_needed, start=START_DATE,
                     end=END_DATE, auto_adjust=True)
prices = raw['Close'].dropna(axis=1, thresh=int(0.7*len(raw)))
valid  = [t for t in UNIVERSE if t in prices.columns]
spy    = prices['SPY']
qqq    = prices['QQQ'] if 'QQQ' in prices.columns else None
stocks = prices[valid]

print(f"Downloaded: {len(valid)} stocks × {len(prices)} days")

# ════════════════════════════════════════════════════════════
# REGIME DETECTOR (same as production)
# ════════════════════════════════════════════════════════════
def detect_regime(spy_series):
    if len(spy_series) < 200:
        return 'BULL', 1.0
    ma200        = spy_series.rolling(200).mean().iloc[-1]
    above_ma     = spy_series.iloc[-1] > ma200
    spy_rets     = np.log(spy_series/spy_series.shift(1)).dropna()
    realized_vol = spy_rets.tail(21).std() * np.sqrt(252)
    if realized_vol > 0.35:
        return 'CRISIS', 0.50
    elif realized_vol > 0.20 or not above_ma:
        return 'BEAR', 0.75
    else:
        return 'BULL', 1.0

# ════════════════════════════════════════════════════════════
# IMPROVED FACTOR SCORER
# Improvement 1: New weights (momentum dominant)
# ════════════════════════════════════════════════════════════
def score_stocks_improved(hist_prices):
    p = hist_prices.dropna(axis=1, thresh=int(0.8*len(hist_prices)))
    if len(p) < 63:
        return pd.Series(dtype=float)

    r    = np.log(p/p.shift(1)).dropna()
    n    = len(p)
    t1m  = max(0, n-22)
    t12m = max(0, n-253)
    t52w = max(0, n-252)

    def z(s):
        s = pd.Series(s).replace([np.inf,-np.inf], np.nan).dropna()
        if s.std() < 1e-9:
            return s * 0
        return (s - s.mean()) / s.std()

    # Factor calculations
    if t12m < t1m:
        mom = (p.iloc[t1m] - p.iloc[t12m]) / \
               p.iloc[t12m].replace(0, np.nan)
    else:
        mom = pd.Series(0.0, index=p.columns)

    ann_vol = r.tail(252).std() * np.sqrt(252)
    lowvol  = -ann_vol

    low_52w  = p.iloc[t52w:].min()
    high_52w = p.iloc[t52w:].max()
    rng      = (high_52w - low_52w).replace(0, np.nan)
    trend    = (p.iloc[-1] - low_52w) / rng

    rev_1m = -(p.iloc[-1] - p.iloc[t1m]) / \
               p.iloc[t1m].replace(0, np.nan)

    common = (mom.dropna().index
              .intersection(lowvol.dropna().index)
              .intersection(trend.dropna().index)
              .intersection(rev_1m.dropna().index))

    if len(common) < TOP_N + SHORT_N:
        return pd.Series(dtype=float)

    # IMPROVEMENT 1 — momentum-dominant weights
    composite = (
        FACTOR_WEIGHTS['momentum']  * z(mom[common])    +
        FACTOR_WEIGHTS['low_vol']   * z(lowvol[common]) +
        FACTOR_WEIGHTS['trend']     * z(trend[common])  +
        FACTOR_WEIGHTS['reversal']  * z(rev_1m[common])
    )
    return composite.dropna()

# ════════════════════════════════════════════════════════════
# VOLATILITY-TARGETED WEIGHTS
# ════════════════════════════════════════════════════════════
def get_vol_weights(tickers, hist_prices,
                   target_vol=0.15, exposure=1.0,
                   factor_alloc=FACTOR_ALLOCATION):
    p     = hist_prices[tickers].dropna(
                axis=1, thresh=int(0.8*len(hist_prices)))
    r     = np.log(p/p.shift(1)).dropna()
    avail = list(p.columns)
    if not avail:
        return {}

    vols    = r.tail(63).std() * np.sqrt(252)
    vols    = vols.replace(0, np.nan).dropna()
    inv_vol = 1 / vols
    raw_w   = inv_vol / inv_vol.sum()

    port_vol = (r[avail].tail(63).dot(
                    raw_w[avail])).std() * np.sqrt(252)
    scale    = min(target_vol/port_vol, 1.5) if port_vol > 0 else 1.0

    w = (raw_w * scale * exposure * factor_alloc).clip(upper=0.12)
    if w.sum() > 0:
        w = w / w.sum() * factor_alloc * exposure
    return w.to_dict()

# ════════════════════════════════════════════════════════════
# EXECUTION COST
# ════════════════════════════════════════════════════════════
def exec_cost(ticker, trade_pct):
    spread = SPREAD['mega'] if ticker in MEGA_CAPS \
             else SPREAD['large']
    impact = SLIPPAGE * np.sqrt(max(trade_pct, 0) * 100)
    return spread + impact

# ════════════════════════════════════════════════════════════
# WALK-FORWARD BACKTEST — TWO STRATEGIES SIMULTANEOUSLY
# Strategy A: Long-only improved (Improvements 1+2)
# Strategy B: Long-short market neutral (Improvement 3)
# ════════════════════════════════════════════════════════════
print("\nRunning improved walk-forward backtest...")
print("B=Bull E=Bear C=Crisis | running both long-only & long-short\n")

rebal_dates  = prices.resample('MS').first().index
rebal_dates  = rebal_dates[rebal_dates >= '2015-06-01']

records      = []
prev_lo_w    = {}
prev_ls_w    = {}

for i in range(len(rebal_dates)-1):
    t0 = rebal_dates[i]
    t1 = rebal_dates[i+1]

    hist     = stocks[stocks.index <= t0]
    spy_hist = spy[spy.index <= t0]

    regime, exposure = detect_regime(spy_hist)
    scores           = score_stocks_improved(hist)

    if len(scores) < TOP_N + SHORT_N:
        continue

    top20   = scores.nlargest(TOP_N).index.tolist()
    bottom10 = scores.nsmallest(SHORT_N).index.tolist()

    # ── Long-only weights (Improvements 1+2) ─────────────
    lo_weights = get_vol_weights(
        top20, hist,
        target_vol    = 0.15,
        exposure      = exposure,
        factor_alloc  = FACTOR_ALLOCATION
    )

    # ── Long-short weights (Improvement 3) ───────────────
    ls_long_w  = get_vol_weights(
        top20, hist,
        target_vol   = 0.15,
        exposure     = exposure,
        factor_alloc = 0.50    # 50% long
    )
    ls_short_w = get_vol_weights(
        bottom10, hist,
        target_vol   = 0.15,
        exposure     = exposure,
        factor_alloc = 0.50    # 50% short
    )

    # ── Get period returns ────────────────────────────────
    mask     = (prices.index >= t0) & (prices.index <= t1)
    p_window = prices[mask]
    if len(p_window) < 2:
        continue

    def period_ret(ticker):
        if ticker not in p_window.columns:
            return None
        p0 = p_window[ticker].iloc[0]
        p1 = p_window[ticker].iloc[-1]
        return (p1-p0)/p0 if p0 > 0 else None

    # ── Long-only portfolio return ────────────────────────
    lo_ret   = 0.0
    lo_cost  = 0.0
    for tk, w in lo_weights.items():
        r = period_ret(tk)
        if r is not None:
            lo_ret += w * r
        old_w   = prev_lo_w.get(tk, 0)
        trd_pct = abs(w - old_w)
        if trd_pct > 0.001:
            lo_cost += trd_pct * exec_cost(tk, trd_pct) * 2

    # IMPROVEMENT 2: Add QQQ return
    if qqq is not None:
        qqq_window = qqq[mask]
        if len(qqq_window) > 1:
            qqq_ret = (qqq_window.iloc[-1]/qqq_window.iloc[0]) - 1
            lo_ret += QQQ_ALLOCATION * float(qqq_ret)

    lo_ret     -= lo_cost
    prev_lo_w   = lo_weights.copy()

    # ── Long-short portfolio return ───────────────────────
    ls_ret  = 0.0
    ls_cost = 0.0

    # Long leg
    for tk, w in ls_long_w.items():
        r = period_ret(tk)
        if r is not None:
            ls_ret += w * float(r)
        old_w   = prev_ls_w.get(tk, 0)
        trd_pct = abs(w - old_w)
        if trd_pct > 0.001:
            ls_cost += trd_pct * exec_cost(tk, trd_pct) * 2

    # Short leg (subtract returns — we profit when they fall)
    for tk, w in ls_short_w.items():
        r = period_ret(tk)
        if r is not None:
            ls_ret -= w * float(r)   # short = negative exposure
        old_w   = prev_ls_w.get(tk, 0)
        trd_pct = abs(w - old_w)
        if trd_pct > 0.001:
            ls_cost += trd_pct * exec_cost(tk, trd_pct) * 3  # short costs more

    ls_ret    -= ls_cost
    prev_ls_w  = {**ls_long_w, **ls_short_w}

    # ── SPY return ────────────────────────────────────────
    spy_r = period_ret('SPY')
    spy_ret = float(spy_r) if spy_r is not None else 0.0

    records.append({
        'date':      t0,
        'long_only': float(np.clip(lo_ret, -0.40, 0.40)),
        'long_short':float(np.clip(ls_ret, -0.30, 0.30)),
        'spy':       spy_ret,
        'regime':    regime,
        'exposure':  exposure,
    })

    print({'BULL':'B','BEAR':'E','CRISIS':'C'}[regime],
          end='', flush=True)

print(f"\n\nMonths simulated: {len(records)}")

# ════════════════════════════════════════════════════════════
# BUILD RESULTS
# ════════════════════════════════════════════════════════════
df = pd.DataFrame(records).set_index('date')
df.index = pd.to_datetime(df.index)

lo_cum  = (1+df['long_only']).cumprod()  * INITIAL_CAP
ls_cum  = (1+df['long_short']).cumprod() * INITIAL_CAP
spy_cum = (1+df['spy']).cumprod()        * INITIAL_CAP

split_idx  = int(len(df)*0.70)
split_date = df.index[split_idx]

# ════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════
def metrics(r, name):
    r        = r.dropna()
    ann_ret  = r.mean() * 12
    ann_vol  = r.std()  * np.sqrt(12)
    sharpe   = (ann_ret-RISK_FREE)/ann_vol if ann_vol > 0 else 0
    cum      = (1+r).cumprod()
    mdd      = ((cum-cum.cummax())/cum.cummax()).min()
    calmar   = ann_ret/abs(mdd) if mdd != 0 else 0
    down_vol = r[r<0].std()*np.sqrt(12)
    sortino  = (ann_ret-RISK_FREE)/down_vol if down_vol > 0 else 0
    win_rate = (r>0).mean()
    return dict(name=name, ann_ret=ann_ret, ann_vol=ann_vol,
                sharpe=sharpe, sortino=sortino, mdd=mdd,
                calmar=calmar, win_rate=win_rate,
                final=INITIAL_CAP*(1+r).cumprod().iloc[-1])

m_lo  = metrics(df['long_only'],  'Long-Only Improved')
m_ls  = metrics(df['long_short'], 'Long-Short Neutral')
m_spy = metrics(df['spy'],        'SPY Benchmark')

# IS / OOS
m_lo_is  = metrics(df['long_only'].iloc[:split_idx],  'LO In-Sample')
m_lo_oos = metrics(df['long_only'].iloc[split_idx:],  'LO Out-of-Sample')
m_ls_is  = metrics(df['long_short'].iloc[:split_idx], 'LS In-Sample')
m_ls_oos = metrics(df['long_short'].iloc[split_idx:], 'LS Out-of-Sample')

# ── PRINT REPORT ───────────────────────────────────────────
print("\n" + "═"*72)
print(f"  IMPROVED STRATEGY REPORT (2015–{CURRENT_YEAR})")
print("═"*72)
print(f"\n{'Metric':<22}{'Long-Only':>16}{'Long-Short':>16}"
      f"{'SPY':>12}{'vs SPY (LO)':>13}")
print("─"*72)

for label, key, pct in [
    ('Annual Return',     'ann_ret',  True),
    ('Annual Volatility', 'ann_vol',  True),
    ('Sharpe Ratio',      'sharpe',   False),
    ('Sortino Ratio',     'sortino',  False),
    ('Max Drawdown',      'mdd',      True),
    ('Calmar Ratio',      'calmar',   False),
    ('Win Rate',          'win_rate', True),
]:
    lo  = m_lo[key];  ls = m_ls[key];  sp = m_spy[key]
    alp = lo - sp
    fmt = (lambda v: f'{v:.2%}') if pct else (lambda v: f'{v:.3f}')
    print(f"{label:<22}{fmt(lo):>16}{fmt(ls):>16}"
          f"{fmt(sp):>12}{fmt(alp):>13}")

print("─"*72)
print(f"  Final value:")
print(f"  Long-Only:   ${m_lo['final']:>10,.0f}")
print(f"  Long-Short:  ${m_ls['final']:>10,.0f}")
print(f"  SPY:         ${m_spy['final']:>10,.0f}")

# ── OOS comparison ─────────────────────────────────────────
print(f"\n{'═'*72}")
print(f"  OUT-OF-SAMPLE VALIDATION")
print(f"{'═'*72}")
print(f"\n{'Metric':<20}{'LO IS':>12}{'LO OOS':>12}"
      f"{'LS IS':>12}{'LS OOS':>12}")
print("─"*72)
for label, key, pct in [
    ('Annual Return', 'ann_ret', True),
    ('Sharpe Ratio',  'sharpe',  False),
    ('Max Drawdown',  'mdd',     True),
]:
    fmt = (lambda v: f'{v:.2%}') if pct else (lambda v: f'{v:.3f}')
    print(f"{label:<20}"
          f"{fmt(m_lo_is[key]):>12}{fmt(m_lo_oos[key]):>12}"
          f"{fmt(m_ls_is[key]):>12}{fmt(m_ls_oos[key]):>12}")

# Degradation check
lo_deg = (m_lo_is['sharpe']-m_lo_oos['sharpe'])/max(abs(m_lo_is['sharpe']),0.001)*100
ls_deg = (m_ls_is['sharpe']-m_ls_oos['sharpe'])/max(abs(m_ls_is['sharpe']),0.001)*100
print(f"\n  Long-Only  Sharpe degradation: {lo_deg:.1f}%")
print(f"  Long-Short Sharpe degradation: {ls_deg:.1f}%")

for deg, name in [(lo_deg,'Long-Only'),(ls_deg,'Long-Short')]:
    if deg < 30:
        print(f"  ✓ {name}: ROBUST")
    elif deg < 50:
        print(f"  ⚠ {name}: MODERATE overfitting")
    else:
        print(f"  ✗ {name}: OVERFIT")

# ── Regime analysis ────────────────────────────────────────
print(f"\n{'═'*72}")
print(f"  RETURNS BY REGIME")
print(f"{'═'*72}")
for regime in ['BULL','BEAR','CRISIS']:
    mask = df['regime'] == regime
    if mask.sum() == 0:
        continue
    lo_r  = df[mask]['long_only'].mean()  * 12
    ls_r  = df[mask]['long_short'].mean() * 12
    spy_r = df[mask]['spy'].mean()        * 12
    cnt   = mask.sum()
    print(f"  {regime:<8} ({cnt:>3} months) "
          f"LO: {lo_r:>+7.1%}  "
          f"LS: {ls_r:>+7.1%}  "
          f"SPY: {spy_r:>+7.1%}")

# ════════════════════════════════════════════════════════════
# FINAL TEARSHEET — 6 PANELS
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0F1117')
fig.suptitle(
    f'Improved Factor Strategy — 3 Upgrades Applied (2015–{CURRENT_YEAR})',
    fontsize=15, fontweight='bold', color='white', y=0.98)
gs  = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35,
                         top=0.93, bottom=0.05)
pnl = dict(facecolor='#1A1D27')
tc  = 'white'
gkw = dict(alpha=0.15, color='white')
C   = {'LO':'#E63946','LS':'#1D9E75','S':'#2196F3',
       'BULL':'#1D9E75','BEAR':'#FF9800','CRISIS':'#E63946'}

# ── Panel 1: Growth curves ───────────────────────────────
ax1 = fig.add_subplot(gs[0,:2], **pnl)
ax1.plot(lo_cum,  color=C['LO'], lw=2.5,
         label=f"Long-Only + QQQ  ${m_lo['final']:,.0f}")
ax1.plot(ls_cum,  color=C['LS'], lw=2.0, ls='-.',
         label=f"Long-Short       ${m_ls['final']:,.0f}")
ax1.plot(spy_cum, color=C['S'],  lw=1.8, ls='--',
         label=f"SPY Benchmark    ${m_spy['final']:,.0f}")
ax1.axvline(split_date, color='yellow', lw=1.5,
            ls='--', alpha=0.7, label='IS/OOS Split')
ax1.axhline(INITIAL_CAP, color='white', lw=0.5, alpha=0.3)
ax1.set_title(f'Portfolio Growth — $10,000 (2015–{CURRENT_YEAR})',
              color=tc, fontsize=11)
ax1.set_ylabel('Value ($)', color=tc)
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))
ax1.legend(fontsize=9, facecolor='#1A1D27', labelcolor=tc)
ax1.tick_params(colors=tc); ax1.grid(**gkw)
for sp in ax1.spines.values(): sp.set_color('#333')

# ── Panel 2: Factor weights pie ──────────────────────────
ax2 = fig.add_subplot(gs[0,2], **pnl)
fw_labels = ['Momentum\n50%','Trend\n25%',
             'Low-Vol\n15%','Reversal\n10%']
fw_vals   = [0.50, 0.25, 0.15, 0.10]
fw_colors = ['#E63946','#FF9800','#2196F3','#1D9E75']
wedges, texts = ax2.pie(
    fw_vals, labels=fw_labels,
    colors=fw_colors, startangle=90,
    textprops={'color':tc,'fontsize':9})
ax2.set_title('Factor Weights\n(Improvement 1)',
              color=tc, fontsize=10)

# ── Panel 3: Drawdown comparison ─────────────────────────
ax3 = fig.add_subplot(gs[1,0], **pnl)
for nm, ret, col in [('Long-Only', df['long_only'],  C['LO']),
                      ('Long-Short',df['long_short'], C['LS']),
                      ('SPY',       df['spy'],        C['S'])]:
    c  = (1+ret).cumprod()
    dd = (c-c.cummax())/c.cummax()
    ax3.fill_between(dd.index, dd, 0, alpha=0.25, color=col)
    ax3.plot(dd.index, dd, color=col, lw=0.9, label=nm)
ax3.set_title('Drawdown', color=tc, fontsize=10)
ax3.set_ylabel('Drawdown', color=tc)
ax3.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x,_: f'{x:.0%}'))
ax3.legend(fontsize=8, facecolor='#1A1D27', labelcolor=tc)
ax3.tick_params(colors=tc, labelsize=8); ax3.grid(**gkw)
for sp in ax3.spines.values(): sp.set_color('#333')

# ── Panel 4: Rolling Sharpe ──────────────────────────────
ax4 = fig.add_subplot(gs[1,1], **pnl)
for nm, ret, col in [('Long-Only', df['long_only'],  C['LO']),
                      ('Long-Short',df['long_short'], C['LS']),
                      ('SPY',       df['spy'],        C['S'])]:
    rs = ((ret.rolling(12).mean()*12 - RISK_FREE) /
          (ret.rolling(12).std()*np.sqrt(12)))
    ax4.plot(rs.index, rs, color=col, lw=1.5, label=nm)
ax4.axhline(0, color='white', lw=0.8, ls='--', alpha=0.4)
ax4.axhline(1, color=C['LS'], lw=1, ls=':', alpha=0.6)
ax4.axvline(split_date, color='yellow', lw=1, ls='--', alpha=0.5)
ax4.set_title('Rolling 12M Sharpe', color=tc, fontsize=10)
ax4.set_ylabel('Sharpe Ratio', color=tc)
ax4.legend(fontsize=8, facecolor='#1A1D27', labelcolor=tc)
ax4.tick_params(colors=tc, labelsize=8); ax4.grid(**gkw)
for sp in ax4.spines.values(): sp.set_color('#333')

# ── Panel 5: IS vs OOS bar ───────────────────────────────
ax5 = fig.add_subplot(gs[1,2], **pnl)
metrics_labels = ['Ann.Ret','Sharpe','WinRate']
lo_is_v  = [m_lo_is['ann_ret'],  m_lo_is['sharpe'],  m_lo_is['win_rate']]
lo_oos_v = [m_lo_oos['ann_ret'], m_lo_oos['sharpe'], m_lo_oos['win_rate']]
ls_is_v  = [m_ls_is['ann_ret'],  m_ls_is['sharpe'],  m_ls_is['win_rate']]
ls_oos_v = [m_ls_oos['ann_ret'], m_ls_oos['sharpe'], m_ls_oos['win_rate']]
x = np.arange(len(metrics_labels))
w = 0.2
ax5.bar(x-1.5*w, lo_is_v,  w, color=C['LO'],  alpha=0.9,  label='LO IS')
ax5.bar(x-0.5*w, lo_oos_v, w, color=C['LO'],  alpha=0.5,  label='LO OOS')
ax5.bar(x+0.5*w, ls_is_v,  w, color=C['LS'],  alpha=0.9,  label='LS IS')
ax5.bar(x+1.5*w, ls_oos_v, w, color=C['LS'],  alpha=0.5,  label='LS OOS')
ax5.set_xticks(x)
ax5.set_xticklabels(metrics_labels, color=tc, fontsize=8)
ax5.set_title('IS vs OOS Validation', color=tc, fontsize=10)
ax5.legend(fontsize=7, facecolor='#1A1D27', labelcolor=tc)
ax5.tick_params(colors=tc, labelsize=8)
ax5.grid(**gkw, axis='y')
for sp in ax5.spines.values(): sp.set_color('#333')

# ── Panel 6: Monthly heatmap (Long-Only) ─────────────────
ax6 = fig.add_subplot(gs[2,:], **pnl)
hm          = df['long_only'].to_frame('r')
hm['Year']  = hm.index.year
hm['Month'] = hm.index.month
pivot       = hm.pivot_table('r','Year','Month')
pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                  'Jul','Aug','Sep','Oct','Nov','Dec']
sns.heatmap(pivot, annot=True, fmt='.1%',
            cmap='RdYlGn', center=0,
            linewidths=0.4, ax=ax6,
            cbar_kws={'label':'Monthly Return'},
            annot_kws={'size':8})
ax6.set_title(
    f'Monthly Returns — Long-Only Improved (2015–{CURRENT_YEAR})',
    color=tc, fontsize=11)
ax6.tick_params(colors=tc, labelsize=9)

plt.savefig('improved_tearsheet.png', dpi=150,
            bbox_inches='tight', facecolor='#0F1117')
plt.show()
print(f"\nImproved tearsheet saved as improved_tearsheet.png")

df.to_csv('improved_returns.csv')
print("Returns saved to improved_returns.csv")
print(f"\n{'═'*65}")
print(f"  ALL 3 IMPROVEMENTS COMPLETE")
print(f"  Long-Only:  momentum-tilted + QQQ core + vol sizing")
print(f"  Long-Short: pure factor alpha, market neutral")
print(f"{'═'*65}")