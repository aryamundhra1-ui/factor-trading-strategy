import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import os
from datetime import datetime, date
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import config

warnings.filterwarnings('ignore')

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

# ── Allocation split ────────────────────────────────────────
QQQ_ALLOCATION    = 0.10   # permanent 10% in QQQ
CASH_BUFFER       = 0.05   # keep 5% cash always
FACTOR_ALLOCATION = 0.85   # 85% in factor-selected stocks

# ════════════════════════════════════════════════════════════
# CONNECTION
# ════════════════════════════════════════════════════════════
def connect_to_alpaca():
    try:
        client  = TradingClient(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            paper=True
        )
        account = client.get_account()
        print("="*55)
        print("  CONNECTED TO ALPACA PAPER TRADING")
        print("="*55)
        print(f"  Account status:    {account.status}")
        print(f"  Portfolio value:   ${float(account.portfolio_value):,.2f}")
        print(f"  Cash available:    ${float(account.cash):,.2f}")
        print(f"  Buying power:      ${float(account.buying_power):,.2f}")
        return client, account
    except Exception as e:
        print(f"  Connection failed: {e}")
        print("  Check your API keys in config.py")
        return None, None

# ════════════════════════════════════════════════════════════
# RISK MANAGER
# ════════════════════════════════════════════════════════════
def check_risk(account):
    portfolio_value = float(account.portfolio_value)
    drawdown        = (portfolio_value - config.INITIAL_CAPITAL) \
                      / config.INITIAL_CAPITAL
    if drawdown < -config.MAX_DRAWDOWN_STOP:
        print(f"\n  CIRCUIT BREAKER TRIGGERED")
        print(f"  Portfolio down {drawdown:.1%} — trading halted")
        return False
    print(f"\n  Risk check passed")
    print(f"  Portfolio P&L: {drawdown:+.2%}")
    return True

# ════════════════════════════════════════════════════════════
# IMPROVED FACTOR SCORER
# Momentum-dominant weights (50/15/25/10)
# 4 factors: momentum, low-vol, trend, reversal
# ════════════════════════════════════════════════════════════
def score_stocks_live():
    print(f"\n  Downloading live market data...")
    raw    = yf.download(UNIVERSE, period='2y',
                         auto_adjust=True, progress=False)
    prices = raw['Close'].dropna(axis=1,
                                  thresh=int(0.8*len(raw)))
    valid  = list(prices.columns)
    print(f"  Scoring {len(valid)} stocks with 4 factors...")

    log_ret = np.log(prices / prices.shift(1)).dropna()
    n       = len(prices)
    t_1m    = max(0, n - 22)
    t_12m   = max(0, n - 253)
    t_52w   = max(0, n - 252)

    def z(s):
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.std() < 1e-9:
            return s * 0
        return (s - s.mean()) / s.std()

    # Factor 1: Momentum 12m-1m — weight 50%
    if t_12m < t_1m:
        mom = (prices.iloc[t_1m] - prices.iloc[t_12m]) / \
               prices.iloc[t_12m].replace(0, np.nan)
    else:
        mom = pd.Series(0.0, index=prices.columns)

    # Factor 2: Low Volatility — weight 15%
    ann_vol = log_ret.tail(252).std() * np.sqrt(252)
    lowvol  = -ann_vol

    # Factor 3: 52-week trend strength — weight 25%
    low_52w  = prices.iloc[t_52w:].min()
    high_52w = prices.iloc[t_52w:].max()
    rng      = (high_52w - low_52w).replace(0, np.nan)
    trend    = (prices.iloc[-1] - low_52w) / rng

    # Factor 4: Short-term reversal — weight 10%
    rev_1m = -(prices.iloc[-1] - prices.iloc[t_1m]) / \
               prices.iloc[t_1m].replace(0, np.nan)

    # Align all 4 factors
    common = (mom.dropna().index
              .intersection(lowvol.dropna().index)
              .intersection(trend.dropna().index)
              .intersection(rev_1m.dropna().index))

    if len(common) < config.TOP_N:
        print(f"  Warning: only {len(common)} stocks available")
        common = mom.dropna().index.intersection(
                 lowvol.dropna().index)

    # Momentum-dominant composite
    composite = (
        0.50 * z(mom[common])    +
        0.15 * z(lowvol[common]) +
        0.25 * z(trend[common])  +
        0.10 * z(rev_1m[common])
    ).dropna()

    top20 = composite.nlargest(config.TOP_N)

    print(f"\n  TOP {config.TOP_N} STOCKS (+ QQQ core):")
    print(f"  {'Ticker':<8} {'Score':>8}  {'Factor':<12}")
    print(f"  {'─'*32}")
    for ticker, score in top20.items():
        print(f"  {ticker:<8} {score:>+8.3f}")
    print(f"  {'QQQ':<8} {'CORE':>8}  "
          f"{'permanent 10%':<12}")

    return top20.index.tolist(), prices

# ════════════════════════════════════════════════════════════
# POSITION SIZER
# 85% factor stocks (vol-targeted) + 10% QQQ + 5% cash
# ════════════════════════════════════════════════════════════
def calculate_positions(account, top20_tickers, prices):
    portfolio_value  = float(account.portfolio_value)
    n                = len(top20_tickers)

    # Volatility-targeted sizing within 85% factor allocation
    log_ret  = np.log(prices / prices.shift(1)).dropna()
    avail    = [t for t in top20_tickers if t in log_ret.columns]
    vols     = log_ret[avail].tail(63).std() * np.sqrt(252)
    vols     = vols.replace(0, np.nan).dropna()
    inv_vol  = 1 / vols
    raw_w    = inv_vol / inv_vol.sum()    # inverse vol weights

    # Scale to 85% of portfolio, cap each at 10%
    weights  = (raw_w * FACTOR_ALLOCATION).clip(upper=0.10)
    if weights.sum() > 0:
        weights = weights / weights.sum() * FACTOR_ALLOCATION

    positions    = {}
    latest       = prices.iloc[-1]

    for ticker in avail:
        if ticker not in latest.index:
            continue
        w      = float(weights.get(ticker, 1/n * FACTOR_ALLOCATION))
        price  = float(latest[ticker])
        dollar = portfolio_value * w
        shares = int(dollar / price)
        if shares > 0:
            positions[ticker] = {
                'shares': shares,
                'price':  price,
                'value':  shares * price,
                'weight': w
            }

    total = sum(p['value'] for p in positions.values())
    print(f"\n  POSITION SIZING (volatility-targeted):")
    print(f"  Portfolio value:   ${portfolio_value:,.2f}")
    print(f"  Factor stocks:     ${total:,.2f} "
          f"({total/portfolio_value:.1%})")
    print(f"  QQQ allocation:    ${portfolio_value*QQQ_ALLOCATION:,.2f} "
          f"({QQQ_ALLOCATION:.0%})")
    print(f"  Cash buffer:       ${portfolio_value*CASH_BUFFER:,.2f} "
          f"({CASH_BUFFER:.0%})")
    return positions

# ════════════════════════════════════════════════════════════
# ORDER EXECUTOR
# ════════════════════════════════════════════════════════════
def execute_rebalance(client, account, new_positions):
    print(f"\n  EXECUTING IMPROVED REBALANCE...")
    print(f"  Strategy: 85% factor + 10% QQQ + 5% cash")

    # Close all existing positions
    existing = client.get_all_positions()
    if existing:
        print(f"\n  Closing {len(existing)} existing positions...")
        client.close_all_positions(cancel_orders=True)
        import time
        time.sleep(5)
        print(f"  All positions closed ✓")
    else:
        print(f"  No existing positions")

    orders_placed = []

    # ── Place factor stock orders ─────────────────────────
    print(f"\n  Placing {len(new_positions)} factor stock orders...")
    for ticker, pos in new_positions.items():
        try:
            order = MarketOrderRequest(
                symbol        = ticker,
                qty           = pos['shares'],
                side          = OrderSide.BUY,
                time_in_force = TimeInForce.DAY
            )
            result = client.submit_order(order)
            orders_placed.append({
                'ticker':    ticker,
                'shares':    pos['shares'],
                'est_price': pos['price'],
                'est_value': pos['value'],
                'weight':    pos['weight'],
                'order_id':  str(result.id),
                'type':      'factor'
            })
            print(f"  BUY {pos['shares']:>5} {ticker:<6} "
                  f"@ ~${pos['price']:>8.2f} "
                  f"= ${pos['value']:>10,.2f} "
                  f"({pos['weight']:.1%})")
        except Exception as e:
            print(f"  Failed {ticker}: {e}")

    # ── Place QQQ core order ──────────────────────────────
    try:
        portfolio_value = float(client.get_account().portfolio_value)
        qqq_dollar      = portfolio_value * QQQ_ALLOCATION

        qqq_data  = yf.download('QQQ', period='2d',
                                 progress=False,
                                 auto_adjust=True)
        qqq_price = float(qqq_data['Close'].iloc[-1].iloc[0] if hasattr(qqq_data['Close'].iloc[-1], '__iter__') else qqq_data['Close'].iloc[-1])
        qqq_shares = int(qqq_dollar / qqq_price)

        if qqq_shares > 0:
            qqq_order = MarketOrderRequest(
                symbol        = 'QQQ',
                qty           = qqq_shares,
                side          = OrderSide.BUY,
                time_in_force = TimeInForce.DAY
            )
            result = client.submit_order(qqq_order)
            orders_placed.append({
                'ticker':    'QQQ',
                'shares':    qqq_shares,
                'est_price': qqq_price,
                'est_value': qqq_shares * qqq_price,
                'weight':    QQQ_ALLOCATION,
                'order_id':  str(result.id),
                'type':      'core'
            })
            print(f"\n  BUY {qqq_shares:>5} QQQ    "
                  f"@ ~${qqq_price:>8.2f} "
                  f"= ${qqq_shares*qqq_price:>10,.2f} "
                  f"[CORE 10%]")
    except Exception as e:
        print(f"  QQQ order failed: {e}")

    print(f"\n  Total orders placed: {len(orders_placed)}")
    print(f"  ({len(orders_placed)-1} factor stocks + 1 QQQ)")
    return orders_placed

# ════════════════════════════════════════════════════════════
# TRADE LOGGER
# ════════════════════════════════════════════════════════════
def log_trades(orders_placed):
    log_file = 'trade_log.csv'
    today    = date.today().isoformat()
    new_rows = []
    for order in orders_placed:
        new_rows.append({
            'date':      today,
            'ticker':    order['ticker'],
            'action':    'BUY',
            'type':      order.get('type', 'factor'),
            'shares':    order['shares'],
            'est_price': order['est_price'],
            'est_value': order['est_value'],
            'weight':    order.get('weight', 0),
            'order_id':  order['order_id']
        })
    new_df = pd.DataFrame(new_rows)
    if os.path.exists(log_file):
        existing = pd.read_csv(log_file)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(log_file, index=False)
    print(f"\n  Trade log saved to {log_file}")

# ════════════════════════════════════════════════════════════
# PORTFOLIO SNAPSHOT
# ════════════════════════════════════════════════════════════
def save_snapshot(client, account):
    snap_file = 'portfolio_history.csv'
    today     = date.today().isoformat()
    port_val  = float(account.portfolio_value)
    try:
        spy = float(yf.download('SPY', period='2d',
                    progress=False,
                    auto_adjust=True)['Close'].iloc[-1])
    except:
        spy = None
    row = pd.DataFrame([{
        'date':            today,
        'portfolio_value': port_val,
        'cash':            float(account.cash),
        'spy_price':       spy,
        'pnl_pct':        (port_val - config.INITIAL_CAPITAL) /
                           config.INITIAL_CAPITAL
    }])
    if os.path.exists(snap_file):
        existing = pd.read_csv(snap_file)
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row
    combined.to_csv(snap_file, index=False)
    print(f"  Snapshot saved to {snap_file}")

# ════════════════════════════════════════════════════════════
# DAILY TRACKER
# ════════════════════════════════════════════════════════════
def run_daily_tracker():
    print("\n" + "="*55)
    print("  DAILY PORTFOLIO TRACKER")
    print("="*55)
    client, account = connect_to_alpaca()
    if not client:
        return

    port_val   = float(account.portfolio_value)
    pnl_pct    = (port_val - config.INITIAL_CAPITAL) / \
                  config.INITIAL_CAPITAL
    pnl_dollar = port_val - config.INITIAL_CAPITAL

    print(f"\n  Date:            {date.today()}")
    print(f"  Portfolio value: ${port_val:>12,.2f}")
    print(f"  P&L:             ${pnl_dollar:>+12,.2f} "
          f"({pnl_pct:+.2%})")

    positions = client.get_all_positions()
    if positions:
        print(f"\n  HOLDINGS ({len(positions)} positions):")
        print(f"  {'Ticker':<8}{'Shares':>8}{'Price':>10}"
              f"{'Value':>12}{'P&L':>10}{'Type':>8}")
        print(f"  {'─'*58}")

        # Sort by value descending
        for pos in sorted(positions,
                          key=lambda x: float(x.market_value),
                          reverse=True):
            ptype = 'CORE' if pos.symbol == 'QQQ' else 'factor'
            print(f"  {pos.symbol:<8}"
                  f"{float(pos.qty):>8.0f}"
                  f"  ${float(pos.current_price):>8.2f}"
                  f"  ${float(pos.market_value):>10,.2f}"
                  f"  {float(pos.unrealized_plpc):>+8.1%}"
                  f"  {ptype:>6}")

    save_snapshot(client, account)
    print("\n  Daily tracking complete ✓")

# ════════════════════════════════════════════════════════════
# MONTHLY REBALANCER
# ════════════════════════════════════════════════════════════
def run_monthly_rebalance():
    print("\n" + "="*55)
    print("  MONTHLY REBALANCE — IMPROVED STRATEGY")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55)
    print(f"\n  Factor weights: Momentum 50% | Trend 25% | "
          f"Low-Vol 15% | Reversal 10%")
    print(f"  Allocation:     85% factor stocks | "
          f"10% QQQ | 5% cash")

    client, account = connect_to_alpaca()
    if not client:
        return

    if not check_risk(account):
        return

    top20_tickers, prices = score_stocks_live()
    positions = calculate_positions(account, top20_tickers, prices)
    orders    = execute_rebalance(client, account, positions)
    log_trades(orders)
    save_snapshot(client, account)

    print(f"\n{'='*55}")
    print(f"  REBALANCE COMPLETE")
    print(f"  {len(orders)-1} factor stocks + QQQ placed")
    print(f"  Next rebalance: first trading day of next month")
    print(f"{'='*55}")

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  FACTOR TRADING SYSTEM — IMPROVED v2.0")
    print("  Momentum 50% | Trend 25% | Low-Vol 15% | Rev 10%")
    print("  85% Factor Stocks + 10% QQQ + 5% Cash")
    print("="*55)
    print("\nWhat would you like to do?")
    print("  1 — Run monthly rebalance (places trades)")
    print("  2 — Run daily tracker (no trades)")
    print("  3 — Check account status only")

    choice = input("\nEnter 1, 2, or 3: ").strip()

    if choice == "1":
        run_monthly_rebalance()
    elif choice == "2":
        run_daily_tracker()
    elif choice == "3":
        connect_to_alpaca()
    else:
        print("Invalid choice.")