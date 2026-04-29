import subprocess
import sys
import os
from datetime import date, datetime
import calendar

def is_first_trading_day():
    """
    Returns True if today is the first trading day of the month.
    Accounts for weekends — if 1st is Saturday/Sunday,
    first trading day is Monday.
    """
    today      = date.today()
    first      = date(today.year, today.month, 1)

    # Find first weekday of month
    if first.weekday() == 5:    # Saturday
        first_trading = date(today.year, today.month, 3)
    elif first.weekday() == 6:  # Sunday
        first_trading = date(today.year, today.month, 2)
    else:
        first_trading = first

    return today == first_trading

def is_weekday():
    """Returns True if today is Monday-Friday."""
    return date.today().weekday() < 5

def run_rebalance():
    """Run monthly rebalance."""
    print(f"[{datetime.now()}] Running monthly rebalance...")
    script = os.path.join(os.path.dirname(__file__), 'live_trader.py')
    result = subprocess.run(
        [sys.executable, script],
        input='1\n',                    # auto-select option 1
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
    log_run('REBALANCE', result.returncode == 0)

def run_daily_track():
    """Run daily portfolio tracker."""
    print(f"[{datetime.now()}] Running daily tracker...")
    script = os.path.join(os.path.dirname(__file__), 'live_trader.py')
    result = subprocess.run(
        [sys.executable, script],
        input='2\n',                    # auto-select option 2
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
    log_run('DAILY_TRACK', result.returncode == 0)

def log_run(run_type, success):
    """Log every automated run."""
    log_file = os.path.join(
        os.path.dirname(__file__), 'automation_log.csv')
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()},{run_type},"
                f"{'SUCCESS' if success else 'FAILED'}\n")

def main():
    """
    Main scheduler logic:
    - First trading day of month → rebalance
    - Every other weekday → daily track
    """
    print(f"\n{'='*50}")
    print(f"  AUTOMATED TRADING SCHEDULER")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    if not is_weekday():
        print("Weekend — no action needed.")
        return

    if is_first_trading_day():
        print("First trading day of month — running rebalance!")
        run_rebalance()
    else:
        print("Regular trading day — running daily tracker.")
        run_daily_track()

if __name__ == "__main__":
    main()