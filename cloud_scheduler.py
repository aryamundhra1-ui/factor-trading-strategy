import schedule
import time
import subprocess
import sys
import os
from datetime import date, datetime

def is_first_trading_day():
    today = date.today()
    first = date(today.year, today.month, 1)
    if first.weekday() == 5:
        first_trading = date(today.year, today.month, 3)
    elif first.weekday() == 6:
        first_trading = date(today.year, today.month, 2)
    else:
        first_trading = first
    return today == first_trading

def run_script(option):
    script = os.path.join(os.path.dirname(__file__), 'live_trader.py')
    result = subprocess.run(
        [sys.executable, script],
        input=f'{option}\n',
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
    return result.returncode == 0

def morning_job():
    """Runs every weekday at 9:35am EST."""
    print(f"\n[{datetime.now()}] Morning job triggered")
    if is_first_trading_day():
        print("First trading day — REBALANCING")
        success = run_script('1')
    else:
        print("Regular day — TRACKING")
        success = run_script('2')

    status = 'SUCCESS' if success else 'FAILED'
    print(f"[{datetime.now()}] Job complete: {status}")

# Schedule every weekday at 9:35am
schedule.every().monday.at("09:35").do(morning_job)
schedule.every().tuesday.at("09:35").do(morning_job)
schedule.every().wednesday.at("09:35").do(morning_job)
schedule.every().thursday.at("09:35").do(morning_job)
schedule.every().friday.at("09:35").do(morning_job)

print("="*50)
print("  CLOUD SCHEDULER RUNNING")
print("  Rebalance: first trading day of month at 9:35am")
print("  Track:     every other weekday at 9:35am")
print("="*50)

while True:
    schedule.run_pending()
    time.sleep(60)    # check every minute