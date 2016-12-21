"""
Runner scheduler.
"""

import schedule
import time

import fast_runner

def fast_job():
  print('\nComputing fast models...')
  fast_runner.main()

def main():

  print('Scheduler started')

  # run fast models every hour
  schedule.every().hour.do(fast_job)

  while 1:
    schedule.run_pending()
    time.sleep(1)

if __name__ == "__main__":
  main()
