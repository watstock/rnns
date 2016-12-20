"""
Helper methods.
"""

from dateutil.relativedelta import relativedelta

def next_work_day(date, distance=1):
  weekend = set([5, 6])
  last_date = date + relativedelta(days=1)

  while last_date.weekday() in weekend:
    last_date += relativedelta(days=1)
  
  return last_date
