"""
Helper methods.
"""

from dateutil.relativedelta import relativedelta

def next_work_day(date, distance=1):
  weekend = set([5, 6])
  last_date = date
  work_days = []
  while len(work_days) < distance:
    last_date += relativedelta(days=1)
    if last_date.weekday() not in weekend:
      work_days.append(last_date)
  
  return work_days[-1]
