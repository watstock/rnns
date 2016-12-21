"""
Helper methods.
"""

from dateutil.relativedelta import relativedelta
from pymongo import MongoClient
import os
import datetime

def next_work_day(date, distance=1):
  weekend = set([5, 6])
  last_date = date
  work_days = []
  while len(work_days) < distance:
    last_date += relativedelta(days=1)
    if last_date.weekday() not in weekend:
      work_days.append(last_date)
  
  return work_days[-1]

def build_model_params(architectures, timesteps, steps_ahead):
  params = []
  for arch in architectures:
    for tstep in timesteps:
        for step_ahead in steps_ahead:
          params.append(arch + [tstep] + [step_ahead])
  return params

def plot_data(df):

  import matplotlib.pyplot as plt

  df.plot()
  plt.show()

def save_prediction_to_db(data):

  MONGODB_CONNECTION = os.environ['MONGODB_CONNECTION']
  client = MongoClient(MONGODB_CONNECTION)
  
  db = client.watstock
  collection = db.prediction_models

  prediction = data

  now = datetime.datetime.utcnow()
  prediction['timestamp'] = now
  prediction['date'] = now.strftime('%Y-%m-%d')

  prediction_id = collection.insert_one(prediction).inserted_id
  print('Prediction saved to the db:', prediction_id)
