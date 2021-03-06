# -*- coding: utf-8 -*-
"""LSTM stock prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14sZHFCQFFKdFLNqjrOO2GP84p3_apd-w
"""
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.engine import base_layer
from tensorflow.keras.optimizers import Adam
from custom_classes import CustomScaler
from sklearn.pipeline import Pipeline
import numpy as np


def create_shallow_LSTM(features=49,
                        epochs=40,
                        LSTM_units=1,
                        num_samples=1, 
                        look_back=1,  
                        dropout_rate=0,
                        recurrent_dropout=0,
                        LSTM_units_1=1,
                        verbose=2):
    
  model=Sequential()
    
  model.add(LSTM(units=LSTM_units, recurrent_dropout=recurrent_dropout, stateful=True, return_sequences=True, batch_input_shape=(1, 1, features)))
  model.add(Dropout(dropout_rate))
  model.add(LSTM(units=LSTM_units_1,recurrent_dropout=recurrent_dropout, stateful=True))
  
            
  model.add(Dense(1, activation="sigmoid"))
  model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(), 
        metrics=["accuracy"]
    )
  return model

         
def LSTM_evaluation(X_train_1, X_train_1_lstm):
  #prepare time series split in order to evaluate model properly
  tscv = TimeSeriesSplit(n_splits=20).split(X_train_1_lstm)
  for train_index, test_index in tscv:
      X_train = X_train_1[train_index[0]:train_index[-1]]
      scaler = CustomScaler().fit(X_train)
      X_train_tscv = scaler.transform(X_train)
      
      X_train_tscv=X_train_tscv.reshape(X_train.shape[0], 1, X_train.shape[1]).astype(np.float32) 
  
      model = create_shallow_LSTM(dropout_rate=0.2, LSTM_units=80,LSTM_units_1=30)
      model.fit(X_train_tscv, y_train_1[train_index[0]:train_index[-1]], epochs=20, batch_size=1, verbose=2)
      X_test_batch = X_train_1[test_index[0]:test_index[-1]]
      score = model.evaluate(scaler.transform(X_test_batch).reshape(X_test_batch.shape[0],1,X_test_batch.shape[1]).astype(np.float32) , y_train_1[test_index[0]:test_index[-1]], batch_size=1, verbose=1)
      print("MODEL 1 accuracy:" , score)


def hyperparameter_search(epochs=300):
  model = KerasClassifier(build_fn=create_shallow_LSTM, epochs=epochs, verbose=2, batch_size=1)
  # Define the range
  dropout_rate=[0.1,0.2,0.3]
  recurrent_dropout=[0.1,0.2]
  LSTM_units = [80,100,150,200]
  LSTM_units_1 = [20, 40, 80, 120]
  #  Prepare the Grid
  param_grid = dict(LSTM_units=LSTM_units, recurrent_dropout=recurrent_dropout,
                    dropout_rate=dropout_rate, LSTM_units_1=LSTM_units_1)

  tscv=TimeSeriesSplit(n_splits=20).split(X_train_1_lstm)

  p = Pipeline([("scaler", CustomScaler()),
                ("model", model)])
                
  # GridSearch in action
  grid = GridSearchCV(p, 
                    param_grid=param_grid, 
                    n_jobs=1, 
                    cv=tscv)

  grid_result = grid.fit(X_train_1_lstm, y_train_1)

  print("Best hyperparameters:")

  print("dropout_rate:", grid_result.best_estimator_.get_params()["dropout_rate"])

  print("accuracy of the best model: ", grid_result.best_score_)
  print("LSTM units:", grid_result.best_estimator_.get_params()["LSTM_units"])
  print("LSTM_1 units:", grid_result.best_estimator_.get_params()["LSTM_units_1"])
  return grid_result


