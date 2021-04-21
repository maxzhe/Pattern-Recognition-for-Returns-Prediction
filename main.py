# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from LSTM_model import LSTM_evaluation
from tensorflow.keras.models import load_model
import os
from custom_classes import AddCoords, CoordConv
from ts_enconde import GAF_data_2


def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = u.ewm(com=period-1, adjust=False).mean() / \
         d.ewm(com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + rs)


def calculate_ema(series_y, size=30):
    return series_y.ewm(span=size).mean()


if __name__ == "__main__":
  ###### Stock csv
  sp500 = pd.read_csv("/sp500.csv") #load here any stock
  
  ### Auxillary
  sp500["D"]=np.where(sp500["Close"] > sp500["Close"].shift(1), 1, -1)
  sp500["OBV"]=(sp500["Volume"]*sp500["D"]).cumsum()
  sp500["K"]=(sp500["Close"]-pd.Series(sp500["Low"]).rolling(window=10).min())/(pd.Series(sp500["High"]).rolling(window=10).max()-pd.Series(sp500["Low"]).rolling(window=10).min())

  ### Lagged returns
  sp500["Log_Ret_1d"]=np.log(sp500["Close"] / sp500["Close"].shift(1))
  sp500["Log_Ret_2d"]=sp500["Log_Ret_1d"].shift(1)
  sp500["Log_Ret_3d"]=sp500["Log_Ret_1d"].shift(2)
  sp500["Log_Ret_4d"]=sp500["Log_Ret_1d"].shift(3)
  sp500["Log_Ret_5d"]=sp500["Log_Ret_1d"].shift(4)
  sp500["Log_Ret_2w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=10).sum()
  sp500["Log_Ret_3w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=15).sum()
  sp500["Log_Ret_4w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=20).sum()
  sp500["Log_Ret_8w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=40).sum()
  sp500["Log_Ret_12w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=60).sum()
  sp500["Log_Ret_16w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=80).sum()
  sp500["Log_Ret_20w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=100).sum()
  sp500["Log_Ret_24w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=120).sum()
  sp500["Log_Ret_28w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=140).sum()
  sp500["Log_Ret_32w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=160).sum()
  sp500["Log_Ret_36w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=180).sum()
  sp500["Log_Ret_40w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=200).sum()
  sp500["Log_Ret_44w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=220).sum()
  sp500["Log_Ret_48w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=240).sum()
  sp500["Log_Ret_52w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=260).sum()
  sp500["Log_Ret_56w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=280).sum()
  sp500["Log_Ret_64w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=320).sum()
  sp500["Log_Ret_60w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=300).sum()
  sp500["Log_Ret_68w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=340).sum()
  sp500["Log_Ret_72w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=360).sum()
  sp500["Log_Ret_76w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=380).sum()
  sp500["Log_Ret_80w"]=pd.Series(sp500["Log_Ret_1d"]).rolling(window=400).sum()

  ### Indicators

  sp500["Max_last_52_weeks_signal"]=sp500["Close"] >= pd.Series(sp500["Close"]).rolling(window=260).max()#
  sp500["Mean_week_50_signal"]=sp500["Close"] > pd.Series(sp500["Close"]).rolling(window=50).mean()
  sp500["Mean_week_200_signal"]=sp500["Close"] > pd.Series(sp500["Close"]).rolling(window=50).mean()
  sp500["s4_l36"]=pd.Series(sp500["Close"]).rolling(window=20).mean() > pd.Series(sp500["Close"]).rolling(window=180).mean()
  sp500["s4_l48"]=pd.Series(sp500["Close"]).rolling(window=20).mean() > pd.Series(sp500["Close"]).rolling(window=240).mean()
  sp500["s8_l36"]=pd.Series(sp500["Close"]).rolling(window=40).mean() > pd.Series(sp500["Close"]).rolling(window=180).mean()
  sp500["s8_l48"]=pd.Series(sp500["Close"]).rolling(window=40).mean() > pd.Series(sp500["Close"]).rolling(window=240).mean()
  sp500["s12_l36"]=pd.Series(sp500["Close"]).rolling(window=60).mean() > pd.Series(sp500["Close"]).rolling(window=180).mean()
  sp500["s12_l48"]=pd.Series(sp500["Close"]).rolling(window=60).mean() > pd.Series(sp500["Close"]).rolling(window=240).mean()
  sp500["s4_l36OBV"]=pd.Series(sp500["OBV"]).rolling(window=20).mean() >= pd.Series(sp500["OBV"]).rolling(window=180).mean()
  sp500["s4_l48OBV"]=pd.Series(sp500["OBV"]).rolling(window=20).mean() >= pd.Series(sp500["OBV"]).rolling(window=240).mean()
  sp500["s8_l36OBV"]=pd.Series(sp500["OBV"]).rolling(window=40).mean() >= pd.Series(sp500["OBV"]).rolling(window=180).mean()
  sp500["s8_l48OBV"]=pd.Series(sp500["OBV"]).rolling(window=40).mean() >= pd.Series(sp500["OBV"]).rolling(window=240).mean()
  sp500["s12_l36OBV"]=pd.Series(sp500["OBV"]).rolling(window=60).mean() >= pd.Series(sp500["OBV"]).rolling(window=180).mean()
  sp500["s12_l48OBV"]=pd.Series(sp500["OBV"]).rolling(window=60).mean() >= pd.Series(sp500["OBV"]).rolling(window=240).mean()
  sp500["S_m_36"]=sp500["Close"] > sp500["Close"].shift(36)
  sp500["S_m_48"]=sp500["Close"] > sp500["Close"].shift(48)
  sp500["s.rsi.14"]=RSI(sp500["Close"], 14)>50
  sp500["s.rsi.25"]=RSI(sp500["Close"], 25)>50
  sp500["MACD"]=calculate_ema(sp500["Close"], size=12)-calculate_ema(sp500["Close"], size=26)>calculate_ema(calculate_ema(sp500["Close"], size=12)-calculate_ema(sp500["Close"], size=26), size=9)
  sp500["s.adx"]=sp500["High"] - sp500["High"].shift(1) > sp500["Low"] - sp500["Low"].shift(1)#
  sp500["s_K"]=sp500["K"]>=pd.Series(sp500["K"]).rolling(window=3).mean()
  models = os.listdir("/models/")
  if models is None:
    raise("Create models with train_CNN()")
  for model in models:
    cnn = load_model("/models/"+model, custom_objects={"CoordConv":CoordConv, "AddCoords":AddCoords}))
    step = model.split("_")[-1][:2]
    GASF_array,GADF_array,MTF_array = GAF_data_2(sp500, int(step))
    sp500[,"H&S_"+step] = np.zeros(len(sp500["Open"]))
    sp500[,"IH&S_"+step] = np.zeros(len(sp500["Open"]))
    sp500[,"BBOT_"+step] = np.zeros(len(sp500["Open"]))
    sp500[,"BTOP_"+step] = np.zeros(len(sp500["Open"]))
    predictions = np.argmax(cnn.predict([GASF_array,GADF_array,MTF_array]), axis=-1)
    for idx, i in enumerate(range((int(step)-1),len(sp500["Open"]))):
      if predictions[idx] == 0:
        sp500[i,df.columns.get_loc("BBOT_"+step)] = 1
      elif predictions[idx] == 1:
        sp500[i,df.columns.get_loc("BTOP_"+step)] = 1
      elif predictions[idx] == 2:
        sp500.iloc[i,df.columns.get_loc("H&S_"+step)] = 1
      elif predictions[idx] == 3:
        sp500[i,df.columns.get_loc("IH&S_"+step))] = 1
  sp500["Label"]=np.where(sp500["Close"].shift(-1) > sp500["Close"], 1, 0)
  sp500 = sp500[sp500["Open"] != 0]
  sp500=sp500.dropna("index")
  sp500=sp500.drop(["Adj Close", "Volume", "K", "D", "OBV"], axis=1)
  sp500.reset_index(inplace=True, drop=True)
  

  X_train, y_train = (sp500.iloc[:,5:len(sp500.columns)-1], sp500["Label"])

  X_train_lstm = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1]).astype(np.float32)

  X_train = X_train.reset_index(drop=True)

  #best_params = hyperparameter_search()
  LSTM_evaluation(X_train, X_train_lstm)
