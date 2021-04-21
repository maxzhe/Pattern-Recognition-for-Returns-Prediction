from pyts.image import GramianAngularField,MarkovTransitionField
import numpy as np
import os
import cv2 as cv

def search(array, to_find):
  result = []
  for idx, i in enumerate(array):
    if (i == to_find).all():
      result.append(idx)
  return result

def remove_values_from_list(the_list, val):
  return [value for value in the_list if value != val]


def open(img_path):
  img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
  return img


def OHLC_to_channels(figure):
  directory = "/Merged_data/"
  fields = ["MTF", "GASF","GADF"]
  merged = []
  files_temp = os.listdir(directory+figure+"/"+str(step)+"/GASF/Open")
  for x in fields:
    temp = []
    for idx,i in enumerate(files_temp):
      ope = open(directory + figure +"/"+str(step)+"/"+ x +"/Open/" + i)
      high = open(directory + figure + "/"+str(step)+"/"+ x +"/High/"+ i)
      close = open(directory + figure +"/"+str(step)+"/"+ x +"/Close/"+ i)
      low = open(directory + figure +"/"+str(step)+"/"+ x +"/Low/"+ i)
      temp.append(np.array([ope,high,close,low]).reshape(1, step, step, 4))
    temp = np.vstack(temp)
    merged.append(temp)
  return tuple(merged)


def OHLC_in_one(figure):
  directory = "/Merged_data/"
  fields = ["MTF", "GASF","GADF"]
  files_temp = os.listdir(directory+figure+"/"+str(step)+"/GASF/Open")
  merged = []
  for x in fields:
    temp = []
    for idx,i in enumerate(files_temp):
      ope = open(directory + figure +"/"+str(step)+"/"+ x +"/Open/" + i)
      high = open(directory + figure + "/"+str(step)+"/"+ x +"/High/"+ i)
      close = open(directory + figure +"/"+str(step)+"/"+ x +"/Close/"+ i)
      low = open(directory + figure +"/"+str(step)+"/"+ x +"/Low/"+ i)
      array = np.hstack((ope, high))
      array_1 = np.hstack((close, low))
      array = np.vstack((array, array_1))
      temp.append(array.reshape(step*2, step*2, 1))
    temp = np.stack(temp)
    merged.append(temp)
  return tuple(merged)


def GAF_data_2(df, step):
  col = ["Open", "High", "Close","Low"]
  gasf = GramianAngularField(image_size=step, method="summation")
  gadf = GramianAngularField(image_size=step, method="difference")
  mtf = MarkovTransitionField(image_size=step)
  X_mtf = []
  X_gasf = []
  X_gadf = []
  for i in range((step-1),len(df[col[0]])):
    high = max(df["High"][i-(step-1):i+1])
    low = min(df["Low"][i-(step-1):i+1])
    ts_1 = [(x-low)/(high - low) for x in list(df[col[0]][i-(step-1):i+1])]
    ts_2 = [(x-low)/(high - low) for x in list(df[col[1]][i-(step-1):i+1])]
    ts_3 = [(x-low)/(high - low) for x in list(df[col[2]][i-(step-1):i+1])]
    ts_4 = [(x-low)/(high - low) for x in list(df[col[3]][i-(step-1):i+1])]
    ope = np.round(mtf.fit_transform([ts_1])[0]*255)
    high = np.round(mtf.fit_transform([ts_2])[0]*255)
    close = np.round(mtf.fit_transform([ts_3])[0]*255)
    low = np.round(mtf.fit_transform([ts_4])[0]*255)
    mtf_oh = np.hstack((ope, high))
    mtf_cl = np.hstack((close, low))
    mtf_ohcl = np.vstack((mtf_oh,mtf_cl))
    X_mtf.append(mtf_ohcl.reshape(step*2,step*2,1))
  X_mtf = np.stack(X_mtf)

  for i in range((step-1),len(df[col[0]])):
    high = max(df["High"][i-(step-1):i+1])
    low = min(df["Low"][i-(step-1):i+1])
    ts_1 = [(x-low)/(high - low) for x in list(df[col[0]][i-(step-1):i+1])]
    ts_2 = [(x-low)/(high - low) for x in list(df[col[1]][i-(step-1):i+1])]
    ts_3 = [(x-low)/(high - low) for x in list(df[col[2]][i-(step-1):i+1])]
    ts_4 = [(x-low)/(high - low) for x in list(df[col[3]][i-(step-1):i+1])]
    gadf_oh = np.hstack((np.round(gadf.fit_transform([ts_1])[0]*255), np.round(gadf.fit_transform([ts_2])[0]*255)))
    gadf_cl = np.hstack((np.round(gadf.fit_transform([ts_3])[0]*255), np.round(gadf.fit_transform([ts_4])[0]*255)))
    gadf_ohcl = np.vstack((gadf_oh,gadf_cl))
    X_gadf.append(gadf_ohcl.reshape(step*2,step*2,1))
  X_gadf = np.stack(X_gadf)

  for i in range((step-1),len(df[col[0]])):
    high = max(df["High"][i-(step-1):i+1])
    low = min(df["Low"][i-(step-1):i+1])
    ts_1 = [(x-low)/(high - low) for x in list(df[col[0]][i-(step-1):i+1])]
    ts_2 = [(x-low)/(high - low) for x in list(df[col[1]][i-(step-1):i+1])]
    ts_3 = [(x-low)/(high - low) for x in list(df[col[2]][i-(step-1):i+1])]
    ts_4 = [(x-low)/(high - low) for x in list(df[col[3]][i-(step-1):i+1])]
    gasf_oh = np.hstack((np.round(gasf.fit_transform([ts_1])[0]*255),  np.round(gasf.fit_transform([ts_2])[0]*255)))
    gasf_cl = np.hstack((np.round(gasf.fit_transform([ts_3])[0]*255), np.round(gasf.fit_transform([ts_4])[0]*255)))
    gasf_ohcl = np.vstack((gasf_oh,gasf_cl))
    X_gasf.append(gasf_ohcl.reshape(step*2,step*2,1))
  X_gasf = np.stack(X_gasf)
  return (X_gasf, X_gadf, X_mtf)


def save_to_GAF_img(df, file, step):
  OHLC = ["Open", "High", "Low", "Close"]
  high = max(df["High"])
  low = min(df["Low"])

  for col in OHLC:
      Path("/content/GASF/" + col + "/").mkdir(parents=True, exist_ok=True)
      Path("/content/GADF/" + col + "/").mkdir(parents=True, exist_ok=True)
      Path("/content/MTF/" + col + "/").mkdir(parents=True, exist_ok=True)
      gasf = GramianAngularField(image_size=step, method="summation")
      gadf = GramianAngularField(image_size=step, method="difference")
      mtf = MarkovTransitionField(image_size=step)
      ts_norm = [(i-low)/(high - low) for i in list(df[col])]
      X_mtf = mtf.fit_transform([ts_norm])
      X_gasf = gasf.fit_transform([ts_norm])
      X_gadf = gadf.fit_transform([ts_norm])
      
      plt.imsave("/content/other_n/GASF/" +  col + "/" + file, X_gasf[0], cmap="gray")
      plt.imsave("/content/other_n/GADF/" +  col + "/" + file, X_gadf[0], cmap="gray")
      plt.imsave("/content/other_n/MTF/" +  col + "/" + file, X_mtf[0], cmap="gray")
