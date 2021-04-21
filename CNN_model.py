from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, concatenate
from tensorflow.keras.optimizers import SGD, Adam
from mplfinance.original_flavor import candlestick2_ohlc
from pathlib import Path
import matplotlib.pyplot as plt
from custom_classes import CoordConv
from sklearn.model_selection import train_test_split
import os



def pic(df, name, folder_name):
  Path("/content/"+folder_name+"/").mkdir(parents=True, exist_ok=True)
  fig = plt.figure()
  ax_1 = fig.add_subplot(2, 2, 1)
  candlestick2_ohlc(ax_1, df["Open"], df["High"], df["Low"], df["Close"], width=0.6)
  plt.savefig("/content/"+folder_name+"/"+name + ".png")


def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]


def create_convolution_layers(input_img):
  model = CoordConv(step*2,step*2,with_r=False, filters=32, kernel_size=(3,3), padding="same")(input_img)
  model = BatchNormalization(axis=-1)(model)
  model = Activation("relu")(model)
  model = CoordConv(step*2,step*2,with_r=False, filters=64, kernel_size=(5,5), padding="same")(model)
  model = BatchNormalization(axis=-1)(model)
  model = Activation("relu")(model)
  return model


def build_net(opt="Adam", loss="Kull",lr=0.001):
  GASF_input = Input(shape=(step*2,step*2,1))
  GASF_model = create_convolution_layers(GASF_input)

  GADF_input = Input(shape=(step*2,step*2,1))
  GADF_model = create_convolution_layers(GADF_input)

  MTF_input = Input(shape=(step*2,step*2,1))
  MTF_model = create_convolution_layers(MTF_input)

  conv = concatenate([GASF_model,GADF_model, MTF_model])

  conv = Flatten()(conv)

  dense = Dense(128)(conv)
  dense = BatchNormalization()(dense)
  dense = Activation("relu")(dense)
  dense = Dropout(0.2)(dense)
  dense = Dense(128)(dense)
  dense = BatchNormalization()(dense)
  dense = Activation("relu")(dense)

  output = Dense(5, activation="softmax")(dense)

  model = Model(inputs=[GASF_input, GADF_input, MTF_input], outputs=[output])
  if loss == "Kull":
    loss="kullback_leibler_divergence"
  else:
    loss="categorical_crossentropy"
    
  if opt =="Adam":
    opt = Adam(learning_rate=lr)
  else:
    opt = SGD(learning_rate=lr)

  model.compile(
        loss=loss,
        optimizer=opt,
        metrics=["accuracy", "Recall"]
    )
  return model



def train_CNN():
  figs = ["H&S", "IH&S","BBOT", "BTOP", "other"]
  steps = range(15,20)
  for step in steps:
    model = build_net()
    files = []
    for idx, i in enumerate(figs):
      GADF, GASF, MTF = OHLC_in_one(i)
      files += os.listdir("/Merged_data/"+i+"/")
      data_MTF = np.vstack([data_MTF, MTF]).astype(np.float32) if idx != 0 else MTF
      data_GASF = np.vstack([data_GASF, GASF]).astype(np.float32) if idx != 0 else GASF
      data_GADF = np.vstack([data_GADF, GADF]).astype(np.float32) if idx != 0 else GADF
    labels = to_categorical([0 if "BBOT" == row.split("_")[1]
                            else 1 if "BTOP" == row.split("_")[1]
                            else 2 if "H&S" == row.split("_")[1]
                            else 3 if "IH&S" == row.split("_")[1]
                            else 4 for row in files],num_classes=5)
    X_GASF_train, X_GASF_test, X_GADF_train, X_GADF_test, X_MTF_train, X_MTF_test, y_train, y_test = train_test_split(data_GASF,data_GADF,data_MTF, labels, test_size=0.8)
    validation_data = ([X_GASF_test, X_GADF_test, X_MTF_test], y_test)
    filepath = "/models/model_+"str(step)"+.h5"
  
    checkpoint = ModelCheckpoint(filepath, verbose=2, save_best_only=True, mode="min")

    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)

    model.fit(x=[X_GASF_train, X_GADF_train, X_MTF_train],
                        y=y_train,
                        epochs=epochs,
                        batch_size=32,
                        validation_data=validation_data,
                        callbacks=[checkpoint,es])

