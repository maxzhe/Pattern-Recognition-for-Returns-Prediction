from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, concatenate
from tensorflow.keras.optimizers import SGD, Adam
from mplfinance.original_flavor import candlestick2_ohlc
from pathlib import Path
import matplotlib.pyplot as plt
from custom_classes import CoordConv


def pic(df, name, folder_name):
  Path("/content/"+folder_name+"/").mkdir(parents=True, exist_ok=True)
  fig = plt.figure()
  ax_1 = fig.add_subplot(2, 2, 1)
  candlestick2_ohlc(ax_1, df["Open"], df["High"], df["Low"], df["Close"], width=0.6)
  plt.savefig("/content/"+folder_name+"/"+name + '.png')


def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]


def create_convolution_layers(input_img):
  model = CoordConv(step*2,step*2,with_r=False, filters=32, kernel_size=(3,3), padding="same")(input_img)
  model = BatchNormalization(axis=-1)(model)
  model = Activation('relu')(model)
  model = CoordConv(step*2,step*2,with_r=False, filters=64, kernel_size=(5,5), padding="same")(model)
  model = BatchNormalization(axis=-1)(model)
  model = Activation('relu')(model)
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
  dense = Activation('relu')(dense)
  dense = Dropout(0.2)(dense)
  dense = Dense(128)(dense)
  dense = BatchNormalization()(dense)
  dense = Activation('relu')(dense)

  output = Dense(5, activation='softmax')(dense)

  model = Model(inputs=[GASF_input, GADF_input, MTF_input], outputs=[output])
  if loss == "Kull":
    loss='kullback_leibler_divergence'
  else:
    loss='categorical_crossentropy'
  if opt =="Adam":
    opt = Adam(learning_rate=lr)
  else:
    opt = SGD(learning_rate=lr)

  model.compile(
        loss=loss,
        optimizer=opt,
        metrics=['accuracy', 'Recall']
    )
  return model
