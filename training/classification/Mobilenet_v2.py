■データセット前準備

from PIL import Image 
import os, glob
import numpy as np
from PIL import ImageFile
# IOError: image file is truncated (0 bytes not processed)回避のため
ImageFile.LOAD_TRUNCATED_IMAGES = True

classes = ["dog", "cat", "giraffe", "elephant", "lion"]
num_classes = len(classes)#リストや文字列など様々な型のオブジェクトのサイズ（要素数や文字数）を取得
image_size = 150
num_testdata = 300

X_train = []
X_test  = []
y_train = []
y_test  = []

#enumerate関数：要素のインデックスと要素を同時に取り出す事が出来る。
#for 変数1, 変数2 in enumerate(リスト):
#print(‘{0}:{1}’.format(変数1, 変数2))
#1行目のfor 変数1, 変数2 in enumerate(list):では、listをenumerateで取得できる間
#ずっと、変数1と変数2に代入し続けるfor文を使用。


for index, classlabel in enumerate(classes):
    photos_dir = "./images/" + classlabel
    files = glob.glob(photos_dir + "/*.jpg") #引数に指定されたパターンにマッチするファイルパス名を取得
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        if i < num_testdata:
            X_test.append(data)
            y_test.append(index)
        else:

            # angleに代入される値
            # -20
            # 0
            # 画像を10度ずつ回転
            for angle in range(-20, 20, 20):

                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                y_train.append(index)
                # FLIP_LEFT_RIGHT　は 左右反転
                #img_trains = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                #data = np.asarray(img_trains)
                #X_train.append(data)
                #y_train.append(index)

print(len(X_train))
print(len(X_test))
X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)

xy = (X_train, X_test, y_train, y_test)
np.save("./dog_cat_giraffe_elephant_lion.npy", xy)


#モデルの拡張（VGG16+全結合分類器）

from keras.applications.mobilenet_v2 import MobileNetV2

# weightsは重みのチェックポイント。include_topは全結合層を含めるかどうか。imagenetの1000クラス分類に対応。
conv_base = MobileNetV2(weights = "imagenet",
                 include_top=False,
                 input_shape=(150,150,3))
conv_base.summary()


#■ネットワークの作成
import tensorflow as tf
from keras import layers
from keras import models
from keras import optimizers
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
%matplotlib inline


"""
データを読み込む関数
"""
from keras.layers import GlobalAveragePooling2D, Dense, Dropout

def load_data():
    X_train, X_test, y_train, y_test = np.load("./dog_cat_giraffe_elephant_lion.npy", allow_pickle=True)
    # 入力データの各画素値を0-1の範囲で正規化(学習コストを下げるため)
    X_train = X_train.astype("float") / 255
    X_test  = X_test.astype("float") / 255
    # to_categorical()にてラベルをone hot vector化
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test  = np_utils.to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test



def train(X_train, y_train, X_test, y_test):
  model = models.Sequential()
  model.add(conv_base)
  model.add(GlobalAveragePooling2D())
  model.add(layers.Dense(5,activation='softmax'))

  conv_base.trainable = False
  model.summary()

  model.compile(loss="categorical_crossentropy",
                 optimizer=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                 metrics=["acc"])
  history =model.fit(X_train, y_train,#画像とラベルデータ
                 batch_size=128,
                 epochs=10,
                 validation_data=(X_test, y_test),
                 shuffle=True)
  model.save('./cats_dogs_giraffes_elephants_lions_classification.h5')

  acc = history.history["acc"]
  val_acc = history.history["val_acc"]
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]

  epochs = range(1,len(acc) + 1)

  plt.plot(epochs, acc,"bo",label="Training Acc")
  plt.plot(epochs, val_acc,"b",label="Validation Acc")
  plt.legend()

  plt.figure()

  plt.plot(epochs,loss,"bo",label="Training Loss")
  plt.plot(epochs,val_loss,"b",label="Validation Loss")
  plt.legend()

  plt.show()

  return model



# データの読み込み
X_train, y_train, X_test, y_test = load_data()

# モデルの学習
model = train(X_train, y_train, X_test, y_test)

