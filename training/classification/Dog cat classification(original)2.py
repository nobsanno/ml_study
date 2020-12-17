#前準備

#ディレクトリの作成

#├── train
#│   ├── cats
#│   └── dogs
#└── validation
#    ├── cats
#    └── dogs

#Kaggleのデータセットから以下の枚数だけ、上記ディレクトリにコピー

#訓練用：2000枚　検証用：1000枚


#訓練用画像のコピー

#犬、猫、それぞれ1000枚の画像をそれぞれ訓練用画像にする

#ファイル名は、cat.{0-999}.jpgのように連番になっているので、0-999の数字をfor文で回す


import shutil

fnames = ["cat.{}.jpg".format(i) for i in range(1000)]

for fname in fnames:
    from_data = "./train1/" + fname　　#train1 = Kaggle data
    to_data = "./train/cats/" + fname
    shutil.copyfile(from_data, to_data)
    
fnames = ["dog.{}.jpg".format(i) for i in range(1000)]

for fname in fnames:
    from_data = "train1/" + fname
    to_data = "./train/dogs/" + fname
    shutil.copyfile(from_data, to_data)
--------------------------------------------------------------------

#検証用画像のコピー

#犬と猫、それぞれ500枚ずつを検証用画像とする。

fnames = ["cat.{}.jpg".format(i) for i in range(1000,1500)]

for fname in fnames:
    from_data = "./train1/" + fname
    to_data = "./validation/cats/" + fname
    shutil.copyfile(from_data, to_data)

fnames = ["dog.{}.jpg".format(i) for i in range(1000,1500)]

for fname in fnames:
    from_data = "./train1/" + fname
    to_data = "./validation/dogs/" + fname
    shutil.copyfile(from_data, to_data)


#ファイル数の確認

#ファイル数の確認には、os.listdirの長さを数える。

print("train cat:{}".format(len(os.listdir("./train/cats/"))))
print("train dog:{}".format(len(os.listdir("./train/dogs/"))))

print("validation cat:{}".format(len(os.listdir("./validation/cats/"))))
print("validation dog:{}".format(len(os.listdir("./validation/dogs/"))))


#出力は以下の通り。

#train cat:1000
#train dog:1000
#validation cat:500
#validation dog:500


#ネットワークの作成

import tensorflow as tf
from tf.keras import layers
from tf.keras import models
from tf.keras import optimizers

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(1,activation="sigmoid"))

model.summary()


#最後の層の活性化関数には、sigmoidを使用。

from keras import optimizers

model.compile(loss="binary_crossentropy",
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=["acc"])

#今回のような2値分類では、活性化関数は「sigmoid」、損失関数は「binary_crossentropy」を使うのが一般的のよう。


#データの前処理

#学習にかけるために、画像ファイルに以下の前処理をしておく。

#①画像ファイルを読み込む。
#②浮動小数点数型（float型）にする。
#③ピクセル値（0-255）を、[0,1]の範囲の値にする


#kerasのImageDataGeneratorを使うとこの処理を自動的にやってくれる。


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode="nearest")

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "./train",
    target_size=(150,150),
    batch_size=20,
    class_mode="binary"
)

validation_generator = validation_datagen.flow_from_directory(
    "./validation",
    target_size=(150,150),
    batch_size=20,
    class_mode="binary"
)


#target_sizeは、画像のサイズ。上では150,150のサイズにリサイズ。
#batch_sizeは、一度に処理する画像の枚数。20枚を1バッチとする。
#class_modeは、”binary”として二値のラベルを作成。
 

#下記が出力される

#Found 2000 images belonging to 2 classes.
#Found 1000 images belonging to 2 classes.



#3classesが出た場合

#!find '.' -name '*.ipynb_checkpoints' -exec rm -r {} +

#ipynb_checkpoints フォルダの削除コード




#内容を確認

for data,label in train_generator:
    print(data.shape)
    print(label.shape)
    break

#下記が出力される。

#(20, 150, 150, 3)
#(20,)


#学習

#訓練用は2000枚あるので１バッチ20枚処理するとすると100ステップ必要。

history = model.fit_generator(train_generator,
                             steps_per_epoch=100,
                             epochs=100,
                             validation_data=validation_generator,
                             validation_steps=50)
model.save('./cats_and_dogs_classification2.h5')



#validation_stepsは、評価用のバッチをいくつ取り出すか決める。

#検証用は1000枚なので、1バッチ20枚とすると、50ステップ指定する。


#正解率の可視化

#import matplotlib.pyplot as plt
#%matplotlib inline

#acc = history.history["acc"]
#val_acc = history.history["val_acc"]
#loss = history.history["loss"]
#val_loss = history.history["val_loss"]

#epochs = range(1,len(acc) + 1)

#plt.plot(epochs, acc,"bo",label="Training Acc")
#plt.plot(epochs, val_acc,"b",label="Validation Acc")
#plt.legend()

#plt.figure()

#plt.plot(epochs,loss,"bo",label="Training Loss")
#plt.plot(epochs,val_loss,"b",label="Validation Loss")
#plt.legend()

#plt.show()