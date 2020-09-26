import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os

class Generator:
    def __init__(self, batch_size, dir_, master_dir, shape):
        self.batch_size = batch_size
        self.dir = dir_
        self.images = os.listdir(dir_)[0:10001]
        self.master_dir = master_dir
        self.shape = shape
        
    def gen(self, batch_num):
        X = np.array([
            np.array(Image.open(os.path.join(self.dir, img)).resize(self.shape)).astype(np.float32) for img in self.images[self.batch_size*batch_num:self.batch_size*(batch_num+1)]
        ])
        Y = np.array([
            np.array(Image.open(os.path.join(self.master_dir, "COCO_train2014_{}".format(img))).resize(self.shape)).astype(np.float32) for img in self.images[self.batch_size*batch_num:self.batch_size*(batch_num+1)]
        ])
        X /= 255.0
        Y /= 255.0
        return X, Y
    
    def generate(self):
        counter = 0
        while True:
            try:
                yield self.gen(counter)
            except:
                counter = 0
                yield self.gen(counter)
                
            counter += 1

einp = tf.keras.layers.Input(shape=(256, 256, 3))
econv = tf.keras.layers.Conv2D(256, (2, 2), padding='same')(einp)
econv = tf.keras.layers.Conv2D(256, (2, 2), padding='same')(econv)

sizes = [128, 64, 32]
for size in sizes:
    econv = tf.keras.layers.Conv2D(size, (2, 2), activation='relu', padding='same')(econv)
    econv = tf.keras.layers.Conv2D(size, (2, 2), activation='relu', padding='same')(econv)
    
encoder = tf.keras.models.Model(einp, econv)
dinp = tf.keras.layers.Input(shape=(256, 256, 32))
dconv = tf.keras.layers.Conv2DTranspose(32, (2, 2), padding='same', activation='relu')(dinp)
dconv = tf.keras.layers.Conv2DTranspose(32, (2, 2), padding='same', activation='relu')(dconv)

sizes = [64, 128, 256]
for size in sizes:
    dconv = tf.keras.layers.Conv2DTranspose(size, (2, 2), activation='relu', padding='same')(dconv)
    dconv = tf.keras.layers.Conv2DTranspose(size, (2, 2), activation='relu', padding='same')(dconv)
    
dconv = tf.keras.layers.Conv2DTranspose(3, (2, 2), activation='relu', padding='same')(dconv)

decoder = tf.keras.models.Model(dinp, dconv)
model = tf.keras.models.Sequential([
    encoder,
    decoder
])

generator = Generator(
    32,
    "/storage/BlurredCOCO/train",
    "/datasets/coco/coco_train2014",
    (128, 128)
)

model.compile(optimizer='adam', loss='mse')
checkpoint = tf.keras.callbacks.ModelCheckpoint("weights", verbose=1)

#steps = len(os.listdir("/storage/BlurredCOCO/train"))//32
steps = 10000 // 32
model.fit(generator.generate(), epochs=10, steps_per_epoch=steps, callbacks=[checkpoint])

model.save("/storage/DeblurringCOCOModel.h5")