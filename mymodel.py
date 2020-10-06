import tensorflow as tf 
import tensorflow.keras as keras 

# make sure that the tensorflow version should be 2 and version 1 not work


# we use the auto-encoder-decoder as the compression and decompression progress
# in the  follwing  we define the model's structure

class Mymodel2(keras.Model):
# compression ratio = 2
    def __init__(self):
        super(Mymodel2, self).__init__()
        ## define the encoder
        self.encoder = keras.Sequential([
        # convolutional layer
            keras.layers.Conv2D(filters=64, kernel_size=5, padding='SAME',activation='relu'),
        # max pooling makes image shape smaller
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=32, kernel_size=3, padding='SAME', activation='relu'),
            keras.layers.Conv2D(filters=6, kernel_size=3, padding='SAME', activation='relu'),
        ])
        # define the decoder
        self.decoder = keras.Sequential([
            keras.layers.Conv2DTranspose(32 , 3, padding='SAME' ,activation='relu'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2DTranspose(64, 3, padding='SAME', activation='relu'),
            keras.layers.Conv2DTranspose(3, 5, padding='SAME', activation='relu'),
        ])
    def call(self, x):
        # input the image x, and get the compressed feature image y
        y = self.encoder(x)
        # decompresss the feature image and get the decompressed image 
        return self.decoder(y)

# others have the same structure

class Mymodel4(keras.Model):
#compression ratio = 4
    def __init__(self):
        super(Mymodel4, self).__init__()
        
        self.encoder = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=5, padding='SAME',activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=32, kernel_size=5, padding='SAME', activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=16, kernel_size=3, padding='SAME', activation='relu'),
            keras.layers.Conv2D(filters=12, kernel_size=3, padding='SAME', activation='relu'),
        ])
        self.decoder = keras.Sequential([
            keras.layers.Conv2DTranspose(16 , 3, padding='SAME' ,activation='relu'),
            keras.layers.Conv2DTranspose(32, 3, padding='SAME', activation='relu'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2DTranspose(64, 5, padding='SAME', activation='relu'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2DTranspose(3, 5, padding='SAME', activation='relu')
        ])
    def call(self, x):
        y = self.encoder(x)
        return self.decoder(y)

class Mymodel8(keras.Model):
#compression ratio = 8
    def __init__(self):
        super(Mymodel8, self).__init__()
        
        self.encoder = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=5, padding='SAME', activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=32, kernel_size=5, padding='SAME', activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=16, kernel_size=3, padding='SAME', activation='relu'),
            keras.layers.Conv2D(filters=6, kernel_size=3, padding='SAME', activation='relu'),
        ])
        self.decoder = keras.Sequential([
            keras.layers.Conv2DTranspose(16 , 3, padding='SAME' ,activation='relu'),
            keras.layers.Conv2DTranspose(32, 3, padding='SAME', activation='relu'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2DTranspose(64, 5, padding='SAME', activation='relu'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2DTranspose(3, 5, padding='SAME', activation='relu')
        ])
    def call(self, x):
        y = self.encoder(x)
        return self.decoder(y)

class Mymodel16(keras.Model):
#compression ratio = 16
    def __init__(self):
        super(Mymodel16, self).__init__()
        
        self.encoder = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=5, padding='SAME', activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=32, kernel_size=5, padding='SAME', activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=32, kernel_size=5, padding='SAME', activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=16, kernel_size=3, padding='SAME', activation='relu'),
            keras.layers.Conv2D(filters=12, kernel_size=3, padding='SAME', activation='relu'),
        ])
        self.decoder = keras.Sequential([
            keras.layers.Conv2DTranspose(16 , 3, padding='SAME' ,activation='relu'),
            keras.layers.Conv2DTranspose(32, 3, padding='SAME', activation='relu'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2DTranspose(32, 5, padding='SAME', activation='relu'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2DTranspose(64, 5, padding='SAME', activation='relu'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2DTranspose(3, 5, padding='SAME', activation='relu')
        ])
    def call(self, x):
        y = self.encoder(x)
        return self.decoder(y)

class Mymodel32(keras.Model):
#compression ratio = 32
    def __init__(self):
        super(Mymodel32, self).__init__()
         
        self.encoder = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=5, padding='SAME', activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=32, kernel_size=5, padding='SAME', activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=32, kernel_size=5, padding='SAME', activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=16, kernel_size=3, padding='SAME', activation='relu'),
            keras.layers.Conv2D(filters=6, kernel_size=3, padding='SAME', activation='relu'),
        ])
        self.decoder = keras.Sequential([
            keras.layers.Conv2DTranspose(16 , 3, padding='SAME' ,activation='relu'),
            keras.layers.Conv2DTranspose(32, 3, padding='SAME', activation='relu'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2DTranspose(32, 5, padding='SAME', activation='relu'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2DTranspose(64, 5, padding='SAME', activation='relu'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2DTranspose(3, 5, padding='SAME', activation='relu')
        ])
    def call(self, x):
        y = self.encoder(x)
        return self.decoder(y)

