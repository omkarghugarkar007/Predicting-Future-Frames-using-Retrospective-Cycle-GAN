from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Concatenate, LeakyReLU, Input, Activation, ReLU
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
weight_initialzer = RandomNormal(stddev=0.02)

img_width = 256
img_height = 256
num_channels = 3
num_stacked_imgs = 4

def residual_block(filters,previous, input):

    for i in range(9):
        x = Conv2D(filters,3,strides=1, padding='same', kernel_initializer=weight_initialzer)(input)
        x = InstanceNormalization(axis=-1)(x)
        x = ReLU()(x)
        x = Conv2D(filters,3,strides=1, padding='same', kernel_initializer=weight_initialzer)(x)
        x = InstanceNormalization(axis=-1)(x)

        if previous is not None:
            x_out = Concatenate()([previous, x])
            previous = x_out
            input = x_out

        else:
            previous = x
            input = x

    return x_out

def generator():

    x1 = Input(shape = (img_width,img_height,num_channels*num_stacked_imgs))

    x2 = Conv2D(128,kernel_size=7,strides=1,padding='same',kernel_initializer=weight_initialzer)(x1)

    x3 = Conv2D(128,kernel_size=3,strides=2,padding='same',kernel_initializer=weight_initialzer)(x2)
    x3 = InstanceNormalization(axis=-1)(x3)
    x3 = ReLU()(x3)

    x4 = Conv2D(256,kernel_size=3,strides=2,padding='same',kernel_initializer=weight_initialzer)(x3)
    x4 = InstanceNormalization(axis=-1)(x4)
    x4 = ReLU()(x4)

    x13 = residual_block(256,None,x4)

    x14 = Conv2DTranspose(128,kernel_size=3,strides=2, padding='same', kernel_initializer=weight_initialzer)(x13)
    x14 = InstanceNormalization(axis=-1)(x14)
    x14 = ReLU()(x14)

    x15 = Conv2DTranspose(256,kernel_size=3,strides=2,padding='same',kernel_initializer=weight_initialzer)(x14)
    x15 = InstanceNormalization(axis=-1)(x15)
    x15 = ReLU()(x15)

    x16 = Conv2D(3,kernel_size=7,strides=1, padding='same',kernel_initializer=weight_initialzer,activation='tanh')(x15)

    return Model(x1,x16)

def discriminator_block(filters,strides, input):

    x = Conv2D(filters,kernel_size=4, strides=strides,padding='same',kernel_initializer=weight_initialzer)(input)
    x= InstanceNormalization(axis=-1)(x)
    x = LeakyReLU(0.2)(x)

    return x



def discriminator(num_input_images):

    x1 = Input(shape=(img_width,img_height,num_channels*num_input_images))

    x2 = Conv2D(64,kernel_size=4,strides=2,padding='same',kernel_initializer=weight_initialzer)(x1)
    x2 = LeakyReLU(0.2)(x2)

    x3 = discriminator_block(128,2,x2)

    x4 = discriminator_block(256,2,x3)

    x5 = discriminator_block(512,1,x4)

    x6 = Conv2D(1,kernel_size=4,strides=1,padding='same',kernel_initializer=weight_initialzer)(x5)

    return Model(x1,x6)

future_generator = generator()
future_generator.summary()

frame_discriminator = discriminator(1)
frame_discriminator.summary()

sequence_discriminator = discriminator(5)
sequence_discriminator.summary()