from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import os
import math
from Future_Model import future_generator, frame_discriminator, sequence_discriminator

lambda1 = 0.005
lambda2 = 0.003
lambda3 = 0.003

epochs = 1000

beta1 = 0.5
beta2 = 0.999

future_generator_optimizer = Adam(lr=0.0003, beta_1=beta1, beta_2=beta2)
frame_discriminator_optimizer = sequence_discriminator_optimizer = Adam(lr=0.0003, beta_1=beta1, beta_2=beta2)
loss = BinaryCrossentropy(from_logits=True)

checkpoint_dirs = '/content/drive/MyDrive/Future_Model'
checkpoint_prefix = os.path.join(checkpoint_dirs, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=future_generator_optimizer,
                                 frame_discriminator_optimizer=frame_discriminator_optimizer,
                                 sequence_discriminator_optimizer=sequence_discriminator_optimizer,
                                 generator=future_generator, frame_discriminator=frame_discriminator,
                                 sequence_discriminator=sequence_discriminator)


def LoG(image, filtersize=7, sigma=1):
    image = (image + 1) * 127.5
    image = tf.squeeze(image, axis=0)
    image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    image = tf.squeeze(image, 2)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    n_channels = 1
    image = tf.expand_dims(image, 2)

    w = math.ceil(sigma * filtersize)
    w_range = int(math.floor(w / 2))

    y = x = tf.range(-w_range, w_range + 1, 1)
    Y, X = tf.meshgrid(x, y)
    z = tf.cast(tf.add(tf.square(X), tf.square(Y)), tf.float32)
    nom = tf.subtract(z, 2 * (sigma ** 2))
    denom = 2 * math.pi * (sigma ** 6)
    exp = tf.exp(-z / 2 * (sigma ** 2))
    fil = tf.divide(tf.multiply(nom, exp), denom)

    fil = tf.stack([fil] * n_channels, axis=2)
    fil = tf.expand_dims(fil, 3)

    new = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    new = tf.expand_dims(new, 0)
    res = tf.nn.depthwise_conv2d(new, fil, strides=[1, 1, 1, 1], padding="SAME")
    res = tf.squeeze(res, 0)
    res = tf.squeeze(res, 2)
    minM = tf.math.reduce_min(res)
    maxM = tf.math.reduce_max(res)
    output = (res - minM) * 255 / (maxM - minM)
    output = tf.expand_dims(output, axis=0)
    output = output / 127.5 - 1
    return output


def discriminator_loss(fake, real):
    return (loss(tf.ones_like(real), real) + loss(tf.zeros_like(fake), fake)) * 0.5


def image_similarity(img1, img2):
    return tf.reduce_mean(tf.abs(img1 - img2))


def LoG_image_similarity(img1, img2):
    return tf.reduce_mean(tf.abs(LoG(img1) - LoG(img2)))


@tf.function
def step(xm_to_xnplus1):
    xm_to_xn = xm_to_xnplus1[:, :, 0:12]
    xm_to_xn = xm_to_xn / 255
    xm_to_xn = tf.expand_dims(xm_to_xn, axis=0)

    xn_plus_1 = xm_to_xnplus1[:, :, 12:15]
    xn_plus_1 = xn_plus_1 / 127.5 - 1
    xn_plus_1 = tf.expand_dims(xn_plus_1, axis=0)

    xm_plus_1_to_xn_plus_1 = xm_to_xnplus1[:, :, 3:]
    xm_plus_1_to_xn_plus_1 = xm_plus_1_to_xn_plus_1 / 255
    xm_plus_1_to_xn_plus_1 = tf.expand_dims(xm_plus_1_to_xn_plus_1, axis=0)

    xm = xm_to_xnplus1[:, :, 0:3]
    xm = xm / 127.5 - 1
    xm = tf.expand_dims(xm, axis=0)

    with tf.GradientTape(persistent=True) as tape:
        xn_plus_1_dash = future_generator(xm_to_xn, training=True)
        xm_dash = future_generator(xm_plus_1_to_xn_plus_1[::-1], training=True)

        xm_to_xn_false = tf.concat((xm_dash, xm_to_xn[:, :, :, 3:12]), axis=-1)

        xm_plus_1_to_xn_plus_1_false = tf.concat((xm_plus_1_to_xn_plus_1[:, :, :, 0:9], xn_plus_1_dash), axis=-1)

        xn_plus_1_double_dash = future_generator(xm_to_xn_false, training=True)
        xm_double_dash = future_generator(xm_plus_1_to_xn_plus_1_false[::-1])

        frame_discriminator_loss = lambda2 * (discriminator_loss(frame_discriminator(xn_plus_1_dash, training=True),
                                                                 frame_discriminator(xn_plus_1,
                                                                                     training=True)) + discriminator_loss(
            frame_discriminator(xn_plus_1_double_dash, training=True),
            frame_discriminator(xn_plus_1, training=True)) + discriminator_loss(
            frame_discriminator(xm_dash, training=True), frame_discriminator(xm, training=True)) + discriminator_loss(
            frame_discriminator(xm_double_dash, training=True), frame_discriminator(xm, training=True)))

        with tape.stop_recording():
            frame_discriminator_gradients = tape.gradient(frame_discriminator_loss,
                                                          frame_discriminator.trainable_variables)
            frame_discriminator_optimizer.apply_gradients(
                zip(frame_discriminator_gradients, frame_discriminator.trainable_variables))

        xm_to_xnplus1 = tf.expand_dims(xm_to_xnplus1, axis=0) / 127.5 - 1

        xm_to_xn_with_xn_plus_1_dash = tf.concat((xm_to_xnplus1[:, :, :, 0:12], xn_plus_1_dash), axis=-1)

        xm_to_xn_with_xn_plus_1_double_dash = tf.concat((xm_to_xnplus1[:, :, :, 0:12], xn_plus_1_double_dash), axis=-1)

        xm_to_xn_with_xm_dash = tf.concat((xm_dash, xm_to_xnplus1[:, :, :, 3:]), axis=-1)

        xm_to_xn_with_xm_double_dash = tf.concat((xm_double_dash, xm_to_xnplus1[:, :, :, 3:]), axis=-1)

        sequence_discriminator_loss = lambda3 * (
                    discriminator_loss(sequence_discriminator(xm_to_xn_with_xm_dash, training=True),
                                       sequence_discriminator(xm_to_xnplus1, training=True)) + discriminator_loss(
                sequence_discriminator(xm_to_xn_with_xn_plus_1_double_dash, training=True),
                sequence_discriminator(xm_to_xnplus1, training=True)) + discriminator_loss(
                sequence_discriminator(xm_to_xn_with_xm_dash[::-1], training=True),
                sequence_discriminator(xm_to_xnplus1[::-1], training=True)) + discriminator_loss(
                sequence_discriminator(xm_to_xn_with_xm_double_dash[::-1], training=True),
                sequence_discriminator(xm_to_xnplus1[::-1], training=True)))

        with tape.stop_recording():
            sequence_discriminator_gradients = tape.gradient(sequence_discriminator_loss,
                                                             sequence_discriminator.trainable_variables)
            sequence_discriminator_optimizer.apply_gradients(
                zip(sequence_discriminator_gradients, sequence_discriminator.trainable_variables))

        img_loss = image_similarity(xm, xm_dash) + image_similarity(xm, xm_double_dash) + image_similarity(xm_dash,
                                                                                                           xm_double_dash) + image_similarity(
            xn_plus_1, xn_plus_1_dash) + image_similarity(xn_plus_1, xn_plus_1_double_dash) + image_similarity(
            xn_plus_1_dash, xn_plus_1_double_dash)
        LoG_loss = LoG_image_similarity(xm, xm_dash) + LoG_image_similarity(xm, xm_double_dash) + LoG_image_similarity(
            xm_dash, xm_double_dash) + LoG_image_similarity(xn_plus_1, xn_plus_1_dash) + LoG_image_similarity(xn_plus_1,
                                                                                                              xn_plus_1_double_dash) + LoG_image_similarity(
            xn_plus_1_dash, xn_plus_1_double_dash)

        future_generator_loss = img_loss + lambda1 * LoG_loss

        with tape.stop_recording():
            future_generator_gradients = tape.gradient(future_generator_loss, future_generator.trainable_variables)
            future_generator_optimizer.apply_gradients(
                zip(future_generator_gradients, future_generator.trainable_variables))

