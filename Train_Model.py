import random
import matplotlib.pyplot as plt
from math import log10, sqrt
import numpy as np
from Train import  checkpoint, checkpoint_dirs, checkpoint_prefix, epochs, step
import tensorflow as tf
from Future_Model import future_generator, frame_discriminator, sequence_discriminator
import time
import os

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dirs))
def PSNR(expected,Predicted):
    mse = np.mean((expected - Predicted) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return mse,psnr

def generate_images(input_array):

  first = random.randint(1,12000)
  input1 =  np.load("/content/Future_data/{}.npy".format(first))
  input2 =  np.load("/content/Future_data/{}.npy".format(first + 1))
  input3 =  np.load("/content/Future_data/{}.npy".format(first+2))
  input4 =  np.load("/content/Future_data/{}.npy".format(first + 3))
  expected = np.load("/content/Future_data/{}.npy".format(first + 4))

  expected = expected/255
  input_to = np.concatenate([input1,input2,input3,input4],axis= -1)
  input_to = tf.expand_dims(input_to,axis=0)

  out1 = future_generator(input_to)
  out = tf.squeeze(out1,axis = 0)
  out = (out + 1.0)/2
  out.numpy()

  plt.figure(figsize=(15,15))
  title = ['Ground Truth','Predicted']
  display_list = [expected,out]
  for i in range(2):

        plt.subplot(1,2,i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')

  plt.show()
  mse,psnr = PSNR(expected*255,out*255)
  print("MSE value is {}".format(mse))
  print("PNSR value is {}".format(psnr))
  ssim = tf.image.ssim(expected, out, max_val=1, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
  print("SSIM value is {}".format(ssim))

for epoch in range(epochs):
    print('Epochs:{}'.format(epoch))
    start = time.time()
    for _,_,files in os.walk("/content/Future_data"):
      random.shuffle(files)
    #Each Batch
    for k,file in enumerate(files):
        m = int(file.split('.')[0])
        # print(m)
        if m <12850:
          # print("Loopy")
          xm = np.load("/content/Future_data/{}.npy".format(m))
          xm_plus1 = np.load("/content/Future_data/{}.npy".format(m+1))
          xm_plus2 = np.load("/content/Future_data/{}.npy".format(m+2))
          xn = np.load("/content/Future_data/{}.npy".format(m+3))
          xn_plus1 = np.load("/content/Future_data/{}.npy".format(m+4))
          xm_to_xnplus1 = np.concatenate([xm,xm_plus1,xm_plus2,xn,xn_plus1], axis = -1)
          # print("before step")
          step(xm_to_xnplus1)
          # print("After step")
          if (k+1)%100 == 0:
              print(k)

    generate_images(xm_to_xnplus1)
    print('Time Taken: {}'.format(time.time()-start))

    checkpoint.save(file_prefix=checkpoint_prefix)