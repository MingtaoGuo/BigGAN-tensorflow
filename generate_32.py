from networks_32 import Generator
import tensorflow as tf
import numpy as np
from PIL import Image
from utils import truncated_noise_sample
import os

NUMS_GEN = 64
NUMS_CLASS = 10
BATCH_SIZE = 64
Z_DIM = 128
IMG_H = 32
IMG_W = 32

def generate():
    if not os.path.exists("./generate"):
        os.mkdir("./generate")
    train_phase = tf.placeholder(tf.bool)
    z = tf.placeholder(tf.float32, [BATCH_SIZE, Z_DIM])
    y = tf.placeholder(tf.int32, [None])
    G = Generator("generator")
    fake_img = G(z, train_phase, y, NUMS_CLASS)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator"))
    saver.restore(sess, "./save_para/.\\model.ckpt")
    Z = truncated_noise_sample(BATCH_SIZE, Z_DIM)
    for nums_c in range(NUMS_CLASS):
        Z = truncated_noise_sample(BATCH_SIZE, Z_DIM)
        FAKE_IMG = sess.run(fake_img, feed_dict={z: Z, train_phase: False, y: nums_c * np.ones([NUMS_GEN])})
        concat_img = np.zeros([8*IMG_H, 8*IMG_W, 3])
        c = 0
        for i in range(8):
            for j in range(8):
                concat_img[i*IMG_H:i*IMG_H+IMG_H, j*IMG_W:j*IMG_W+IMG_W] = FAKE_IMG[c]
                c += 1
        Image.fromarray(np.uint8((concat_img + 1) * 127.5)).save("./generate/"+str(nums_c)+".jpg")

if __name__ == "__main__":
    generate()
