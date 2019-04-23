from networks_32 import Generator
from test_networks_32 import test_Generator
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

def Consecutive_category_morphing():
    if not os.path.exists("./generate"):
        os.mkdir("./generate")
    train_phase = tf.placeholder(tf.bool)
    z = tf.placeholder(tf.float32, [BATCH_SIZE, Z_DIM])
    y1 = tf.placeholder(tf.int32, [None])
    y2 = tf.placeholder(tf.int32, [None])
    alpha = tf.placeholder(tf.float32, [None, 1])
    G = test_Generator("generator")
    fake_img = G(z, train_phase, y1, y2, NUMS_CLASS, alpha)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "generator"))
    saver.restore(sess, "./save_para/.\\model.ckpt")
    Z = truncated_noise_sample(BATCH_SIZE, Z_DIM)
    CLASS1 = 2
    CLASS2 = 3
    count = 0
    for ALPHA in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ALPHA = np.ones([BATCH_SIZE, 1]) * ALPHA
        FAKE_IMG = sess.run(fake_img, feed_dict={z: Z, y1: CLASS1 * np.ones([BATCH_SIZE]), y2: CLASS2 * np.ones([BATCH_SIZE]), train_phase: False, alpha: ALPHA})
        concat_img = np.zeros([8 * IMG_H, 8 * IMG_W, 3])
        c = 0
        for i in range(8):
            for j in range(8):
                concat_img[i * IMG_H:i * IMG_H + IMG_H, j * IMG_W:j * IMG_W + IMG_W] = FAKE_IMG[c]
                c += 1
        Image.fromarray(np.uint8((concat_img + 1) * 127.5)).save("./generate/" + str(count) + ".jpg")
        count += 1


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
    Consecutive_category_morphing()
