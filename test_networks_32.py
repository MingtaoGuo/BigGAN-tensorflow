from test_ops import *



class test_Generator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, train_phase, y1, y2, nums_class, alpha):
        z_dim = int(inputs.shape[-1])
        nums_layer = 3
        remain = z_dim % 3
        chunk_size = (z_dim - remain) // nums_layer
        z_split = tf.split(inputs, [chunk_size] * (nums_layer - 1) + [chunk_size + remain], axis=1)
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            inputs = dense("dense", inputs, 256*4*4)
            inputs = tf.reshape(inputs, [-1, 4, 4, 256])
            inputs = test_G_Resblock("ResBlock1", inputs, 256, train_phase, z_split[0], y1, y2, nums_class, alpha)
            inputs = test_G_Resblock("ResBlock2", inputs, 256, train_phase, z_split[1], y1, y2, nums_class, alpha)
            inputs = non_local("Non-local", inputs, None, True)
            inputs = test_G_Resblock("ResBlock3", inputs, 256, train_phase, z_split[2], y1, y2, nums_class, alpha)
            inputs = relu(conditional_batchnorm(inputs, train_phase, "BN"))
            inputs = conv("conv", inputs, k_size=3, nums_out=3, strides=1, is_sn=True)
        return tf.nn.tanh(inputs)

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


