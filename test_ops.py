from ops import *


def test_G_Resblock(name, inputs, nums_out, is_training, splited_z, y1, y2, nums_class, alpha):
    with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = test_conditional_batchnorm(inputs, is_training, "bn1", splited_z, y1, y2, nums_class, alpha)
        inputs = relu(inputs)
        inputs = upsampling(inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, is_sn=True)
        inputs = test_conditional_batchnorm(inputs, is_training, "bn2", splited_z, y1, y2, nums_class, alpha)
        inputs = relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, is_sn=True)
        #Identity mapping
        temp = upsampling(temp)
        temp = conv("identity", temp, nums_out, 1, 1, is_sn=True)
    return inputs + temp

def test_conditional_batchnorm(x, train_phase, scope_bn, splited_z=None, y1=None, y2=None, nums_class=10, alpha=None):
    # Batch Normalization
    # Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    with tf.variable_scope(scope_bn):
        if y1 == None:
            beta = tf.get_variable(name=scope_bn + 'beta', shape=[x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
        else:
            y1 = tf.one_hot(y1, nums_class)
            y2 = tf.one_hot(y2, nums_class)
            y = y1 * alpha + y2 * (1 - alpha)
            z = tf.concat([splited_z, y], axis=1)
            gamma = dense("gamma", z, x.shape[-1], None, True)
            beta = dense("beta", z, x.shape[-1], None, True)
            gamma = tf.reshape(gamma, [-1, 1, 1, x.shape[-1]])
            beta = tf.reshape(beta, [-1, 1, 1, x.shape[-1]])
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments', keep_dims=True)
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed