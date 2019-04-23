import tensorflow as tf
import tensorflow.contrib as contrib


def conditional_batchnorm(x, train_phase, scope_bn, splited_z=None, y=None, nums_class=10):
    #Batch Normalization
    #Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    with tf.variable_scope(scope_bn):
        if y == None:
            beta = tf.get_variable(name=scope_bn + 'beta', shape=[x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
        else:
            y = tf.one_hot(y, nums_class)
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

def non_local(name, inputs, update_collection, is_sn):
    H = inputs.shape[1]
    W = inputs.shape[2]
    C = inputs.shape[3]
    C_ = C // 8
    inputs_ = tf.transpose(inputs, perm=[0, 3, 1, 2])
    inputs_ = tf.reshape(inputs_, [-1, C, H * W])
    with tf.variable_scope(name):
        f = conv("f", inputs, C_, 1, 1, update_collection, is_sn)
        g = conv("g", inputs, C_, 1, 1, update_collection, is_sn)
        h = conv("h", inputs, C, 1, 1, update_collection, is_sn)
        f = tf.transpose(f, [0, 3, 1, 2])
        f = tf.reshape(f, [-1, C_, H * W])
        g = tf.transpose(g, [0, 3, 1, 2])
        g = tf.reshape(g, [-1, C_, H * W])
        h = tf.transpose(h, [0, 3, 1, 2])
        h = tf.reshape(h, [-1, C, H * W])
        s = tf.matmul(f, g, transpose_a=True)
        beta = tf.nn.softmax(s, dim=0)
        o = tf.matmul(h, beta)
        gamma = tf.get_variable("gamma", [], initializer=tf.constant_initializer(0.))
        y = gamma * o + inputs_
        y = tf.reshape(y, [-1, C, H, W])
        y = tf.transpose(y, perm=[0, 2, 3, 1])
    return y

def conv(name, inputs, nums_out, k_size, strides, update_collection=None, is_sn=False):
    nums_in = inputs.shape[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [k_size, k_size, nums_in, nums_out], initializer=tf.orthogonal_initializer())
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.0]))
        if is_sn:
            W = spectral_normalization("sn", W, update_collection=update_collection)
        con = tf.nn.conv2d(inputs, W, [1, strides, strides, 1], "SAME")
    return tf.nn.bias_add(con, b)

def upsampling(inputs):
    H = inputs.shape[1]
    W = inputs.shape[2]
    return tf.image.resize_nearest_neighbor(inputs, [H * 2, W * 2])

def downsampling(inputs):
    return tf.nn.avg_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

def relu(inputs):
    return tf.nn.relu(inputs)

def global_sum_pooling(inputs):
    inputs = tf.reduce_sum(inputs, [1, 2], keep_dims=False)
    return inputs

def Hinge_loss(real_logits, fake_logits):
    D_loss = -tf.reduce_mean(tf.minimum(0., -1.0 + real_logits)) - tf.reduce_mean(tf.minimum(0., -1.0 - fake_logits))
    G_loss = -tf.reduce_mean(fake_logits)
    return D_loss, G_loss

def ortho_reg(vars_list):
    s = 0
    for var in vars_list:
        if "W" in var.name:
            if var.shape.__len__() == 4:
                nums_kernel = int(var.shape[-1])
                W = tf.transpose(var, perm=[3, 0, 1, 2])
                W = tf.reshape(W, [nums_kernel, -1])
                ones = tf.ones([nums_kernel, nums_kernel])
                eyes = tf.eye(nums_kernel, nums_kernel)
                y = tf.matmul(W, W, transpose_b=True) * (ones - eyes)
                s += tf.nn.l2_loss(y)
    return s

def dense(name, inputs, nums_out, update_collection=None, is_sn=False):
    nums_in = inputs.shape[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [nums_in, nums_out], initializer=tf.orthogonal_initializer())
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.0]))
        if is_sn:
            W = spectral_normalization("sn", W, update_collection=update_collection)
    return tf.nn.bias_add(tf.matmul(inputs, W), b)

def Inner_product(global_pooled, y, nums_class, update_collection=None):
    W = global_pooled.shape[-1]
    V = tf.get_variable("V", [nums_class, W], initializer=tf.orthogonal_initializer())
    V = tf.transpose(V)
    V = spectral_normalization("embed", V, update_collection=update_collection)
    V = tf.transpose(V)
    temp = tf.nn.embedding_lookup(V, y)
    temp = tf.reduce_sum(temp * global_pooled, axis=1, keep_dims=True)
    return temp

def G_Resblock(name, inputs, nums_out, is_training, splited_z, y, nums_class):
    with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = conditional_batchnorm(inputs, is_training, "bn1", splited_z, y, nums_class)
        inputs = relu(inputs)
        inputs = upsampling(inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, is_sn=True)
        inputs = conditional_batchnorm(inputs, is_training, "bn2", splited_z, y, nums_class)
        inputs = relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, is_sn=True)
        #Identity mapping
        temp = upsampling(temp)
        temp = conv("identity", temp, nums_out, 1, 1, is_sn=True)
    return inputs + temp

def D_Resblock(name, inputs, nums_out, update_collection=None, is_down=True):
    with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = relu(inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, update_collection, is_sn=True)
        inputs = relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, update_collection, is_sn=True)
        if is_down:
            inputs = downsampling(inputs)
            #Identity mapping
            temp = conv("identity", temp, nums_out, 1, 1, update_collection, is_sn=True)
            temp = downsampling(temp)
        # else:
        #     temp = conv("identity", temp, nums_out, 1, 1, update_collection, is_sn=True)
    return inputs + temp

def D_FirstResblock(name, inputs, nums_out, update_collection, is_down=True):
    with tf.variable_scope(name):
        temp = tf.identity(inputs)
        inputs = conv("conv1", inputs, nums_out, 3, 1, update_collection=update_collection, is_sn=True)
        inputs = relu(inputs)
        inputs = conv("conv2", inputs, nums_out, 3, 1, update_collection=update_collection, is_sn=True)
        if is_down:
            inputs = downsampling(inputs)
            #Identity mapping
            temp = downsampling(temp)
            temp = conv("identity", temp, nums_out, 1, 1, update_collection=update_collection, is_sn=True)
    return inputs + temp



def _l2normalize(v, eps=1e-12):
  """l2 normize the input vector."""
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_normalization(name, weights, num_iters=1, update_collection=None,
                           with_sigma=False):
  w_shape = weights.shape.as_list()
  w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
  u = tf.get_variable(name + 'u', [1, w_shape[-1]],
                      initializer=tf.truncated_normal_initializer(),
                      trainable=False)
  u_ = u
  for _ in range(num_iters):
    v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
    u_ = _l2normalize(tf.matmul(v_, w_mat))

  sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
  w_mat /= sigma
  if update_collection is None:
    with tf.control_dependencies([u.assign(u_)]):
      w_bar = tf.reshape(w_mat, w_shape)
  else:
    w_bar = tf.reshape(w_mat, w_shape)
    if update_collection != 'NO_OPS':
      tf.add_to_collection(update_collection, u.assign(u_))
  if with_sigma:
    return w_bar, sigma
  else:
    return w_bar

# inputs = tf.placeholder(tf.float32, [None, 32, 32, 128])
# non_local("non_local", inputs, None, True)