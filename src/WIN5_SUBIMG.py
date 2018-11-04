import tensorflow as tf
import numpy as np

# import cv2
# from const import *

# lamb = 0.0001
# init_lr = 0.0001
# use_bn = False
# bn_decay = 0.99
# epsilon = 0.001
# weight_init = 0.01
# K_POOL = 5
# INIT_RANGE = 0.05

def define_additional_variables():
    current_epoch=tf.Variable(-1)
    current_step=tf.Variable(-1)
    max_acc_tensor=tf.Variable(0.,dtype=tf.float32)
    tf.add_to_collection('current_epoch', current_epoch)
    tf.add_to_collection('current_step',current_step)
    tf.add_to_collection('max_acc_tensor', max_acc_tensor)
    return  current_epoch,current_step,max_acc_tensor
    
def weight_conv_variable(shape, name, mode="normal", init_range = 0.05):
    std = shape[0] * shape[1] * shape[2]
    std = np.sqrt(2. / std)
    if mode=="normal":
        initial = tf.truncated_normal(shape, stddev=init_range) 
    elif mode=="xavier" or mode=="glorot":
        initial = tf.truncated_normal(shape, stddev=weight_init, mean=0.0) ## need some changes
    else:
        print("ERROR in mode for initialization.")
    return tf.Variable(initial,name=name)

def weight_fc_variable(shape,name, mode="normal", init_range = 0.05):
    std = shape[0]
    std = np.sqrt(2. / std)
    initial = tf.truncated_normal(shape, stddev=init_range) #tf.truncated_normal(shape, stddev=weight_init, mean=0.0)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name, mode="constant", init_range = 0.05):
    initial = tf.constant(init_range, shape=shape) #tf.constant(0., shape=shape)
    return tf.Variable(initial,name=name)

# def flatten(x):
    # dim = x.get_shape().as_list()
    # dim = np.prod(dim[1:])
    # dime = dim
    # x = tf.reshape(x, [-1, dim])
    # return x, dime

def conv2d(x, W, stride, padding_type = 'SAME'):
    return tf.nn.conv2d(x, W, [1, stride, stride, 1], padding_type)

def max_pool(x, filter, stride, padding_type = 'SAME'):
    return tf.nn.max_pool(x, [1, filter, filter, 1], [1, stride, stride, 1], padding_type)

def batch_normalization(type,input,is_training,decay,variable_averages, epsilon=0.001):
    shape=np.shape(input)
    if type == 'conv':
        gamma=tf.Variable(tf.constant(1.,shape=[shape[3]]))
        beta=tf.Variable(tf.constant(0.,shape=[shape[3]]))
        batch_mean, batch_var = tf.nn.moments(input, [0,1,2])
        pop_mean= tf.Variable(tf.zeros([shape[3]],dtype=tf.float32),trainable=False)
        pop_var = tf.Variable(tf.ones([shape[3]], dtype=tf.float32), trainable=False)
        
    elif type == 'fc':
        gamma=tf.Variable(tf.constant(1.,shape=[shape[1]]))
        beta=tf.Variable(tf.constant(0.,shape=[shape[1]]))
        batch_mean,batch_var = tf.nn.moments(input, [0])
        pop_mean= tf.Variable(tf.zeros([shape[1]],dtype=tf.float32),trainable=False)
        pop_var = tf.Variable(tf.ones([shape[1]], dtype=tf.float32), trainable=False)


    def update_mean_var():
        update_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
        update_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([update_mean,update_var]):
                return tf.identity(batch_mean), tf.identity(batch_var)
    mean,var=tf.cond(is_training,update_mean_var,lambda: (pop_mean,pop_var))
    return tf.nn.batch_normalization(input,mean,var,beta,gamma,epsilon)

def input(nb_class, img_size, nb_channel, n_tuple):
    x = tf.placeholder(tf.float32, [None, n_tuple, img_size, img_size, nb_channel])
    y_ = tf.placeholder(tf.float32, [None, nb_class])

    tf.add_to_collection('x', x)
    tf.add_to_collection('y_', y_)
    return x, y_
    
def cnn_layers(x, img_size=64, nb_channel=1, use_bn=False, bn_decay = 0.99, epsilon=0.001):
    variable_averages = tf.train.ExponentialMovingAverage(bn_decay)
    is_training = tf.placeholder(dtype=tf.bool)

    x_image = tf.reshape(x, [-1, img_size, img_size, nb_channel]) # batch_size, w, h, num_channels

    with tf.variable_scope('conv1') as scope:
        W_conv1 = weight_conv_variable([5, 5, nb_channel, 32], 'W')
        b_conv1 = bias_variable([32], 'b')
        h_conv1 = conv2d(x_image, W_conv1, 1) + b_conv1
        if use_bn:
            h_conv1 = batch_normalization('conv', h_conv1, is_training, bn_decay, variable_averages, epsilon)
        h_conv1 = tf.nn.relu(h_conv1, name=scope.name)
    h_pool1 = max_pool(h_conv1, 2, 2)

    with tf.variable_scope('conv2') as scope:
        W_conv2 = weight_conv_variable([5, 5, 32, 64], 'W')
        b_conv2 = bias_variable([64], 'b')
        h_conv2 = conv2d(h_pool1, W_conv2, 1) + b_conv2
        if use_bn:
            h_conv2 = batch_normalization('conv', h_conv2, is_training, bn_decay, variable_averages,epsilon)
        h_conv2 = tf.nn.relu(h_conv2, name=scope.name)
    h_pool2 = max_pool(h_conv2, 2, 2)

    with tf.variable_scope('conv3') as scope:
        W_conv3 = weight_conv_variable([5, 5, 64, 256], 'W')
        b_conv3 = bias_variable([256], 'b')
        h_conv3 = conv2d(h_pool2, W_conv3, 1) + b_conv3
        if use_bn:
            h_conv3 = batch_normalization('conv', h_conv3, is_training, bn_decay, variable_averages,epsilon)
        h_conv3 = tf.nn.relu(h_conv3, name=scope.name)
    h_pool3 = max_pool(h_conv3, 2, 2)

    with tf.variable_scope('conv4') as scope:
        W_conv4 = weight_conv_variable([5, 5, 256, 1024], 'W')
        b_conv4 = bias_variable([1024], 'b')
        h_conv4 = conv2d(h_pool3, W_conv4, 1) + b_conv4
        if use_bn:
            h_conv4 = batch_normalization('conv', h_conv4, is_training, bn_decay, variable_averages,epsilon)
        h_conv4 = tf.nn.relu(h_conv4, name=scope.name)
    h_pool4 = max_pool(h_conv4, 2, 2)
    regul_loss_cnn_layers = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(W_conv4)
    
    tf.add_to_collection('is_training', is_training)
    
    return h_pool4, regul_loss_cnn_layers
    
def average_pool_inference(x, y_true, nb_class=100, img_size=64, nb_channel=1, n_tuple=20, use_bn=False, bn_decay=0.99, epsilon=0.001, l2_loss_lamb = 0.0001, kmax=0, logger=None):
    h_pool4, regul_loss_cnn_layers = cnn_layers(x, img_size, nb_channel, use_bn, bn_decay, epsilon)
    # print(tf.shape(h_pool4))
    
    y_local_feature = tf.reshape(h_pool4, [-1, n_tuple, 4, 4, 1024]) # make batch size becoming flat
    y_local_feature = tf.reshape(y_local_feature, [-1, n_tuple * 4 * 4, 1024]) # make batch size becoming flat
    y_global_feature= tf.reduce_mean(y_local_feature, axis=1) # average pool / accumulating aggregation
    
    # y_feature_group_tp = tf.transpose(h_pool4_img_flat, perm=[0, 2, 1])
    # y_feature_k_maxpool_tp, indexes = tf.nn.top_k(y_feature_group_tp, k=K_POOL, sorted=False)
    # y_feature_k_maxpool_tp_rev = tf.transpose(y_feature_k_maxpool_tp, perm=[0, 2, 1]) #transpose reverse
    # y_feature_k_maxpool_val = tf.reduce_mean(y_feature_k_maxpool_tp_rev, 1)

    # with tf.variable_scope('fc5') as scope:
        # W_fc5 = weight_fc_variable([1024, 1024], 'W')
        # b_fc5 = bias_variable([1024], 'b')
        # y_fc5 = tf.matmul(tf.contrib.layers.flatten(y_global_feature), W_fc5) + b_fc5
        # if use_bn:
            # y_fc5 = batch_normalization('fc', y_fc5, is_training, bn_decay, variable_averages)
            # y_fc5 = tf.nn.relu(y_fc5, name=scope.name)
        # keep_prob_fc = tf.placeholder(tf.float32)
        # y_fc5 = tf.nn.dropout(y_fc5, keep_prob_fc)

    '''Softmax Layer'''
    with tf.variable_scope('softmax') as scope:
        W_fc6 = weight_fc_variable([1024, nb_class], 'W')
        b_fc6 = bias_variable([nb_class], 'b')
        y_out = tf.matmul(y_global_feature, W_fc6) + b_fc6

    '''------------------------------ END OF MODEL -----------------------------------------'''


    # Define summary op

    loss_summary_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('loss', loss_summary_placeholder)
    acc_train_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('accuracy', acc_train_placeholder)
    summary_op = tf.summary.merge_all()

    # Triplet loss
    the = 50
    # h_fc1RS = tf.reshape(y_feature_k_maxpool, [2, -1, 1024])
    # fp1RS, fp2RS = tf.split(h_fc1RS, 2, 0)
    # fp1 = tf.reshape(fp1RS, [-1, 1024])
    # fp2 = tf.reshape(fp2RS, [-1, 1024])

    # y1RS = tf.reshape(y_true, [2, -1])
    # lp1RS, lp2RS = tf.split(y1RS, 2, 0)
    # lp1 = tf.reshape(lp1RS, [-1, nb_class])
    # lp1 = tf.arg_max(lp1,1)
    # lp2 = tf.reshape(lp2RS, [-1, nb_class])
    # lp2 = tf.arg_max(lp2,1)

    #mask = tf.cast(tf.equal(lp1, lp2), tf.float32)

    #distance = tf.sqrt(tf.reduce_sum((fp1 - fp2) * (fp1 - fp2), 1))

    #same_loss = 0.00 * tf.reduce_mean(mask * distance)
    #different_loss = 0.00 * tf.reduce_mean((1 - mask) * tf.maximum(0.,the - distance))

    #triplet_loss = same_loss + different_loss
    
    #triplet_loss = 0.005 * tf.reduce_mean(mask * distance)

    # L2-loss
    
    regul_loss = l2_loss_lamb * (regul_loss_cnn_layers + tf.nn.l2_loss(W_fc6)) # + tf.nn.l2_loss(W_fc5))

    # Add to collection
    tf.add_to_collection('y_out', y_out)
    tf.add_to_collection('y_global_feature',y_global_feature)
    # tf.add_to_collection('distance', distance)
    tf.add_to_collection('regul_loss', regul_loss)
    # tf.add_to_collection('triplet_loss', triplet_loss)
    # tf.add_to_collection('triplet_loss', same_loss)
    # tf.add_to_collection('triplet_loss', different_loss)
    tf.add_to_collection('loss_summary_placeholder', loss_summary_placeholder)
    tf.add_to_collection('acc_train_placeholder', acc_train_placeholder)
    tf.add_to_collection('summary_op', summary_op)
    # tf.add_to_collection('keep_prob_fc', keep_prob_fc)
    if logger!= None:
        logger.info("average_pool_inference was created.")
    return y_out
    
def kmax_pool_inference(x, y_true, nb_class=100, img_size=64, nb_channel=1, n_tuple=20, use_bn=False, bn_decay=0.99, epsilon=0.001, l2_loss_lamb = 0.0001, kmax=1, logger=None):
    h_pool4, regul_loss_cnn_layers = cnn_layers(x, img_size, nb_channel, use_bn, bn_decay, epsilon)
    # print(tf.shape(h_pool4))
    
    y_local_feature = tf.reshape(h_pool4, [-1, n_tuple, 4, 4, 1024]) # make batch size becoming flat
    y_local_feature = tf.reshape(y_local_feature, [-1, n_tuple * 4 * 4, 1024]) # make batch size becoming flat
    y_local_feature_tp = tf.transpose(y_local_feature, perm=[0, 2, 1])
    y_global_feature_kmax_pool, indexes = tf.nn.top_k(y_local_feature_tp, k=kmax, sorted=False)
    y_global_feature_kmax_pool_tp = tf.transpose(y_global_feature_kmax_pool, perm=[0, 2, 1]) #transpose reverse
    y_global_feature = tf.reduce_mean(y_global_feature_kmax_pool_tp, 1)
    
    # y_feature_group_tp = tf.transpose(h_pool4_img_flat, perm=[0, 2, 1])
    # y_feature_k_maxpool_tp, indexes = tf.nn.top_k(y_feature_group_tp, k=K_POOL, sorted=False)
    # y_feature_k_maxpool_tp_rev = tf.transpose(y_feature_k_maxpool_tp, perm=[0, 2, 1]) #transpose reverse
    # y_feature_k_maxpool_val = tf.reduce_mean(y_feature_k_maxpool_tp_rev, 1)

    # with tf.variable_scope('fc5') as scope:
        # W_fc5 = weight_fc_variable([1024, 1024], 'W')
        # b_fc5 = bias_variable([1024], 'b')
        # y_fc5 = tf.matmul(tf.contrib.layers.flatten(y_global_feature), W_fc5) + b_fc5
        # if use_bn:
            # y_fc5 = batch_normalization('fc', y_fc5, is_training, bn_decay, variable_averages)
            # y_fc5 = tf.nn.relu(y_fc5, name=scope.name)
        # keep_prob_fc = tf.placeholder(tf.float32)
        # y_fc5 = tf.nn.dropout(y_fc5, keep_prob_fc)

    '''Softmax Layer'''
    with tf.variable_scope('softmax') as scope:
        W_fc6 = weight_fc_variable([1024, nb_class], 'W')
        b_fc6 = bias_variable([nb_class], 'b')
        y_out = tf.matmul(y_global_feature, W_fc6) + b_fc6

    '''------------------------------ END OF MODEL -----------------------------------------'''

    # Define summary op
    loss_summary_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('loss', loss_summary_placeholder)
    acc_train_placeholder = tf.placeholder(tf.float32)
    tf.summary.scalar('accuracy', acc_train_placeholder)
    summary_op = tf.summary.merge_all()

    # Triplet loss
    the = 50
    # h_fc1RS = tf.reshape(y_feature_k_maxpool, [2, -1, 1024])
    # fp1RS, fp2RS = tf.split(h_fc1RS, 2, 0)
    # fp1 = tf.reshape(fp1RS, [-1, 1024])
    # fp2 = tf.reshape(fp2RS, [-1, 1024])

    # y1RS = tf.reshape(y_true, [2, -1])
    # lp1RS, lp2RS = tf.split(y1RS, 2, 0)
    # lp1 = tf.reshape(lp1RS, [-1, nb_class])
    # lp1 = tf.arg_max(lp1,1)
    # lp2 = tf.reshape(lp2RS, [-1, nb_class])
    # lp2 = tf.arg_max(lp2,1)

    #mask = tf.cast(tf.equal(lp1, lp2), tf.float32)

    #distance = tf.sqrt(tf.reduce_sum((fp1 - fp2) * (fp1 - fp2), 1))

    #same_loss = 0.00 * tf.reduce_mean(mask * distance)
    #different_loss = 0.00 * tf.reduce_mean((1 - mask) * tf.maximum(0.,the - distance))

    #triplet_loss = same_loss + different_loss
    
    #triplet_loss = 0.005 * tf.reduce_mean(mask * distance)

    # L2-loss
    
    regul_loss = l2_loss_lamb * (regul_loss_cnn_layers + tf.nn.l2_loss(W_fc6)) # + tf.nn.l2_loss(W_fc5))

    # Add to collection
    tf.add_to_collection('y_out', y_out)
    tf.add_to_collection('y_global_feature',y_global_feature)
    # tf.add_to_collection('distance', distance)
    tf.add_to_collection('regul_loss', regul_loss)
    # tf.add_to_collection('triplet_loss', triplet_loss)
    # tf.add_to_collection('triplet_loss', same_loss)
    # tf.add_to_collection('triplet_loss', different_loss)
    tf.add_to_collection('loss_summary_placeholder', loss_summary_placeholder)
    tf.add_to_collection('acc_train_placeholder', acc_train_placeholder)
    tf.add_to_collection('summary_op', summary_op)
    # tf.add_to_collection('keep_prob_fc', keep_prob_fc)
    if logger!= None:
        logger.info("kmax_pool_inference with kmax=%d was created."%kmax)
    return y_out
    
    
def loss(y_conv, y_, mode="cross_entropy"):
    regul_loss = tf.get_collection('regul_loss')[0]
    # triplet_loss = tf.get_collection('triplet_loss')[0]
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    if mode=="cross_entropy":
        total_loss = cross_entropy
        print("total_loss = cross_entropy")
    elif mode=="regular_loss":
        total_loss = cross_entropy + regul_loss
        print("total_loss = cross_entropy + regul_loss")
    elif mode=="triplet_loss":
        # total_loss = cross_entropy + triplet_loss
        print("ERROR total_loss = cross_entropy + triplet_loss")
    elif mode=="all":
        # total_loss = cross_entropy + regul_loss + triplet_loss
        print("ERROR total_loss = cross_entropy + regul_loss + triplet_loss")
    tf.add_to_collection('origin_loss', cross_entropy)
    tf.add_to_collection('total_loss', total_loss)
    return total_loss

def train_op(loss, global_step, lr=0.0001):
    # learning_rate = tf.train.exponential_decay(init_lr, global_step, 1000, 0.96, staircase = True)
    #train_step = tf.train.MomentumOptimizer(learning_rate=init_lr, momentum=0.99).minimize(loss, global_step=global_step)
    # learning_rate = 0.0001
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    grads_and_vars = optimizer.compute_gradients(loss)
    grads, variables = zip(*grads_and_vars)
    clipped_gradients, _ = (tf.clip_by_global_norm(grads, 1.))
    grad_check = tf.check_numerics(clipped_gradients[0], 'check_numerics caught bad gradients')
    with tf.control_dependencies([grad_check]):
        train_step = optimizer.apply_gradients(zip(clipped_gradients, variables)) #minimize(loss) #, global_step=global_step)
        
    tf.add_to_collection('train_step', train_step)
    # tf.add_to_collection('learning_rate', init_lr)
    return train_step

