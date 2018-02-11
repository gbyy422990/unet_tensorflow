#coding:utf-8
#Bin GAO

import os
import tensorflow as tf
import numpy as np

learning_rate=0.001
num_class=2
loss_weight = np.array([1.0,1.0])

def conv2d(x,n_filters,training,name,pool=True,activation=tf.nn.relu):
    with tf.variable_scope('layer{}'.format(name)):
        for index,filter in enumerate(n_filters):
            conv = tf.layers.conv2d(x,filter,(3,3),strides=1,padding='same',activation=None,name='conv_{}'.format(index+1))
            conv = tf.layers.batch_normalization(conv,training=training,name='bn_{}'.format(index+1))
            conv = activation(conv,name='relu{}_{}'.format(name,index+1))
        if pool is False:
            return conv

        pool = tf.layers.max_pooling2d(conv,pool_size=(2,2),strides=2,name='pool_{}'.format(name))

        return conv,pool

def upsampling_2d(tensor,name,size=(2,2)):
    h_,w_,c_ = tensor.get_shape().as_list()[1:]
    h_multi,w_multi = size
    h = h_multi * h_
    w = w_multi * w_
    target = tf.image.resize_nearest_neighbor(tensor,size=(h,w),name='upsample_{}'.format(name))

    return target

def upsampling_concat(input_A,input_B,name):
    upsampling = upsampling_2d(input_A,name=name,size=(2,2))
    up_concat = tf.concat([upsampling,input_B],axis=-1,name='up_concat_{}'.format(name))

    return up_concat

def unet(input,training):
    #归一化[-1,1]
    input = input/127.5 - 1


    #input = tf.layers.conv2d(input,3,(1,1),name = 'color')   #filters:一个整数，输出空间的维度，也就是卷积核的数量
    conv1,pool1 = conv2d(input,[8,8],training,name=1)       #卷积两次输出维度64维
    conv2,pool2 = conv2d(pool1,[16,16],training,name=2)
    conv3,pool3 = conv2d(pool2,[32,32],training,name=3)
    conv4,pool4 = conv2d(pool3,[64,64],training,name=4)
    conv5 = conv2d(pool4,[128,128],training,pool=False,name=5)

    up6 = upsampling_concat(conv5,conv4,name=6)
    conv6 = conv2d(up6,[64,64],training,pool=False,name=6)
    up7 = upsampling_concat(conv6,conv3,name=7)
    conv7 = conv2d(up7,[32,32],training,pool=False,name=7)
    up8 = upsampling_concat(conv7,conv2,name=8)
    conv8 = conv2d(up8,[16,16],training,pool=False,name=8)
    up9 = upsampling_concat(conv8,conv1,name=9)
    conv9 = conv2d(up9,[8,8],training,pool=False,name=9)

    return tf.layers.conv2d(conv9, 1, (1, 1), name='final', activation=tf.nn.sigmoid, padding='same')

#交叉熵损失+权重
def loss_CE(y_pred,y_true):
    '''flat_logits = tf.reshape(y_pred,[-1,num_class])
    flat_labels = tf.reshape(y_true,[-1,num_class])
    class_weights = tf.constant(loss_weight,dtype=np.float32)
    weight_map = tf.multiply(flat_labels,class_weights)
    weight_map = tf.reduce_sum(weight_map,axis=1)

    loss_map = tf.nn.softmax_cross_entropy_with_logits(labels=flat_labels,logits=flat_logits)

    weighted_loss = tf.multiply(loss_map,weight_map)

    cross_entropy_mean = tf.reduce_mean(weighted_loss)'''

    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=y_true,logits=y_pred,pos_weight=loss_weight)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #将x的数据格式转化成dtype
    '''labels = tf.cast(y_true,tf.int32)

    flat_logits = tf.reshape(y_pred,(-1,num_class))

    epsilon = tf.constant(value=1e-10)

    flat_logits=tf.add(flat_logits,epsilon)
    tf.shape(flat_logits, name='flat_logits')

    flat_labels = tf.reshape(labels,(-1,1))
    tf.shape(flat_labels,name='flat_shape1')

    labels = tf.reshape(tf.one_hot(flat_labels,depth=num_class),[-1,num_class])
    tf.shape(flat_labels, name='flat_shape2')

    softmax = tf.nn.softmax(flat_logits)
    tf.shape(softmax, name='softmaxx')
    cross_entropy = -tf.reduce_sum(tf.multiply(labels*tf.log(softmax+epsilon),loss_weight),axis=[1])
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')'''

    return cross_entropy_mean

#IOU损失
def loss_IOU(y_pred,y_true):
    H, W, _ = y_pred.get_shape().as_list()[1:]
    flat_logits = tf.reshape(y_pred, [-1, H * W])
    flat_labels = tf.reshape(y_true, [-1, H * W])
    intersection = 2 * tf.reduce_sum(flat_logits * flat_labels, axis=1) + 1e-7   #沿着第一维相乘求和
    denominator = tf.reduce_sum(flat_logits, axis=1) + tf.reduce_sum(flat_labels, axis=1) + 1e-7
    iou = tf.reduce_mean(intersection / denominator)

    return iou

def train_op(loss,learning_rate):

    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    return optimizer.minimize(loss,global_step=global_step)


