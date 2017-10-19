import tensorflow as tf
# import numpy as np
# x = np.array([[1., 1.], [2., 2.]])
# y = np.array([[1, 1.5], [3.3, 3.1]])
#
# y1 = np.array([1., 2., 3., 4., 5., 6., 7.])
# y3 = np.array([2., 4., 6., 8., 10., 12., 14.])
# y2 = np.array([1.2, 2.2, 3.2, 4.2, 5.1, 6.3, 7.2])

def reduce_MAPE(y_real, y_predict):
    # return tf.fill(x.shape, tf.reduce_mean(x))
    a = tf.abs(y_real - y_predict)
    b = tf.divide(a, y_real)
    c = tf.reduce_mean(b)
    return c

def reduce_MAE(y_real, y_predict):
    a = tf.abs(y_real - y_predict)
    c = tf.reduce_mean(a)
    return c

def reduce_Person(y_real, y_predict):
    y_real_mean = tf.fill(y_real.shape, tf.reduce_mean(y_real))
    y_predict_mean = tf.fill(y_predict.shape, tf.reduce_mean(y_predict))
    fenzi = tf.reduce_sum(tf.multiply(y_real - y_real_mean, y_predict - y_predict_mean))
    fenmu1 = tf.pow(tf.reduce_sum(tf.square(y_real - y_real_mean)), 0.5)
    fenmu2 = tf.pow(tf.reduce_sum(tf.square(y_predict - y_predict_mean)), 0.5)
    c=tf.divide(fenzi,tf.multiply(fenmu1,fenmu2))
    return c


# sess=tf.Session()
#
# print(sess.run(reduce_MAE(y1,y3))) # 4.0
# print(sess.run(reduce_MAPE(y1,y2)))# 0.0736054421769
# print(sess.run(reduce_MAPE(y1,y3)))# 1.0
# print(sess.run(reduce_Person(y1, y2))) # 0.999651908638
# print(sess.run(reduce_Person(y1, y3))) # 1.0
