#encoding:utf-8

import numpy as np
import tensorflow as tf
'''tensorflow 0.12.1'''
import sys

cluster=tf.train.ClusterSpec({"worker":["localhost:2222","localhost:2223"]})

point_num=int(sys.argv[1])

x=tf.placeholder(tf.float32,point_num)
y=tf.placeholder(tf.float32,point_num)

with tf.device("/job:worker/task:1"):
    print("worker1")
    print("x",x)
    batch_x1=tf.slice(x,[0],[int(point_num/2)])
    batch_y1=tf.slice(y,[0],[int(point_num/2)])
    result1=tf.add(tf.square(batch_x1),tf.square(batch_y1))

with tf.device("/job:worker/task:0"):
    print("worker0")
    batch_x2=tf.slice(x,[int(point_num/2)],[-1])
    batch_y2=tf.slice(y,[int(point_num/2)],[-1])
    result2=tf.add(tf.square(batch_x2),tf.square(batch_y2))
    print("resutt",result1.shape,result2.shape)
    distance=tf.concat([result1,result2],0)


with tf.Session("grpc://localhost:2222") as sess:
    print("sess1")
    result=sess.run(distance,feed_dict={x:np.random.random(point_num),y:np.random.random(point_num)})
    sum=0
    print("sess2")
    for i in range(point_num):
        if result[i]<1:
            sum+=1
    print("pi=%f" % (float(sum)/point_num*4))


















