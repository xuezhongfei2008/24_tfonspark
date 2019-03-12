#encoding:utf-8
import sys

task_number=int(sys.argv[1])

import tensorflow as tf
'''tensorflow 0.12.1'''

cluster=tf.train.ClusterSpec({"worker":["localhost:2222","localhost:2223"]})
server=tf.train.Server(cluster,job_name="worker",task_index=task_number)


print("Starting server #{}".format(task_number))

server.start()
server.join()