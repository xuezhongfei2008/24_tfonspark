# coding=utf-8
import numpy as np
import tensorflow as tf

# python example.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=ps --task_index=0
# python example.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=worker --task_index=0
# python example.py --ps_hosts=127.0.0.1:2222 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=worker --task_index=1

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.00003, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 1000,
                            'Steps to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    print("ps_hosts",ps_hosts)
    print("worker_hosts",worker_hosts)
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    print("FLAGS.task_index",FLAGS.task_index)
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index,start=True)
    print("sss",server.target,type(server.target),server.target.decode(),type(server.target.decode()))
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            global_step =  tf.train.get_or_create_global_step()

            input = tf.placeholder("float")
            label = tf.placeholder("float")

            weight = tf.get_variable("weight", [1], tf.float32, initializer=tf.random_normal_initializer())
            biase = tf.get_variable("biase", [1], tf.float32, initializer=tf.random_normal_initializer())
            pred = tf.multiply(input, weight) + biase

            loss_value = loss(label, pred)

            train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_value, global_step=global_step)
            init_op = tf.initialize_all_variables()

            saver = tf.train.Saver()
            tf.summary.scalar('cost', loss_value)
            summary_op = tf.summary.merge_all()

        print("sv")
        hooks=[tf.train.StopAtStepHook(last_step=1000)]
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])


        with tf.train.MonitoredTrainingSession(
                                    master=server.target.decode(),
                                    config=config,
                                    is_chief=(FLAGS.task_index == 0 and (FLAGS.job_name=="worker")),
                                    hooks=hooks,
                                    checkpoint_dir="./checkpoint/") as mon_sess:
            print("sess111")
            while not mon_sess.should_stop():
                print("sess1")
                step = 0
                while step < 1000:
                    print("step",step)
                    train_x = np.random.randn(1)
                    train_y = 2 * train_x + np.random.randn(1) * 0.33 + 10
                    _, loss_v, step = mon_sess.run([train_op, loss_value, global_step],
                                                   feed_dict={input: train_x, label: train_y})
                    print("steps_to_validate",steps_to_validate)
                    if step % steps_to_validate == 0:
                            w, b = mon_sess.run([weight, biase])
                            print("step: %d, weight: %f, biase: %f, loss: %f" % (step, w, b, loss_v))

            # sv.stop()


def loss(label, pred):
    return tf.square(label - pred)


if __name__ == "__main__":
    tf.app.run()
