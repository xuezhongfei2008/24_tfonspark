import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata = {"batchsize": [], "loss": []}


def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]


# 生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3  # y=2x，但是加入了噪声
# 图形显示
# plt.plot(train_X, train_Y, 'ro', label='Original data')
# plt.legend()
# plt.show()
tf.reset_default_graph()
'''
在一台机器上开3个不同的端口，分布代表ps、chief supervisors和worker
'''
# 定义ip和端口
strps_hosts = "localhost:1681"
strworker_hosts = "localhost:1682,localhost:1683"
# 定义角色名称
strjob_name = "ps"
task_index = 0
ps_hosts = strps_hosts.split(',')
worker_hosts = strworker_hosts.split(',')
cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
# 创建server
server = tf.train.Server(
    {'ps': ps_hosts, 'worker': worker_hosts},
    job_name=strjob_name,
    task_index=task_index)
# ps角色使用join进行等待，开始接收连接信息
if strjob_name == 'ps':
    print("wait")
    server.join()
'''
与正常程序不同，在创建网络结构时，使用tf.device函数将全部节点都放在当前任务下。
tf.device函数中的任务是通过tf.train.replica_device_setter来指定的。
tf.train.replica_device_setter中使用worker_device来定义具体任务名称
使用cluster的配置来指定角色及对应IP地址，从而实现管理整个任务下的图节点
'''
with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % task_index,
        cluster=cluster_spec)):
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    # 模型参数
    W = tf.Variable(tf.random_normal([1]), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")
    # 为了使载入检查点文件时能够同步循环次数，这里加了一个global_step变量。
    global_step = tf.contrib.framework.get_or_create_global_step()  # 获得迭代次数

    # 前向结构
    z = tf.multiply(X, W) + b
    tf.summary.histogram('z', z)  # 将预测值以直方图显示
    # 反向优化
    cost = tf.reduce_mean(tf.square(Y - z))
    tf.summary.scalar('loss_function', cost)  # 将损失以标量显示
    learning_rate = 0.01
    # global_step放到优化器中。这样每次运行一次优化器，global_step会自动获得当期迭代次数。
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,
                                                                          global_step=global_step)  # Gradient descent
    saver = tf.train.Saver(max_to_keep=1)
    merged_summary_op = tf.summary.merge_all()  # 合并所有summary
    # 下面代码是将其前面的变量全部初始化，如果后面再有变量，则不会被初始化，所以，一般要将这行代码放在最后
    init = tf.global_variables_initializer()
# 参数设置
training_epochs = 2200
display_step = 2
'''
在Supervisor函数中，is_chief表明了是否为chief supervisors角色，这里将0号worker设置为chief
logdir为检查点文件和summary文件保存的路径
init_op表示使用初始化变量的函数。
saver需要将保存检查点的saver对象传入，supervisor就会自动保存检查点文件。如果不想自动保存，可以设为None
summary_op也是自动保存summary文件。这里设置为none，表示不自动保存。
save_model_secs为保存检查点文件的时间间隔。这里设为5，表示每5秒自动保存一次检查点文件。
'''
sv = tf.train.Supervisor(is_chief=(task_index == 0),  # 0号worker为chief
                         logdir="log/super/",
                         init_op=init,
                         summary_op=None,
                         saver=saver,
                         global_step=global_step,
                         save_model_secs=5)
# 连接目标角色创建session
with sv.managed_session(server.target) as sess:
    # sess.run(init)
    print("sess ok")
    print(global_step.eval(session=sess))

    for epoch in range(global_step.eval(session=sess), training_epochs * len(train_X)):

        for (x, y) in zip(train_X, train_Y):
            _, epoch = sess.run([optimizer, global_step], feed_dict={X: x, Y: y})
            # 生成summary
            summary_str = sess.run(merged_summary_op, feed_dict={X: x, Y: y});
            # 将summary 写入文件
            sv.summary_computed(sess, summary_str, global_step=epoch)
            if epoch % display_step == 0:
                loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
                print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))
                if not (loss == "NA"):
                    plotdata["batchsize"].append(epoch)
                    plotdata["loss"].append(loss)
                # sv.saver.save(sess,"log/mnist_with_summaries/",global_step=epoch)

    print(" Finished!")
    sv.saver.save(sess, "log/mnist_with_summaries/" + "sv.cpk", global_step=epoch)
sv.stop()