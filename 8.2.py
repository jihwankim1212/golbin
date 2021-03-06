import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

# 하이퍼파라미터
learning_rate= 0.01     # 학습률
training_epoch = 10     # 전체 데이터를 학습할 총횟수
batch_size = 100        # 미니배치로 한 번에 학습할 데이터의 개수
n_hidden = 56           # 입력값의 크기
n_input = 28*28         # MNIST의 이미지 크기
n_output = 28*28        # MNIST의 이미지 크기

# 플레이스홀더 X (비지도 학습이기 떄문에 Y 없음)
X = tf.placeholder(tf.float32, [None, n_input])

# 인코더
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))

# 디코더
W_decode = tf.Variable(tf.random_normal([n_hidden, n_output]))
b_decode = tf.Variable(tf.random_normal([n_output]))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_decode))

# 손실 비용
cost = tf.reduce_mean(tf.pow(X - decoder, 2))
# optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# 신경망 모델 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_batch = int(mnist.train.num_examples / batch_size)
print('mnist.test.num_examples  : ' , mnist.test.num_examples)
print('mnist.train.num_examples : ' , mnist.train.num_examples)

for epoch in range(training_epoch):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs})

        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Abg. cost =', '{:.4f}'.format(total_cost / total_batch))

print('최적화 완료 !')

sample_size = 10
samples = sess.run(decoder, feed_dict={X: mnist.test.images[:sample_size]})

fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

plt.show()
