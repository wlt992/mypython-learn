import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec import gridspec
import os

def variable_init(size):
	in_dim = size[0]
	w_stddev = 1/tf.sqrt(in_dim/2)
	return tf.random_normal(shape=size, stddev=w_stddev)

X = tf.placeholder(tf.float32, shape=[None, 784])
D_W1 = tf.Variable(variable_init([784, 128]))
D_b1 = tf.Variable(variable_init([128]))

D_W2 = tf.Variable(variable_init([128, 1]))
D_b2 = tf.Variable(variable_init([1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

Z = tf.placeholder(tf.float32, shape=[None, 100])

G_W1 = tf.Variable(variable_init([100, 128]))
G_b1 = tf.Variable(variable_init([128]))

G_W2 = tf.Variable(variable_init([128, 784]))
G_b2 = tf.Variable(variable_init([784]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

def sample_Z(m, n):
	return np.random.uniform(-1, 1, shape=[m, n])

def generator(z):
	G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
	G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
	G_prob = tf.nn.sigmoid(G_log_prob)
	return G_prob

def discriminator(x):
	D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
	D_logit = tf.matmul(D_h1, D_W2) + D_b2

	D_prob = tf.nn.sigmoid(D_logit)

	return D_prob, D_logit


def plot(sample):
	fig = plt.figure(figsize=(4,4))
	gs = gridspec.Gridspec(4, 4)
	gs.update(wspace=0.05, hspace=0.05)

	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xtricklabels([])
		ax.set_ytricklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(28, 28), cmap = 'Grey_r')

	return fig

G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, lables=tf.zeros_like(D_logit_fake)))

D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mb_size = 128
Z_dim = 100

minist = input_date.read_data_sets('./data/MINIST', one_hot = True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('/out'):
	os.mkdir('/out')

i = 0
for it in range(60000):
	# 每2000次输出一张生成器生成的图片
	if it % 2000 == 0:
		samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)}) # 一次生成16个噪音
		fig = plot(samples)
		plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
		i += 1


	x_mb, _ = mnist.train.next_batch(mb_size) # 一次从训练的X中取mb_size作为一个batch

	# 注意：这里没有限定生成器和判别器的输入噪声一样。应该是为了让D和G更加鲁棒，最后不论给什么样的噪声都会生成一张难以辨别的图像
	# 类似破解了X类图像的秘密：不论什么噪声，都会自动生成X类图片。这里感觉有个trick，G可以直接把X的图片存下来啊，不不。G存不下来
	# G里面只有网络参数theta_G。相当于用网络参数theta_G描述了X类图片。有点像编码了。。。
	_, D_loss_curr = sess.run([D_solver, D_loss], feed_dict = {X: X_mb, Z: sample_Z(mb_size, Z_dim)})
	_, G_loss_curr = sess.run([G_solver, G_loss], feed_dict = {Z: sample_Z(mb_size, Z_dim)})

	if it % 2000 == 0:
		print('Iter: {}'.format(it))
		print('D loss: {:.4}'.format(D_loss_curr))
		print('G_loss: {:.4'.format(G_loss_curr))
		