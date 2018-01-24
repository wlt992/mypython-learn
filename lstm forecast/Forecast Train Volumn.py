import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import requests
import io

# 加载数据
url = 'http://blog.topspeedsnail.com/wp-content/uploads/2016/12/铁路客运量.csv'
ass_data = requests.get(url).content

df = pd.read_csv(io.StringIO(ass_data.decode('utf-8')))  # python2使用StringIO.StringIO

data = np.array(df.iloc[:, 1])
# normalize
normalized_data = (data - np.mean(data)) / np.std(data)

seq_size = 24 # 周期为12时，最好能拿到前一年的数据，类似holt-winter/arima, 不进要考虑到环比，还要考虑同比的因素
forecast_len = 24 # 剔除的长度，留作test
forecast_act = 12 # 实际预测的长度
train_x, train_y = [], [] # y为输入24长度的x后的一个值
for i in range(len(normalized_data) - seq_size - 1 - forecast_len):
	train_x.append(np.expand_dims(normalized_data[i : i + seq_size], axis=1).tolist())
	train_y.append(normalized_data[i + 1 : i + seq_size + 1].tolist())

input_dim = 1


# regression
def ass_rnn(hidden_layer_size=6):
	tf.reset_default_graph() # 防止出现ValueError: Variable rnn/basic_lstm_cell/kernel already exists, disallowed.的错误，train和predict可以放在一起
	X = tf.placeholder(tf.float32, [None, seq_size, input_dim]) # 所有相关的变量都要重新定义，X不能放在reset_default_graph的外面
	Y = tf.placeholder(tf.float32, [None, seq_size])
	W = tf.Variable(tf.random_normal([hidden_layer_size, 1]), name='W') # name variable to save
	b = tf.Variable(tf.random_normal([1]), name='b')

	cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)
	outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32) # outputs.shape: [None, seq_size, hidden_layer_size]
	W_repeated = tf.tile(tf.expand_dims(W, 0), 
						[tf.shape(X)[0], 1, 1]) # tf.tile([1,hidden_layer_size,1],[None,1,1]): [None,hidden_layer_size,1]
	out = tf.matmul(outputs, W_repeated) + b # [None, seq_size, 1]
	out = tf.squeeze(out) # out.shape: [None, seq_size]
	return X, Y, out

def train_rnn():
	X, Y, out = ass_rnn()
	loss = tf.reduce_mean(tf.square(out - Y))
	train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss)

	saver = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		tf.get_variable_scope().reuse_variables()
		sess.run(tf.global_variables_initializer())

		for step in range(10000):
			_, loss_ = sess.run([train_op, loss], feed_dict={X: train_x, Y: train_y})
			if step % 1000 == 0:
				# 用测试数据评估loss
				print(step, loss_)
		print("保存模型: ", saver.save(sess, './ass.model'))

train_rnn()

def prediction():
	X, Y, out = ass_rnn()

	saver = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		#tf.get_variable_scope().reuse_variables()
		saver.restore(sess, './ass.model')
		
		prev_seq = train_x[-1]
		predict = []
		for i in range(forecast_act):
			next_seq = sess.run(out, feed_dict={X: [prev_seq]})
			predict.append(next_seq[-1])
			prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

		plt.figure()
		plt.plot(list(range(len(normalized_data))), normalized_data, color='b')
		plt.plot(list(range(len(normalized_data) - forecast_len, len(normalized_data) - forecast_len + forecast_act)), predict, color='r')
		plt.show()

		plt.figure()
		plt.plot(list(range(len(normalized_data))), normalized_data*np.std(data)+np.mean(data), color='b')
		plt.plot(list(range(len(normalized_data) - forecast_len, len(normalized_data) - forecast_len + forecast_act)), np.dot(predict, np.std(data))+np.mean(data), color='r')
		plt.show()
		return normalized_data, predict

y, pred_y = prediction()