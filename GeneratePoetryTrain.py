import collections
import numpy as np
import tensorflow as tf
import os

os.chdir("d:/data")
#-------------------------------数据预处理---------------------------#

poetry_file ='poetry.txt'

# 诗集
poetrys = []
with open(poetry_file, "r", encoding='utf-8',) as f:
	for line in f:
		try:
			title, content = line.strip().split(':')
			content = content.replace(' ','')
			if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
				continue
			if len(content) < 5 or len(content) > 79:
				continue
			content = '[' + content + ']'
			poetrys.append(content)
		except Exception as e: 
			pass

# 按诗的字数排序
poetrys = sorted(poetrys,key=lambda line: len(line))
print('唐诗总数: ', len(poetrys))

# 统计每个字出现次数
all_words = []
for poetry in poetrys:
	all_words += [word for word in poetry]
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)

# 取前多少个常用字
words = words[:len(words)] + (' ',)
# 每个字映射为一个数字ID
word_num_map = dict(zip(words, range(len(words))))
to_num = lambda word: word_num_map.get(word, len(words))

# 把序号转化为中文字符
num_word_map = dict(zip(word_num_map.values(), word_num_map.keys()))
to_word = lambda num: num_word_map.get(num, ' ')


# 把诗转换为向量形式，参考TensorFlow练习1
# map(to_num, poetry): 把poetry应用to_num函数映射成数字
poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys]
#[[314, 3199, 367, 1556, 26, 179, 680, 0, 3199, 41, 506, 40, 151, 4, 98, 1],
#[339, 3, 133, 31, 302, 653, 512, 0, 37, 148, 294, 25, 54, 833, 3, 1, 965, 1315, 377, 1700, 562, 21, 37, 0, 2, 1253, 21, 36, 264, 877, 809, 1]
#....]

# 每次取64首诗进行训练
batch_size = 64
n_chunk = len(poetrys_vector) // batch_size
x_batches = []
y_batches = []
for i in range(n_chunk):
	start_index = i * batch_size % len(poetrys_vector)
	end_index = (start_index + batch_size) % len(poetrys_vector)

	# 选batches首诗出来
	batches = poetrys_vector[start_index:end_index]
	# 最长的诗的长度记录下来
	length = max(map(len,batches))
	# 列长度：按最长的诗的长度生成空白格' '代表的数字的诗，对于每个batch
	xdata = np.full((batch_size,length), word_num_map[' '], np.int)
	
	# 把batches填充到xdata中, 此两步的功能即是把batches的所有诗都拉长到最长诗的长度, 以' '补充
	for row in range(batch_size):
		xdata[row,:len(batches[row])] = batches[row]
	ydata = np.copy(xdata)
	# 把ydata的第二列之后往前挪，移掉第一列，保证'，'预测'，', '。'预测'。' 
	ydata[:,:-1] = xdata[:,1:]
	"""
	xdata             ydata
	[6,2,4,6,9]       [2,4,6,9,9]
	[1,4,2,8,5]       [4,2,8,5,5]
	"""
	x_batches.append(xdata) 
	y_batches.append(ydata)


#---------------------------------------RNN--------------------------------------#

input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])
# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2):
	if model == 'rnn':
		cell_fun = tf.nn.rnn_cell.BasicRNNCell
	elif model == 'gru':
		cell_fun = tf.nn.rnn_cell.GRUCell
	elif model == 'lstm':
		cell_fun = tf.nn.rnn_cell.BasicLSTMCell

	cell = cell_fun(rnn_size, state_is_tuple=True)
	cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

	initial_state = cell.zero_state(batch_size, tf.float32)

	with tf.variable_scope('rnnlm'):
		softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)+1])
		softmax_b = tf.get_variable("softmax_b", [len(words)+1])
		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [len(words)+1, rnn_size])
			inputs = tf.nn.embedding_lookup(embedding, input_data) # batch_size = 1时，input_data代表一首诗序号
			# inputs是把一首诗的序号embedding成矩阵形式，每个字符用rnn_size的向量表示
	
	# dynamic_rnn可以让不同batch的数据长度不同
	outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
	output = tf.reshape(outputs,[-1, rnn_size]) # outputs: (batch_size, 诗长度length, rnn_size)
	# output: 每行是一个字符向量
	# output: 如下每行embedding成向量
	# [
	# 春
	# 眠
	# 不
	# 觉
	# 小
	# ，
	# 处
	# 处
	# 闻
	# 啼
	# 鸟
	# 。]
	# 每个字符对应下一个字符的概率
	logits = tf.matmul(output, softmax_w) + softmax_b
	# logits: output的embedding向量改为len(words) + 1长度的输出
	probs = tf.nn.softmax(logits)
	return logits, last_state, probs, cell, initial_state
#训练
def train_neural_network():
	logits, last_state, _, _, _ = neural_network()
	targets = tf.reshape(output_targets, [-1])
	# tf.contrib.legacy_seq2seq.sequence_loss_by_example不需要targets进行onehotencoding
	loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], len(words))
	cost = tf.reduce_mean(loss)
	learning_rate = tf.Variable(0.0, trainable=False)
	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train_op = optimizer.apply_gradients(zip(grads, tvars))

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		saver = tf.train.Saver(tf.all_variables())

		for epoch in range(50):
			sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
			n = 0
			for batche in range(n_chunk):
				train_loss, _ , _ = sess.run([cost, last_state, train_op], feed_dict={input_data: x_batches[n], output_targets: y_batches[n]})
				n += 1
				print(epoch, batche, train_loss)
			if epoch % 7 == 0:
				saver.save(sess, './poetry.module', global_step=epoch)

train_neural_network()



