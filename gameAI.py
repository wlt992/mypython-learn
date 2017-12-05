import pygame
import random
from pygame.locals import *
import numpy as np
from collections import deque
import tensorflow as tf  # http://blog.topspeedsnail.com/archives/10116
import cv2               # http://blog.topspeedsnail.com/archives/4755

BLACK     = (0  ,0  ,0  )
WHITE     = (255,255,255)

SCREEN_SIZE = [100,100]
BAR_SIZE = [30, 5] # 滑块的大小
BALL_SIZE = [20, 20] # 球的大小

# 神经网络的输出
MOVE_STAY = [1, 0, 0]
MOVE_LEFT = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]

class Game(object):
	def __init__(self):
		pygame.init()
		self.clock = pygame.time.Clock()
		self.screen = pygame.display.set_mode(SCREEN_SIZE)
		pygame.display.set_caption('Simple Game')

		self.ball_pos_x = SCREEN_SIZE[0]//2 - BALL_SIZE[0]/2 # 球的位置坐标
		self.ball_pos_y = SCREEN_SIZE[1]//2 - BALL_SIZE[1]/2

		self.ball_dir_x = -1 # -1 = left 1 = right  # 控制球的运动方向
		self.ball_dir_y = -1 # -1 = up   1 = down
		self.ball_pos = pygame.Rect(self.ball_pos_x, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])

		self.bar_pos_x = SCREEN_SIZE[0]//2-BAR_SIZE[0]//2
		self.bar_pos = pygame.Rect(self.bar_pos_x, SCREEN_SIZE[1]-BAR_SIZE[1], BAR_SIZE[0], BAR_SIZE[1])

	# action是MOVE_STAY、MOVE_LEFT、MOVE_RIGHT
	# ai控制棒子左右移动；返回游戏界面像素数和对应的奖励。(像素->奖励->强化棒子往奖励高的方向移动)
	def step(self, action):

		if action == MOVE_LEFT:
			self.bar_pos_x = self.bar_pos_x - 10 # 控制滑块移动
		elif action == MOVE_RIGHT:
			self.bar_pos_x = self.bar_pos_x + 10
		else:
			pass
		# 防止滑块滑出屏幕
		if self.bar_pos_x < 0: 
			self.bar_pos_x = 0
		if self.bar_pos_x > SCREEN_SIZE[0] - BAR_SIZE[0]:
			self.bar_pos_x = SCREEN_SIZE[0] - BAR_SIZE[0]
			
		self.screen.fill(BLACK)
		self.bar_pos.left = self.bar_pos_x
		pygame.draw.rect(self.screen, WHITE, self.bar_pos)

		self.ball_pos.left += self.ball_dir_x * 2 # 控制球滑动，每次沿着运动方向x轴运动两个单位,最好是画布的因子
		self.ball_pos.bottom += self.ball_dir_y * 3 # 控制球滑动，每次沿着运动方向Y轴运动3个单位, 3<=BAR_SIZE[1]
		pygame.draw.rect(self.screen, WHITE, self.ball_pos) # 球的下一个位置draw到画布中

		# 防止球滑出画布, 当球在滑块Y轴下面时，可以开始往回走
		if self.ball_pos.top <= 0 or self.ball_pos.bottom >= (SCREEN_SIZE[1] - BAR_SIZE[1] + 2): 
			self.ball_dir_y = self.ball_dir_y * -1
		if self.ball_pos.left <= 0 or self.ball_pos.right >= (SCREEN_SIZE[0]):
			self.ball_dir_x = self.ball_dir_x * -1

		reward = 0
		if self.bar_pos.top <= self.ball_pos.bottom and self.ball_dir_y == 1: # 限定球的运动方向为延y轴向下
			if self.bar_pos.left <= self.ball_pos.right and self.bar_pos.right >= self.ball_pos.left:
				reward = 1    # 击中奖励
			else:
				reward = -1   # 没击中惩罚

		# 获得游戏界面像素
		screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
		pygame.display.update()
		# 返回游戏界面像素和对应的奖励
		return reward, screen_image

# learning_rate
LEARNING_RATE = 0.9
# 更新梯度
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
# 测试观测次数
EXPLORE = 500000 
OBSERVE = 500
# 存储过往经验大小
REPLAY_MEMORY = 500

BATCH = 1
input_shape = SCREEN_SIZE[0] * SCREEN_SIZE[1]
hidden_size1 = 1000
out_size = 3

output = 3  # 输出层神经元数。代表3种操作-MOVE_STAY:[1, 0, 0]  MOVE_LEFT:[0, 1, 0]  MOVE_RIGHT:[0, 0, 1]
input_image = tf.placeholder("float", [None, input_shape])  # 游戏像素
action = tf.placeholder("float", [None, output])     # 操作

# 定义DNN
def deep_neural_network(input_image):
	weights = {'w1': tf.Variable(tf.random_normal([input_shape, hidden_size1])),
				# 'w2': tf.Variable(tf.random_normal([hidden_size1, hidden_size2])),
				# 'w3': tf.Variable(tf.random_normal([hidden_size2, hidden_size3])),
				'wo': tf.Variable(tf.random_normal([hidden_size1, out_size]))}
	biases = {'b1': tf.Variable(tf.zeros([hidden_size1])),
				# 'b2': tf.Variable(tf.zeros([hidden_size2])),
				# 'b3': tf.Variable(tf.zeros([hidden_size3])),
				'bo': tf.Variable(tf.zeros([out_size]))}

	h1 = tf.nn.relu(tf.matmul(input_image, weights['w1']) + biases['b1'])
	# h2 = tf.nn.relu(tf.matmul(h1, weights['w2']) + biases['b2'])
	# h3 = tf.nn.relu(tf.matmul(h2, weights['w3']) + biases['b3'])
	out = tf.matmul(h1, weights['wo']) + biases['bo']
	return out
# 定义CNN-卷积神经网络 参考:http://blog.topspeedsnail.com/archives/10451
# def convolutional_neural_network(input_image):
# 	weights = {'w_conv1':tf.Variable(tf.random_normal([8, 8, 1, 32])),
#                'w_conv2':tf.Variable(tf.random_normal([4, 4, 32, 64])),
#                'w_conv3':tf.Variable(tf.random_normal([3, 3, 64, 64])),
#                'w_fc4':tf.Variable(tf.random_normal([3456, 784])),
#                'w_out':tf.Variable(tf.random_normal([784, output]))}

# 	biases = {'b_conv1':tf.Variable(tf.zeros([32])),
#               'b_conv2':tf.Variable(tf.zeros([64])),
#               'b_conv3':tf.Variable(tf.zeros([64])),
#               'b_fc4':tf.Variable(tf.zeros([784])),
#               'b_out':tf.Variable(tf.zeros([output]))}

# 	conv1 = tf.nn.relu(tf.nn.conv2d(input_image, weights['w_conv1'], strides = [1, 4, 4, 1], padding = "VALID") + biases['b_conv1'])
# 	conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w_conv2'], strides = [1, 2, 2, 1], padding = "VALID") + biases['b_conv2'])
# 	conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights['w_conv3'], strides = [1, 1, 1, 1], padding = "VALID") + biases['b_conv3'])
	
# 	# when padding = 'VALID', o = (i-f)//s + 1
# 	# a = lambda x: ((((x-8)//4+1)-4)//2+1)-3 + 1
# 	# a(80) * a(100) = 6*9=54, 54 * 64 = 3456
# 	conv3_flat = tf.reshape(conv3, [-1, 3456])
# 	fc4 = tf.nn.relu(tf.matmul(conv3_flat, weights['w_fc4']) + biases['b_fc4'])

# 	output_layer = tf.matmul(fc4, weights['w_out']) + biases['b_out']
# 	return output_layer

# 深度强化学习入门: https://www.nervanasys.com/demystifying-deep-reinforcement-learning/
# 训练神经网络
def train_neural_network(input_image):
	predict_action = deep_neural_network(input_image) # CNN预测得到的每个动作的概率

	argmax = tf.placeholder("float", [None, output]) # 没有优化的CNN or 随机运动得到的动作向量
	gt = tf.placeholder("float", [None]) # 贴现未来的状态得到的reward

	# argmax: 前一个CNN训练得到的action, predict_action: 后面CNN训练得到的action
	# predict_action: (BATCH_SIZE, 3)
	# argmax: (BATCH_SIZE, 3)
	# 通过CNN训练得到reward，按照之前CNN的选择，该reward与预期的gt的差值尽可能小
	action = tf.reduce_sum(tf.matmul(predict_action, tf.reshape(argmax, (3, -1))), reduction_indices = 1) 
	cost = tf.reduce_mean(tf.square(action - gt))
	optimizer = tf.train.AdamOptimizer(1e-6).minimize(cost)

	game = Game()
	D = deque()

	_, image = game.step(MOVE_STAY)
	# 转换为灰度值
	image = cv2.cvtColor(cv2.resize(image, (SCREEN_SIZE[0], SCREEN_SIZE[1])), cv2.COLOR_BGR2GRAY)
	# 转换为二值
	ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY) # >1 变为255, <=1 变为0
	input_image_data = np.reshape(image, (-1))#np.stack((image, image, image, image), axis = 2) # image的个数可以调节
	win_cnt = 0
	lose_cnt = 0
	random_cnt = 0
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		
		saver = tf.train.Saver()
		# saver.restore(sess, 'game.cpk-2000')

		n = 0
		epsilon = INITIAL_EPSILON
		while True:
			action_t = predict_action.eval(feed_dict = {input_image : [input_image_data]})[0]
			argmax_t = np.zeros([output], dtype=np.int) # 记录step向量

			if(random.random() <= 0.1): # 看是否随机选择移动方向
				maxIndex = random.randrange(output) # 随机选择移动方向
				random_cnt += 1
			else:
				maxIndex = np.argmax(action_t) # 根据CNN选择移动方向
			argmax_t[maxIndex] = 1
			
			if epsilon > FINAL_EPSILON:
				epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE # EXPLORE次数后，epsilon = FINAL_EPSILON

			#for event in pygame.event.get():  macOS需要事件循环，否则白屏
			#	if event.type == QUIT:
			#		pygame.quit()
			#		sys.exit()
			reward, image = game.step(list(argmax_t))

			image = cv2.cvtColor(cv2.resize(image, (SCREEN_SIZE[0], SCREEN_SIZE[1])), cv2.COLOR_BGR2GRAY) # 生成(80,100)的shape
			ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
			input_image_data1 = np.reshape(image, (-1))#np.stack((image, image, image, image), axis = 2)
			# input_image_data1 = np.append(image, input_image_data[:, :, :-1], axis = 2) # 把新的图像替换第三维的0向量

			D.append((input_image_data, argmax_t, reward, input_image_data1)) # D存储memory, 格式为<s,a,r,s'>

			if len(D) > REPLAY_MEMORY: # D长度超过最大memory长度时，从左边pop掉之前的记忆
				D.popleft()

			if n > OBSERVE: # 超过观察记忆长度后
				minibatch = random.sample(D, BATCH) # 从D中sample BATCH长度的记忆
				input_image_data_batch = [d[0] for d in minibatch] # 取minibatch中的所有s
				argmax_batch = [d[1] for d in minibatch] # minibatch中的a
				reward_batch = [d[2] for d in minibatch] # minibatch中的r
				input_image_data1_batch = [d[3] for d in minibatch] # minibatch中的s'

				gt_batch = []

				out_batch = predict_action.eval(feed_dict = {input_image : input_image_data1_batch}) # Q(s',a')

				for i in range(0, len(minibatch)):
					gt_batch.append(reward_batch[i] + LEARNING_RATE * np.max(out_batch[i])) # step最大概率 * LEARNING_RATE

				# 输入s, a, r'
				# argmax_batch: 未强化学习前的step选择(0,1值)
				# input_image_data_batch: 输入到CNN中学习新的每个步骤的reward
				# gt_batch: 贴现了未来的reward得到，比现有的rewar_batch[i]要大，作为强化学习的指引者
				# 更新权重，使得input_image_data_batch学习出来的predict_action, 在argmax_batch的选择下，能不断贴近gt_batch
				# 每接近一次，gt_batch会更新，不断强化这个过程
				# argmax有的模型不用，真实的x,y 为 s, r' = r + Q(s',a')
				optimizer.run(feed_dict = {gt : gt_batch, argmax : argmax_batch, input_image : input_image_data_batch}) 
				if reward == 1:
					win_cnt += 1
				elif reward == -1:
					lose_cnt += 1
				else:
					pass
			input_image_data = input_image_data1
			n = n+1

			# if n % 1000 == 0:
			# 	saver.save(sess, './game.cpk', global_step = n)  # 保存模型

			print(n, "random_cnt:", random_cnt, " " ,"action:", maxIndex, " " ,"reward:", reward, "win_cnt", win_cnt, "lost_cnt", lose_cnt)


train_neural_network(input_image)