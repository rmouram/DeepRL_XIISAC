'''
 Disclaimer: 
 	Esse código (e seus dependentes) são fortemente baseados no código do DQN-Atari-Tensorflow, do usuário Songrotek (https://github.com/songrotek/DQN-Atari-Tensorflow)
 	Pequenas modificações foram feitas para adequar o código ao workshop ministrado na XII Semana Acadêmica de Computação.
'''

# ### Libraries ### #
import tensorflow as tf 
import numpy as np 
import random
from collections import deque 

# ### Hyper Parameteres ### #
FRAME_PER_ACTION = 1		# Numéro de timesteps a esperar por uma ação
GAMMA = 0.95 				# Taxa de decaimento de observações anteriores
OBSERVE = 4000. 			# Timesteps dedicados a Observação
EXPLORE = 4500. 			# Timesteps dedicados a Exploração (modificação de Epsilon)
FINAL_EPSILON = 0.1			# Valor final de Epsilon (0.01)
INITIAL_EPSILON = 1.0		# Valor inicial de Epsilon
REPLAY_MEMORY = 200000 		# Tamanho do Replay Memory
BATCH_SIZE = 32 			# Tamanho do Minibatch de Memória para Treino
UPDATE_TIME = 10000			# Número de timesteps para copiar a Rede Neural Target

# ### Classes ### #
# Essa é a classe principal do nosso Workshop. Ela irá agregar todas as funções do nosso agente para o Deep Q-Learning.
class BrainDQN:
	# Construtor
	def __init__(self,actions):
		# 1º Passo: Inicialização dos parâmetros de treino
		self.replayMemory = deque()
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.actions = actions
		
		# 2º Passo: Inicialização de toda a Rede Neural Convolucional do DQN (Função createQNetwork())
		self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()

		# 3º Passo: Inicialização da Rede Neural Target para o treinamento
		self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()
		self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),
											self.b_conv1T.assign(self.b_conv1),
											self.W_conv2T.assign(self.W_conv2),
											self.b_conv2T.assign(self.b_conv2),
											self.W_conv3T.assign(self.W_conv3),
											self.b_conv3T.assign(self.b_conv3),
											self.W_fc1T.assign(self.W_fc1),
											self.b_fc1T.assign(self.b_fc1),
											self.W_fc2T.assign(self.W_fc2),
											self.b_fc2T.assign(self.b_fc2)]

		# 4ª Passo: Inicialização do método de treinamento
		self.createTrainingMethod()

		# 5º Passo: Criando sessão do Tensorflow e recarregando RedesNeurais pré-treinadas
		self.session = tf.InteractiveSession()
		self.session.run(tf.global_variables_initializer())

		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.session, checkpoint.model_checkpoint_path)
			print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
			print("Could not find old network weights")


	# Essa função será responsável por definir todas as camadas, pesos e bias da Rede Neural
	# Além disso, como o Tensorflow nos permite, também já definiremos os neurônios diretamente pelas suas operações de ativação
	def createQNetwork(self):
		# 1º Passo: Inicialização dos pesos e bias das Redes Neurais 
		# (permite configurar suas dimensões)
		W_conv1 = self.weight_variable([8, 8, 4, 32])
		b_conv1 = self.bias_variable([32])

		W_conv2 = self.weight_variable([4, 4, 32, 64])
		b_conv2 = self.bias_variable([64])

		W_conv3 = self.weight_variable([3, 3, 64, 64])
		b_conv3 = self.bias_variable([64])

		W_fc1 = self.weight_variable([3136, 512])
		b_fc1 = self.bias_variable([512])

		W_fc2 = self.weight_variable([512, self.actions])
		b_fc2 = self.bias_variable([self.actions])

		# 2º Passo: Inicialização de um Placeholder para a informação de entrada (a imagem do estado)
		stateInput = tf.placeholder("float", [None,84,84,4])

		# 3º Passo: Inicialização dos neurônios de cada Camada Oculta de acordo com suas operações de ativação
		h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 4) + b_conv1)
		h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2, 2) + b_conv2)
		h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)

		# 4º Passo: Inicialização dos neurônios da Camada Densa de acordo com suas operações de ativação
		# (Importante ressaltar que a saída final é, justamente, os Q-Values utilizados para o treino)
		h_conv3_flat = tf.reshape(h_conv3,[-1, 3136])

		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
		QValue = tf.matmul(h_fc1,W_fc2) + b_fc2

		return stateInput, QValue, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2

	# Essa função apenas define a execução, na sessão, da operação de cópia da Rede Neural para a Rede Neural Target
	def copyTargetQNetwork(self):
		self.session.run(self.copyTargetQNetworkOperation)

	# Essa função inicializa os parâmetros e operações do treinamento Q-Learning
	def createTrainingMethod(self):
		# 1º Passo: Inicialização da Entrada e Saída da Rede Neural
		self.actionInput = tf.placeholder("float",[None,self.actions])
		self.yInput = tf.placeholder("float", [None]) 
		
		# 2º Passo: Definição da operação de cálculo do Q-Value de acordo com o custo da ação tomada e o valor esperado
		Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
		
		self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
		self.trainStep = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.cost)

	# Essa função compreende a principal função de treinamento do nosso agente, que engloba tanto os conceitos de Deep Learning
	# como Q-Learning.
	def trainQNetwork(self):
		# 1º Passo: Selecionar, e organizar, uma amostra aleatória (batch) do Replay_Memory
		minibatch = random.sample(self.replayMemory, BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]

		# 2º Passo: Calcular, para esse batch, os valores esperados de saída da Rede Neural (dado a Entrada)
		y_batch = []
		QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT: nextState_batch})

		for i in range(0, BATCH_SIZE):
			terminal = minibatch[i][4]
			if terminal:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

		# 3º Passo: Realizamos um step de treinamento, utilizando as funções extraídas.
		self.trainStep.run(feed_dict={self.yInput: 		y_batch,
									  self.actionInput:	action_batch,
									  self.stateInput: 	state_batch})

		# 4º Passo: Salvamos o estado da Rede Neural a cada 10000 Timesteps
		if self.timeStep % 10000 == 0:
			self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.timeStep)

		# 5º Passo: Copiamos o estado da Rede Neural para o Target, caso o Timestep seja igual ao tempo de atualização
		if self.timeStep % UPDATE_TIME == 0:
			self.copyTargetQNetwork()

	# Essa função será responsável por receber a informação de um estado do Emulador, e utilizar essas informações
	# como entradas para o Treinamento da Rede Neural.
	def setPerception(self, nextObservation, action, reward, terminal):
		# 1º Passo: Adicionamos o novo estado ao estado atual
		newState = np.append(nextObservation,self.currentState[:,:,1:], axis = 2)
		self.replayMemory.append((self.currentState,action,reward,newState,terminal))

		if len(self.replayMemory) > REPLAY_MEMORY:
			self.replayMemory.popleft()

		# 2º Passo: Realizamos um step de treinamento (caso estejamos no modo de treino)
		if self.timeStep > OBSERVE:
			self.trainQNetwork()

		# 3º Passo (Visualização): Imprimimos o estado em que o algoritmo se encontra
		if self.timeStep <= OBSERVE:
			state = "observar"
		elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
			state = "explorar"
		else:
			state = "treinar"

		print("TIMESTEP:", self.timeStep, "/ STATE:", state, "/ EPSILON:", self.epsilon)

		# 4º Passo: Atualizamos as informações de treino
		self.currentState = newState
		self.timeStep += 1

	# Essa função será utilizada para selecionar a melhor ação possível dado o estado atual do Agente,
	# tendo em vista as ações permitidas e os parâmetros de treino circunstanciais.
	def getAction(self):
		# 1º Passo: Extraimos o vetor de Q-Values do nosso estado atual
		QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})[0]
		
		# 2º Passo: Selecionamos a melhor ação possível, uma ação aleatória ou não fazemos nada
		action = np.zeros(self.actions)
		action_index = 0

		if self.timeStep % FRAME_PER_ACTION == 0:
			if random.random() <= self.epsilon:
				action_index = random.randrange(self.actions)
				action[action_index] = 1
			else:
				action_index = np.argmax(QValue)
				action[action_index] = 1
		else:
			action[0] = 1

		# 3º Passo: Atualizamos o EPSILON por cada ação tomada
		if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

		# Retorno:
		return action

	# #### Funções Auxiliares #### #
	def setInitState(self, observation):
		self.currentState = np.stack((observation, observation, observation, observation), axis = 2)

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev = 0.01)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)

	def conv2d(self, x, W, stride):
		return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")