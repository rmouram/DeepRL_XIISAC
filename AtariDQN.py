# ### System Configuration ### #
import sys
sys.path.append("game/")

# ### Libraries ### #
import cv2
from Atari import Atari 
from BrainDQN_Nature import *
import numpy as np 

# ### Functions ### #
# Vamos implementar, primeiramente, a função que implementa o Pré-processamento descrito
# pela técnica do Deep Q-Learning
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
	observation = observation[26:110,:]
	
	ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	
	return np.reshape(observation,(84,84,1))

# Vamos implementar a função principal que inicializará os nossos agentes e os colocarão
# para aplicar o Deep Q-Learning no treinamento.
def playAtari():
	# 1º Passo: Inicializar o Jogo e o Agente
	atari = Atari('breakout.bin')
	actions = len(atari.legal_actions)

	brain = BrainDQN(actions)
	
	# 2º Passo: Obter estado inicial
	action0 = np.array([1,0,0,0])  # do nothing

	observation0, reward0, terminal = atari.next(action0)

	observation0 = cv2.cvtColor(cv2.resize(observation0, (84, 110)), cv2.COLOR_BGR2GRAY)
	observation0 = observation0[26:110,:]
	ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)

	brain.setInitState(observation0)

	# 3º Passo: Realizar treinamento indefenidamente
	while 1!= 0:
		action = brain.getAction()
		
		nextObservation,reward,terminal = atari.next(action)
		nextObservation = preprocess(nextObservation)
		
		brain.setPerception(nextObservation,action,reward,terminal)

# ### Main Program ### #
def main():
	playAtari()

if __name__ == '__main__':
	main()