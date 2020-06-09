# ### Libraries ### #
from ale_python_interface import ALEInterface
import numpy as np 
import cv2
from random import randrange

# ### Classes ### #
# Vamos definir uma classe para facilitar a comunicação com o Arcade Learning Environment.
# A função dessa classe será, basicamente, carregar os jogos e seus parâmetros, além de facilitar a representação
# de alguns dados para o programa principal
class Atari:
	# Constructor
	def __init__(self,rom_name):
		# 1º Passo: carregamos o jogo e definimos seus parâmetros
		self.ale = ALEInterface()
		self.max_frames_per_episode = self.ale.getInt(b"max_num_frames_per_episode")
		self.ale.setInt(b"random_seed",123)
		self.ale.setInt(b"frame_skip",4)
		self.ale.loadROM(('game/' + rom_name).encode())

		self.screen_width,self.screen_height = self.ale.getScreenDims()
		self.legal_actions = self.ale.getMinimalActionSet()
		self.action_map = dict()

		for i in range(len(self.legal_actions)):
			self.action_map[self.legal_actions[i]] = i

		# 2º Passo: criamos a janela para exibição
		self.windowname = rom_name
		cv2.startWindowThread()
		cv2.namedWindow(rom_name)

	# Essa função será utilizada para receber uma imagem do emulador, já em um formato esperado
	# por nosso algoritmo de treinamento.
	def get_image(self):
		numpy_surface = np.zeros(self.screen_height*self.screen_width*3, dtype=np.uint8)
		self.ale.getScreenRGB(numpy_surface)
		image = np.reshape(numpy_surface, (self.screen_height, self.screen_width, 3))
		return image

	# Simplesmente inicializa o jogo
	def newGame(self):
		self.ale.reset_game()
		return self.get_image()

	# Essa função será responsável por retornar as informações da observação do estado após certa ação ser tomada.
	def next(self, action):
		reward = self.ale.act(self.legal_actions[np.argmax(action)])	
		nextstate = self.get_image()
		
		cv2.imshow(self.windowname,nextstate)
		if self.ale.game_over():
			self.newGame()

		return nextstate, reward, self.ale.game_over()