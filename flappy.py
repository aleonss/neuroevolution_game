from itertools import *
import random
import sys

import pygame
from pygame.locals import *

from neural_network import *
import numpy as np
import copy
import pickle



scores = []

# Neuroevolution : hiperparametros
selection_ratio=0.5
mutation_rate=0.1
top_notch = 2
pop_size = 15
nn_layout = [6,10,8,6,1,]

# Neuroevolution : Funciones
def nn_selection(players,selection_ratio):
	mating_pool = []
	n_parent = int(1 - pop_size * selection_ratio)
	fittest_index = np.argsort(scores)[n_parent:][::-1]

	for index in fittest_index:
		mating_pool.append(players[index])
	return mating_pool

def nn_mutation(children, mutation_rate):
	offspring = []
	for child in children:
		offspring.append(child.mutate(mutation_rate))
	return offspring

def nn_cross_over(mating_pool):
	children = []
	for comb in list(combinations(mating_pool, 2)):
		children.append(comb[0].cross_over(comb[1]))

	while (children.__len__() < pop_size):
		rand_parent0 = mating_pool[np.random.randint(0, mating_pool.__len__())]
		rand_parent1 = mating_pool[np.random.randint(0, mating_pool.__len__())]
		children.append(rand_parent0.cross_over(rand_parent1))

	return children[:pop_size]



# feed neeural network
def nn_flap(nn, f_counter, playery, playerVelY, upperPipes, lowerPipes):
	input_nn = [playery, playerVelY,]
	for uPipe, lPipe in zip(upperPipes, lowerPipes): 
		input_nn.append(uPipe['x'])
		input_nn.append(lPipe['y'])

	flap = nn.feed(input_nn)[0]
	#print(input_nn,"->",flap)
	return flap > 0.5

# Funciones utiles
def save(players,highscore):
	with open('saves/nn_data.pkl', 'wb') as output:
		pickle.dump(players, output, pickle.HIGHEST_PROTOCOL)
	with open('saves/highscore.pkl', 'wb') as output:
		pickle.dump(highscore, output, pickle.HIGHEST_PROTOCOL)
	print("saved! ",highscore)

def switch_fps(FPS,):
	if FPS==30:
		FPS= 6000
	else:
		FPS= 30
	return FPS






# Parametros del juego

SCREENWIDTH  = 288*1
SCREENHEIGHT = 512*1
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
	# red bird
	(
		'assets/sprites/redbird-upflap.png',
		'assets/sprites/redbird-midflap.png',
		'assets/sprites/redbird-downflap.png',
	),
	# blue bird
	(
		# amount by which base can maximum shift to left
		'assets/sprites/bluebird-upflap.png',
		'assets/sprites/bluebird-midflap.png',
		'assets/sprites/bluebird-downflap.png',
	),
	# yellow bird
	(
		'assets/sprites/yellowbird-upflap.png',
		'assets/sprites/yellowbird-midflap.png',
		'assets/sprites/yellowbird-downflap.png',
	),
)

# list of backgrounds
BACKGROUNDS_LIST = (
	#'assets/sprites/background-day.png',
	'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
	'assets/sprites/pipe-green.png',
	'assets/sprites/pipe-red.png',
)



try:
	xrange
except NameError:
	xrange = range




def main(demo,load_players,save_players):
	global SCREEN, FPSCLOCK
	pygame.init()
	FPSCLOCK = pygame.time.Clock()
	SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
	pygame.display.set_caption('tarea3')

	# numbers sprites for score display
	IMAGES['numbers'] = (
		pygame.image.load('assets/sprites/0.png').convert_alpha(),
		pygame.image.load('assets/sprites/1.png').convert_alpha(),
		pygame.image.load('assets/sprites/2.png').convert_alpha(),
		pygame.image.load('assets/sprites/3.png').convert_alpha(),
		pygame.image.load('assets/sprites/4.png').convert_alpha(),
		pygame.image.load('assets/sprites/5.png').convert_alpha(),
		pygame.image.load('assets/sprites/6.png').convert_alpha(),
		pygame.image.load('assets/sprites/7.png').convert_alpha(),
		pygame.image.load('assets/sprites/8.png').convert_alpha(),
		pygame.image.load('assets/sprites/9.png').convert_alpha()
	)

	# game over sprite
	IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
	# message sprite for welcome screen
	IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
	# base (ground) sprite
	IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

	# sounds
	if 'win' in sys.platform:
		soundExt = '.wav'
	else:
		soundExt = '.ogg'

	#global vars
	FPS = 30
	iplayer = 0
	highscore = 0
	pop_cnt = 0
	players = []

	if(demo<1):
		FPS = 6000
	if(load_players):
		#for i in range(0,pop_size):
		with open('saves/nn_data.pkl', 'rb') as input:
			players = pickle.load(input)
		with open('saves/highscore.pkl', 'rb') as input:
			highscore = pickle.load(input)
		for i in range(0,pop_size):
			scores.append(0)
	else:
		for i in range(0,pop_size):
			layers = make_layers(nn_layout)
			nn = NeuralNetwork(layers)
			players.append(nn)
			scores.append(0)

	while True:
		# select random background sprites
		randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
		IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

		# select random player sprites
		randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
		IMAGES['player'] = (
			pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
			pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
			pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
		)

		# select random pipe sprites
		pipeindex = random.randint(0, len(PIPES_LIST) - 1)
		IMAGES['pipe'] = (
			pygame.transform.rotate(
				pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
			pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
		)

		# hismask for pipes
		HITMASKS['pipe'] = (
			getHitmask(IMAGES['pipe'][0]),
			getHitmask(IMAGES['pipe'][1]),
		)

		# hitmask for player
		HITMASKS['player'] = (
			getHitmask(IMAGES['player'][0]),
			getHitmask(IMAGES['player'][1]),
			getHitmask(IMAGES['player'][2]),
		)

		movementInfo = showWelcomeAnimation()
		crashInfo = mainGame(movementInfo,FPS,players, iplayer,highscore,pop_cnt)
		showGameOverScreen(crashInfo)

		#update highscore
		if(scores[iplayer]>highscore):
			highscore= scores[iplayer]
			if(save_players):
				save(players,highscore)
		
		iplayer+=1
		if(iplayer==pop_size):

			fittest_index = np.argsort(scores)[pop_size-1]
			print(pop_cnt, ",",scores[fittest_index])

			mating_pool = nn_selection(players, selection_ratio)

			bestest = mating_pool[:top_notch]

			children = nn_cross_over(mating_pool)
			players = nn_mutation(children, mutation_rate)

			players = bestest[:top_notch] + children[:(pop_size-top_notch)]
			iplayer= 0
			pop_cnt+=1


			#fittest = fget_fittest(input)
		
		if(demo>0):
			if(pop_cnt%demo==0 and iplayer==0):
				FPS=30
			elif(pop_cnt%demo==0 and iplayer==1):
				FPS=6000



def showWelcomeAnimation():
	"""Shows welcome screen animation of flappy bird"""
	# index of player to blit on screen
	playerIndex = 0
	playerIndexGen = cycle([0, 1, 2, 1])
	# iterator used to change playerIndex after every 5th iteration
	loopIter = 0

	playerx = int(SCREENWIDTH * 0.2)
	playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

	messagex = int((SCREENWIDTH - IMAGES['message'].get_width()) / 2)
	messagey = int(SCREENHEIGHT * 0.12)

	basex = 0
	# amount by which base can maximum shift to left
	baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

	# player shm for up-down motion on welcome screen
	playerShmVals = {'val': 0, 'dir': 1}

	while True:
		for event in pygame.event.get():
			if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
				pygame.quit()
				sys.exit()

		return {
			'playery': playery + playerShmVals['val'],
			'basex': basex,
			'playerIndexGen': playerIndexGen,
		}




def showGameOverScreen(crashInfo):
	"""crashes the player down ans shows gameover image"""
	score = crashInfo['score']
	playerx = SCREENWIDTH * 0.2
	playery = crashInfo['y']
	playerHeight = IMAGES['player'][0].get_height()
	playerVelY = crashInfo['playerVelY']
	playerAccY = 2
	playerRot = crashInfo['playerRot']
	playerVelRot = 7

	basex = crashInfo['basex']
	upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']

	#autoplay
	return


def playerShm(playerShm):
	"""oscillates the value of playerShm['val'] between 8 and -8"""
	if abs(playerShm['val']) == 8:
		playerShm['dir'] *= -1

	if playerShm['dir'] == 1:
		 playerShm['val'] += 1
	else:
		playerShm['val'] -= 1


def getRandomPipe():
	"""returns a randomly generated pipe"""
	# y of gap between upper and lower pipe
	gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
	gapY += int(BASEY * 0.2)
	pipeHeight = IMAGES['pipe'][0].get_height()
	pipeX = SCREENWIDTH + 10

	return [
		{'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
		{'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
	]




def showScore(score):
	"""displays score in center of screen"""
	scoreDigits = [int(x) for x in list(str(score))]
	totalWidth = 0 # total width of all numbers to be printed

	for digit in scoreDigits:
		totalWidth += IMAGES['numbers'][digit].get_width()

	Xoffset = (SCREENWIDTH - totalWidth) / 2

	for digit in scoreDigits:
		SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, 4))
		Xoffset += IMAGES['numbers'][digit].get_width()


def showHighscore(highscore):
	highscoreDigits = [int(x) for x in list(str(highscore))]
	totalWidth = 0 # total width of all numbers to be printed
	for digit in highscoreDigits:
		totalWidth += IMAGES['numbers'][digit].get_width()

	Xoffset = (SCREENWIDTH - totalWidth) / 2
	for digit in highscoreDigits:
		SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, 50))
		Xoffset += IMAGES['numbers'][digit].get_width()


def showPlayerId(pop_cnt, iplayer):
	iplayerDigits = [int(x) for x in list(str(iplayer))]
	offset = 4
	for digit in iplayerDigits:
		SCREEN.blit(IMAGES['numbers'][digit], (offset, 4))
		offset += IMAGES['numbers'][digit].get_width()

	popDigits = [int(x) for x in list(str(pop_cnt))]
	offset = 4
	for digit in popDigits:
		SCREEN.blit(IMAGES['numbers'][digit], (offset, 50))
		offset += IMAGES['numbers'][digit].get_width()






def checkCrash(player, upperPipes, lowerPipes):
	"""returns True if player col lders with base or pipes."""
	pi = player['index']
	player['w'] = IMAGES['player'][0].get_width()
	player['h'] = IMAGES['player'][0].get_height()

	# if player crashes into ground
	if player['y'] + player['h'] >= BASEY - 1:
		return [True, True]

	elif player['y'] < -20:
		return [True, True]
	else:

		playerRect = pygame.Rect(player['x'], player['y'],
					  player['w'], player['h'])
		pipeW = IMAGES['pipe'][0].get_width()
		pipeH = IMAGES['pipe'][0].get_height()

		for uPipe, lPipe in zip(upperPipes, lowerPipes):
			# upper and lower pipe rects
			uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
			lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

			# player and upper/lower pipe hitmasks
			pHitMask = HITMASKS['player'][pi]
			uHitmask = HITMASKS['pipe'][0]
			lHitmask = HITMASKS['pipe'][1]

			# if bird collided with upipe or lpipe
			uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
			lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

			if uCollide or lCollide:
				return [True, False]

	return [False, False]

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
	"""Checks if two objects collide and not just their rects"""
	rect = rect1.clip(rect2)

	if rect.width == 0 or rect.height == 0:
		return False

	x1, y1 = rect.x - rect1.x, rect.y - rect1.y
	x2, y2 = rect.x - rect2.x, rect.y - rect2.y

	for x in xrange(rect.width):
		for y in xrange(rect.height):
			if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
				return True
	return False

def getHitmask(image):
	"""returns a hitmask using an image's alpha."""
	mask = []
	for x in xrange(image.get_width()):
		mask.append([])
		for y in xrange(image.get_height()):
			mask[x].append(bool(image.get_at((x,y))[3]))
	return mask





def mainGame(movementInfo, FPS, players, iplayer, highscore, pop_cnt):
	score = playerIndex = loopIter = 0
	playerIndexGen = movementInfo['playerIndexGen']
	playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']


	basex = movementInfo['basex']
	baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

	# get 2 new pipes to add to upperPipes lowerPipes list
	newPipe1 = getRandomPipe()
	newPipe2 = getRandomPipe()

	# list of upper pipes
	upperPipes = [
		{'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
		{'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
	]

	# list of lowerpipe
	lowerPipes = [
		{'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
		{'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
	]

	pipeVelX = -4

	# player velocity, max velocity, downward accleration, accleration on flap
	playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
	playerMaxVelY =  10   # max vel along Y, max descend speed
	playerMinVelY =  -8   # min vel along Y, max ascend speed
	playerAccY    =   1   # players downward accleration
	playerRot     =  45   # player's rotation
	playerVelRot  =   3   # angular speed
	playerRotThr  =  20   # rotation threshold
	playerFlapAcc =  -9   # players speed on flapping
	playerFlapped = False # True when player flaps


	f_counter = 0
	nn = players[iplayer]

	while True:
		for event in pygame.event.get():
			if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
				pygame.quit()
				sys.exit()
			elif event.type == KEYDOWN:
				if(event.key == K_SPACE):
					FPS= switch_fps(FPS,)
				elif(event.key == K_s):
					save(players,highscore)

		if(nn_flap(nn, f_counter, playery, playerVelY, upperPipes, lowerPipes)):
			playerVelY = playerFlapAcc
			playerFlapped = True
			#SOUNDS['wing'].play()
			


		# check for crash here
		crashTest = checkCrash({'x': playerx, 'y': playery, 'index': playerIndex},
							   upperPipes, lowerPipes)
		if crashTest[0]:
			scores[iplayer] = score
			return {
				'y': playery,
				'groundCrash': crashTest[1],
				'basex': basex,
				'upperPipes': upperPipes,
				'lowerPipes': lowerPipes,
				'score': score,
				'playerVelY': playerVelY,
				'playerRot': playerRot
			}

		# check for score
		playerMidPos = playerx + IMAGES['player'][0].get_width() / 2
		for pipe in upperPipes:
			pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
			if pipeMidPos <= playerMidPos < pipeMidPos + 4:
				score += 1000
				#SOUNDS['point'].play()
		score += 1
		f_counter+= 1

		# playerIndex basex change
		if (loopIter + 1) % 3 == 0:
			playerIndex = next(playerIndexGen)
		loopIter = (loopIter + 1) % 30
		basex = -((-basex + 100) % baseShift)

		# rotate the player
		if playerRot > -90:
			playerRot -= playerVelRot

		# player's movement
		if playerVelY < playerMaxVelY and not playerFlapped:
			playerVelY += playerAccY
		if playerFlapped:
			playerFlapped = False

			# more rotation to cover the threshold (calculated in visible rotation)
			playerRot = 45

		playerHeight = IMAGES['player'][playerIndex].get_height()
		playery += min(playerVelY, BASEY - playery - playerHeight)

		# move pipes to left
		for uPipe, lPipe in zip(upperPipes, lowerPipes):
			uPipe['x'] += pipeVelX
			lPipe['x'] += pipeVelX

		# add new pipe when first pipe is about to touch left of screen
		if 0 < upperPipes[0]['x'] < 5:
			newPipe = getRandomPipe()
			upperPipes.append(newPipe[0])
			lowerPipes.append(newPipe[1])

		# remove first pipe if its out of the screen
		if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
			upperPipes.pop(0)
			lowerPipes.pop(0)

		# draw sprites
		SCREEN.blit(IMAGES['background'], (0,0))

		for uPipe, lPipe in zip(upperPipes, lowerPipes):
			SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
			SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

		SCREEN.blit(IMAGES['base'], (basex, BASEY))

		# print score so player overlaps the score
		showScore(score)
		showPlayerId(pop_cnt, iplayer)
		showHighscore(highscore)

		# Player rotation has a threshold
		visibleRot = playerRotThr
		if playerRot <= playerRotThr:
			visibleRot = playerRot
		
		playerSurface = pygame.transform.rotate(IMAGES['player'][playerIndex], visibleRot)
		SCREEN.blit(playerSurface, (playerx, playery))

		pygame.display.update()
		FPSCLOCK.tick(FPS)


if __name__ == '__main__':
	demo= 5
	load_players= True
	save_players= False

	main(demo,load_players,save_players)


#highscore=5000
#with open('saves/highscore.pkl', 'wb') as output:
#	pickle.dump(highscore, output, pickle.HIGHEST_PROTOCOL)