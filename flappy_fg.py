"""Flappy, game inspired by Flappy Bird.

Exercises

1. Keep score.
2. Vary the speed.
3. Vary the size of the balls.
4. Allow the bird to move forward and back.

"""


from random import *
from turtle import *
from freegames import *


bird = vector(0, 0)
balls = []

score =0
game_frequency = 18

width = 1600
height = 900



def tap(x, y):
	"Move bird up in response to screen tap."
	up = vector(0, 30)
	bird.move(up)


def inside(point):
	"Return True if point on screen."
	return (-width/2 < point.x < width/2 )and (-height/2 < point.y < height/2)


def draw(alive):
	"Draw screen objects."
	clear()

	goto(bird.x, bird.y)

	if alive:
		dot(10, 'green')
	else:
		dot(10, 'red')

	for ball in balls:
		goto(ball.x, ball.y)
		dot(20, 'black')
	update()



def move():
	"Update object positions."
	bird.y -= 5

	for ball in balls:
		ball.x -= 3

	if randrange(10) == 0:
		y = randrange(-height/2, height/2 )
		ball = vector(width/2 -1, y)
		balls.append(ball)

	while len(balls) > 0 and not inside(balls[0]):
		balls.pop(0)

	if not inside(bird):
		draw(False)
		return

	for ball in balls:
		if abs(ball - bird) < 15:
			draw(False)
			return

	draw(True)
	ontimer(move, game_frequency)


def func():
	print("score")
	tap(0,0)

def nn_tap():
	alive=True
	while(alive):
		for i in range(1,10):
			ontimer(func,i*100)
		alive = False


def run_game():
	setup(width +20, height+20, 400, 400)
	hideturtle()
	up()
	tracer(False)
	for i in range(1,10):
		ontimer(nn_tap,i*5000)
	onscreenclick(tap)
	move()
	done()


def run_game():
	setup(width +20, height+20, 400, 400)
	hideturtle()
	up()
	tracer(False)
	onscreenclick(tap)
	move()
	done()
 

run_game()