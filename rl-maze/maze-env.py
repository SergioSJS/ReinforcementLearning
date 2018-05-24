import numpy as np
import time
import sys
import random
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40 # Pixels
MAZE_H = 5 # Grid height
MAZE_W = 5 # Grid width
ACTION_SPACE = ['u','d','l','r']
EPISODES = 10

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.nActions = len(ACTION_SPACE)
        self.title('Maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
        self._buildMaze()       

    def _buildMaze(self):
        print 3
        self.canvas = tk.Canvas(self, bg="white", height=MAZE_H*UNIT, width=MAZE_W*UNIT)
        
        #Create grids
        for row in range(0, MAZE_H*UNIT, UNIT):
            x0, y0, x1, y1 = 0, row, MAZE_H*UNIT, row
            self.canvas.create_line(x0, y0, x1, y1)
        for col in range(0, MAZE_W*UNIT, UNIT):
            x0, y0, x1, y1 = col, 0, col, MAZE_W*UNIT
            self.canvas.create_line(x0, y0, x1, y1)

        # Origin agent
        self.origin = np.array([20, 20])
        self.agent = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='blue'
        )

        # Holes
        hole1Center = self.origin + np.array([UNIT*3, UNIT*2])
        self.hole1 = self.canvas.create_rectangle(
            hole1Center[0] - 15, hole1Center[1] - 15,
            hole1Center[0] + 15, hole1Center[1] + 15,
            fill='black'
        )
        hole2Center = self.origin + np.array([UNIT*2, UNIT*3])
        self.hole2 = self.canvas.create_rectangle(
            hole2Center[0] - 15, hole2Center[1] - 15,
            hole2Center[0] + 15, hole2Center[1] + 15,
            fill='black'
        )

        # Goal
        goalCenter = self.origin + UNIT*3
        self.goal = self.canvas.create_oval(
            goalCenter[0] - 15, goalCenter[1] - 15,
            goalCenter[0] + 15, goalCenter[1] + 15,
            fill='yellow'
        )
     

        self.canvas.pack()

    def render(self):
        time.sleep(0.1)
        self.update()

    def step(self, action):
        s = self.canvas.coords(self.agent)
        baseAction = np.array([0,0])
        
        if action == 'l':
            if s[0] > UNIT:
                baseAction[0] -= UNIT
        elif action == 'r':
            if s[0] < (MAZE_W - 1)*UNIT:
                baseAction[0] += UNIT
        elif action == 'u':
            if s[1] > UNIT:
                baseAction[1] -= UNIT
        elif action == 'd':
            if s[1] < (MAZE_H - 1)*UNIT:
                baseAction[1] += UNIT
        # Moving agent
        self.canvas.move(self.agent, baseAction[0], baseAction[1])
        # Next state
        s_ = self.canvas.coords(self.agent)
        
        # Reward controll
        if s_ == self.canvas.coords(self.goal):
            reward = 1
            done = True
            s_ = 'finished'
        elif s_ in [self.canvas.coords(self.hole1), self.canvas.coords(self.hole2)]:
            reward = -1
            done = True
            s_ = 'finished'
        else:
            reward = -0.01
            done = False
       
        return s_, reward, done

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.agent)
        self.agent = self.canvas.create_rectangle(
            self.origin[0] - 15, self.origin[1] - 15,
            self.origin[0] + 15, self.origin[1] + 15,
            fill='blue'
        )
        return self.canvas.coords(self.agent)

def update():
    for t in range(EPISODES):
        env.reset()
        while True:
            env.render()
            a = random.choice(ACTION_SPACE)
            s, r, done = env.step(a)
            if done:
                print r
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
