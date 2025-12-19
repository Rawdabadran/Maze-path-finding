import random
from enum import Enum
import numpy as np
import cv2
import sys

sys.setrecursionlimit(8000)


class Directions(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class Backtracking:
    def __init__(self, height, width, path, displayMaze):
        print("Using OpenCV version: " + cv2.__version__)


        if width % 2 == 0:
            width += 1
        if height % 2 == 0:
            height += 1

        self.width = width
        self.height = height
        self.path = path
        self.displayMaze = displayMaze

    def createMaze(self):

        maze = np.ones((self.height, self.width), dtype=float)


        for i in range(self.height):
            for j in range(self.width):
                if i % 2 == 1 or j % 2 == 1:
                    maze[i, j] = 0  # جدار
                if i == 0 or j == 0 or i == self.height - 1 or j == self.width - 1:
                    maze[i, j] = 0.5


        sx = random.choice(range(2, self.width - 2, 2))
        sy = random.choice(range(2, self.height - 2, 2))


        self.generator(sx, sy, maze)

        maze[maze == 0.5] = 1

        maze[0, :] = 0
        maze[-1, :] = 0
        maze[:, 0] = 0
        maze[:, -1] = 0

        self.start = (1, 2)
        self.end = (self.height - 2, self.width - 3)
        self.end2 = (self.height - 2, 8)

        maze[self.start] = 1
        maze[self.end] = 1
        maze[self.end2] = 1


        if self.displayMaze:
            cv2.namedWindow('Maze', cv2.WINDOW_NORMAL)
            cv2.imshow('Maze', maze)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        maze = maze * 255.0
        cv2.imwrite(self.path, maze)

        return 0

    def generator(self, cx, cy, grid):
        grid[cy, cx] = 0.5
        directions = [Directions.UP, Directions.DOWN, Directions.LEFT, Directions.RIGHT]
        random.shuffle(directions)

        for d in directions:
            if d == Directions.UP:
                nx, ny = cx, cy - 2
                mx, my = cx, cy - 1
            elif d == Directions.DOWN:
                nx, ny = cx, cy + 2
                mx, my = cx, cy + 1
            elif d == Directions.LEFT:
                nx, ny = cx - 2, cy
                mx, my = cx - 1, cy
            else:
                nx, ny = cx + 2, cy
                mx, my = cx + 1, cy

            if 0 < nx < self.width - 1 and 0 < ny < self.height - 1:
                if grid[ny, nx] != 0.5:
                    grid[my, mx] = 0.5
                    self.generator(nx, ny, grid)

if __name__ == "__main__":
    mazeGen = Backtracking(50, 50, "maze.png", True)
    mazeGen.createMaze()
