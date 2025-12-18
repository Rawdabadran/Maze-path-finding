import random
from enum import Enum
import numpy as np
import cv2
import sys
import heapq
import tkinter as tk
import time
sys.setrecursionlimit(8000)


def get_screen_resolution():
    root = tk.Tk()
    root.withdraw()
    return root.winfo_screenwidth(), root.winfo_screenheight()

class Directions(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class Backtracking:
    def __init__(self, height, width, path, displayMaze):
        print("Using OpenCV version:", cv2.__version__)

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
                    maze[i, j] = 0
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
            screen_width, screen_height = get_screen_resolution()
            maze_resized = cv2.resize(maze, (screen_width, screen_height), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Maze", maze_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        cv2.imwrite(self.path, maze * 255)
        return maze

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


class AStarSolver:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.rows, self.cols = maze.shape
        self.nodes_explored = 0

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(self, node):
        y, x = node
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.rows and 0 <= nx < self.cols:
                if self.maze[ny, nx] == 1:
                    yield (ny, nx)

    def solve(self):
        open_set = []
        heapq.heappush(open_set, (0, self.start))
        came_from = {}
        g_score = {self.start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            self.nodes_explored += 1   # كل مرة نزور عقدة نزود العداد

            if current == self.goal:
                return self.reconstruct_path(came_from)

            for neighbor in self.neighbors(current):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_set, (f, neighbor))
        return None

    def reconstruct_path(self, came_from):
        path = []
        current = self.goal
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(self.start)
        return path[::-1]


if __name__ == "__main__":
    mazeGen = Backtracking(50, 50, "maze.png", False)
    maze = mazeGen.createMaze()

    solver = AStarSolver(maze, mazeGen.start, mazeGen.end)

    # حساب الزمن
    start_time = time.time()
    path = solver.solve()
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000  # بالمللي ثانية

    if path:
        solved = cv2.cvtColor((maze * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for y, x in path:
            solved[y, x] = [0, 0, 255]

        target_width = 1840
        target_height = 920
        solved_resized = cv2.resize(solved, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Solved Maze - A*", solved_resized)
        cv2.imwrite("maze_solved_astar.png", solved_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # طباعة جدول المقارنة
        print("\n=== نتائج A* ===")
        print(f"{'Algorithm':<10} | {'Path Length':<12} | {'Nodes Explored':<15} | {'Time (ms)':<10}")
        print("-"*60)
        print(f"{'A*':<10} | {len(path):<12} | {solver.nodes_explored:<15} | {elapsed_ms:.3f}")
    else:
        print("No path")
