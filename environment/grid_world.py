import numpy as np


class GridWorld:
    def __init__(self):
        self.grid = [
            ['S', '.', '.', '.'],
            ['.', '#', '.', '.'],
            ['.', '#', '.', '.'],
            ['.', '.', '.', 'G']
        ]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.start_pos = (0, 0)
        self.goal_pos = (3, 3)
        self.agent_pos = self.start_pos

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action):
        row, col = self.agent_pos

        if action == 'up':
            new_row, new_col = row - 1, col
        elif action == 'down':
            new_row, new_col = row + 1, col
        elif action == 'left':
            new_row, new_col = row, col - 1
        elif action == 'right':
            new_row, new_col = row, col + 1
        else:
            new_row, new_col = row, col

        if not (0 <= new_row < self.rows and 0 <= new_col < self.cols):
            new_row, new_col = row, col
        elif self.grid[new_row][new_col] == '#':
            new_row, new_col = row, col

        self.agent_pos = (new_row, new_col)

        if self.agent_pos == self.goal_pos:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        return self.agent_pos, reward, done