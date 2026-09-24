import numpy as np
import random


class QLearningAgent:
    def __init__(self, rows, cols, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0):
        self.rows = rows
        self.cols = cols
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        self.q_table = np.zeros((rows, cols, len(actions)))

    def choose_action(self, state):
        row, col = state

        if random.uniform(0, 1) < self.exploration_rate:
            action_index = random.randint(0, len(self.actions) - 1)
        else:
            action_index = np.argmax(self.q_table[row, col])

        return self.actions[action_index]

    def learn(self, state, action, reward, next_state):
        row, col = state
        next_row, next_col = next_state
        action_index = self.actions.index(action)

        current_q = self.q_table[row, col, action_index]
        max_future_q = np.max(self.q_table[next_row, next_col])

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )

        self.q_table[row, col, action_index] = new_q

    def decay_exploration(self, decay_rate=0.995, min_rate=0.01):
        self.exploration_rate = max(min_rate, self.exploration_rate * decay_rate)