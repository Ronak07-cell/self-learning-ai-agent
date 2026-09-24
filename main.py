from environment.grid_world import GridWorld
from agent.q_learning_agent import QLearningAgent


BIG_MAZE = [
    ['S', '.', '.', '.', '#', '.', '.', '.'],
    ['#', '#', '.', '#', '#', '.', '#', '.'],
    ['.', '.', '.', '#', '.', '.', '#', '.'],
    ['.', '#', '#', '#', '.', '#', '#', '.'],
    ['.', '#', '.', '.', '.', '#', '.', '.'],
    ['.', '#', '.', '#', '#', '#', '.', '#'],
    ['.', '.', '.', '#', '.', '.', '.', '#'],
    ['#', '#', '.', '.', '.', '#', '.', 'G'],
]


def train(episodes=2000, grid=None):
    env = GridWorld(grid)
    actions = ['up', 'down', 'left', 'right']
    agent = QLearningAgent(env.rows, env.cols, actions)

    steps_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        max_steps = 300

        while not done and steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            steps += 1

        agent.decay_exploration()
        steps_per_episode.append(steps)

        if (episode + 1) % 200 == 0:
            avg_recent = sum(steps_per_episode[-200:]) / 200
            print(f"Episode {episode + 1}: avg steps (last 200) = {avg_recent:.1f}, exploration_rate = {agent.exploration_rate:.3f}")

    return agent, env, steps_per_episode


def demonstrate(agent, env):
    state = env.reset()
    done = False
    path = [state]

    original_exploration = agent.exploration_rate
    agent.exploration_rate = 0

    while not done and len(path) < 50:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        path.append(next_state)
        state = next_state

    agent.exploration_rate = original_exploration

    print("\n--- DEMONSTRATION (pure learned behavior, no randomness) ---")
    print(f"Path taken ({len(path) - 1} steps): {path}")
    print("Reached goal!" if done else "Did not reach goal.")


if __name__ == "__main__":
    trained_agent, env, history = train(episodes=2000, grid=BIG_MAZE)

    print("\n--- TRAINING COMPLETE ---")
    print(f"First episode took {history[0]} steps")
    print(f"Last episode took {history[-1]} steps")

    demonstrate(trained_agent, env)