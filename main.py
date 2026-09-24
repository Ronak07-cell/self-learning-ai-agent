from environment.grid_world import GridWorld
from agent.q_learning_agent import QLearningAgent


def train(episodes=500):
    env = GridWorld()
    actions = ['up', 'down', 'left', 'right']
    agent = QLearningAgent(env.rows, env.cols, actions)

    steps_per_episode = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            steps += 1

        agent.decay_exploration()
        steps_per_episode.append(steps)

        if (episode + 1) % 50 == 0:
            avg_recent = sum(steps_per_episode[-50:]) / 50
            print(f"Episode {episode + 1}: avg steps (last 50) = {avg_recent:.1f}, exploration_rate = {agent.exploration_rate:.3f}")

    return agent, env, steps_per_episode


if __name__ == "__main__":
    trained_agent, env, history = train(episodes=500)

    print("\n--- TRAINING COMPLETE ---")
    print(f"First episode took {history[0]} steps")
    print(f"Last episode took {history[-1]} steps")