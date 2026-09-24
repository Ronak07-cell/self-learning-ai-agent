import matplotlib.pyplot as plt


def plot_learning_curve(steps_per_episode):
    plt.figure(figsize=(10, 5))
    plt.plot(steps_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Steps to reach goal")
    plt.title("Q-Learning: Steps per Episode Over Training")
    plt.grid(True)
    plt.savefig("learning_curve.png")
    plt.close()
    print("Saved learning_curve.png")
def plot_maze_with_path(grid, path):
    rows = len(grid)
    cols = len(grid[0])

    fig, ax = plt.subplots(figsize=(cols, rows))

    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            if cell == '#':
                color = 'black'
            elif cell == 'S':
                color = 'green'
            elif cell == 'G':
                color = 'red'
            else:
                color = 'white'
            ax.add_patch(plt.Rectangle((c, rows - 1 - r), 1, 1, facecolor=color, edgecolor='gray'))

    path_x = [pos[1] + 0.5 for pos in path]
    path_y = [rows - 1 - pos[0] + 0.5 for pos in path]
    ax.plot(path_x, path_y, color='blue', linewidth=2, marker='o', markersize=4)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("Learned Path Through Maze")
    plt.savefig("maze_path.png")
    plt.close()
    print("Saved maze_path.png")