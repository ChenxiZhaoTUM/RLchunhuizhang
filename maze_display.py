import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation


def interface():
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    # plt.plot([1, 1], [0, 1], color='red', linewidth=2)
    # plt.plot([1, 2], [2, 2], color='red', linewidth=2)
    # plt.plot([2, 2], [2, 1], color='red', linewidth=2)
    # plt.plot([2, 3], [1, 1], color='red', linewidth=2)

    plt.plot([2, 3], [1, 1], color='red', linewidth=2)
    plt.plot([0, 1], [1, 1], color='red', linewidth=2)
    plt.plot([1, 1], [1, 2], color='red', linewidth=2)
    plt.plot([1, 2], [2, 2], color='red', linewidth=2)

    plt.text(0.5, 2.5, 'S0', size=14, ha='center')
    plt.text(1.5, 2.5, 'S1', size=14, ha='center')
    plt.text(2.5, 2.5, 'S2', size=14, ha='center')
    plt.text(0.5, 1.5, 'S3', size=14, ha='center')
    plt.text(1.5, 1.5, 'S4', size=14, ha='center')
    plt.text(2.5, 1.5, 'S5', size=14, ha='center')
    plt.text(0.5, 0.5, 'S6', size=14, ha='center')
    plt.text(1.5, 0.5, 'S7', size=14, ha='center')
    plt.text(2.5, 0.5, 'S8', size=14, ha='center')
    plt.text(0.5, 2.3, 'START', ha='center')
    plt.text(2.5, 0.3, 'GOAL', ha='center')
    # plt.axis('off')
    plt.tick_params(axis='both', which='both',
                    bottom=False, top=False,
                    right=False, left=False,
                    labelbottom=False, labelleft=False
                    )
    line, = ax.plot([0.5], [2.5], marker='o', color='g', markersize=60)

    return fig, line


def play_process(state_history):

    fig, line = interface()

    def init():
        line.set_data([], [])
        return (line, )

    def animate(i):
        state = state_history[i]
        x = (state % 3) + 0.5
        y = 2.5 - int(state / 3)
        line.set_data(x, y)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(state_history), interval=200, repeat=False)
    anim.save('maze.mp4')

