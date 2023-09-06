import matplotlib.pyplot as plt
from matplotlib import animation


def display_frame_to_video(frames, output):
    plt.figure(figsize=(frames[0].shape[0] / 72, frames[0].shape[1] / 72), dpi=72)  # length/72, width/72, const dpi
    plt.axis('off')
    patch = plt.imshow(frames[0])

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=range(len(frames)), interval=50)  # time interval, i.e. fps=20
    anim.save(output)


def display_frame_to_video_2(frames):
    plt.figure(figsize=(frames[0].shape[0] / 72, frames[0].shape[1] / 72), dpi=72)  # length/72, width/72
    plt.axis('off')
    patch = plt.imshow(frames[0])

    def animate(frame):
        patch.set_data(frame)

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=frames[1:], interval=50)
    anim.save('cartpole_2.mp4')
    anim.save('cartpole_2.gif', writer='imagemagick')
