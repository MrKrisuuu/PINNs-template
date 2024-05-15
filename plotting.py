import matplotlib.pyplot as plt
import torch
import imageio
import numpy as np
import os


def print_loss(loss, pinn):
    losses = loss.verbose(pinn)
    print(f'Total loss: \t{losses[0]:.6f}    ({losses[0]:.3E})')
    print(f'Interior loss: \t{losses[1]:.6f}    ({losses[1]:.3E})')
    print(f'Initial loss: \t{losses[2]:.6f}    ({losses[2]:.3E})')
    print(f'Bondary loss: \t{losses[3]:.6f}    ({losses[3]:.3E})')


def running_average(values, window=100):
    s = int(window / 2)
    avgs = []
    p = max(0, -s)
    q = min(len(values), s + 1)
    current_sum = sum(values[p:q])
    for i in range(0, len(values)):
        new_p = max(0, i - s)
        new_q = min(len(values), i + s + 1)
        if new_p != p:
            current_sum -= values[p]
        if new_q != q:
            current_sum += values[new_q - 1]
        avgs.append(current_sum / (new_q - new_p + 1))
        p = new_p
        q = new_q
    return np.array(avgs)


def plot_loss(loss_values, window=100, title="problem", save="loss"):
    average_loss_total = running_average(loss_values[:, 0], window=window)
    average_loss_residual = running_average(loss_values[:, 1], window=window)
    average_loss_initial = running_average(loss_values[:, 2], window=window)
    average_loss_boundary = running_average(loss_values[:, 3], window=window)

    epochs = list(range(len(average_loss_total)))
    plt.plot(epochs, average_loss_total, label="Total loss")
    plt.plot(epochs, average_loss_residual, label="Residual loss")
    plt.plot(epochs, average_loss_initial, label="Initial loss")
    plt.plot(epochs, average_loss_boundary, label="Boundary loss")

    plt.title(f"Loss function for {title} (running average)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    if not os.path.exists("./results"):
        os.makedirs("./results")
    plt.savefig(f"./results/{save}.png")
    plt.show()


def plot_1D(pinn, t, ylabel="Values", labels=None, title="problem", save="plot"):
    plt.plot(t.detach().cpu().numpy(), pinn(t).detach().cpu().numpy(), label=labels)
    plt.title(f"Result for {title} by PINN" )
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    if labels:
        plt.legend()
    plt.grid(True)
    if not os.path.exists("./results"):
        os.makedirs("./results")
    plt.savefig(f"./results/{save}.png")
    plt.show()


def plot_1D_in_2D(pinn, t, title="1D_2D", save="1D_2D"):
    data = pinn(t).detach().cpu().numpy()
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    if not os.path.exists("./results"):
        os.makedirs("./results")
    plt.savefig(f"./results/{save}.png")
    plt.show()


def plot_2D(pinn, x, t, name="2D"):
    files = []
    for t_raw in t:
        t0 = torch.full_like(x, t_raw.item())
        plt.ylim(-2, 2)
        plt.plot(x.detach().cpu().numpy(), pinn(x, t0).detach().cpu().numpy())
        time = round(t_raw.item(), 2)
        plt.title(f"Step: {time}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        if not os.path.exists("./plot2D"):
            os.makedirs("./plot2D")
        plt.savefig(f"./plot2D/{time}.png")
        files.append(f"./plot2D/{time}.png")
        plt.clf()

    if not os.path.exists("./results"):
        os.makedirs("./results")
    with imageio.get_writer(f"./results/{name}.gif", mode="I") as writer:
        for filename in files:
            image = imageio.v2.imread(filename)
            writer.append_data(image)


def plot_3D(pinn, x, y, t, name="3D"):
    files = []
    x_grid, y_grid = torch.meshgrid(x.reshape(-1), y.reshape(-1), indexing="ij")
    for t_raw in t:
        t0 = torch.full_like(x_grid, t_raw.item())
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.axes.set_zlim3d(bottom=0, top=1)
        ax.plot_surface(x_grid.detach().cpu().numpy(), y_grid.detach().cpu().numpy(),
                        pinn(x_grid.reshape(-1, 1), y_grid.reshape(-1, 1), t0.reshape(-1, 1)).detach().cpu().numpy().reshape(x_grid.shape))
        time = round(t_raw.item(), 2)
        plt.title(f"Step: {time}")
        if not os.path.exists("./plot3D"):
            os.makedirs("./plot3D")
        plt.savefig(f"./plot3D/{time}.png")
        files.append(f"./plot3D/{time}.png")
        plt.clf()
        plt.close()

    if not os.path.exists("./results"):
        os.makedirs("./results")
    with imageio.get_writer(f"./results/{name}.gif", mode="I") as writer:
        for filename in files:
            image = imageio.v2.imread(filename)
            writer.append_data(image)
