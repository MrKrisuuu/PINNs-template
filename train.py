import numpy as np
from copy import deepcopy
import torch
from plotting import print_loss, plot_loss, plot_1D, plot_1D_in_2D, plot_compare, plot_difference


def train_PINN(pinn, loss, epochs):
    best_pinn, loss_values = train_model(pinn, loss, epochs=epochs)
    best_pinn = best_pinn.cpu()

    #torch.save(best_pinn, "../results/SIR.pth")

    return best_pinn, loss_values


def train_model(nn_approximator, loss_fn, epochs):
    optimizer = torch.optim.Adam(nn_approximator.parameters())
    #optimizer = torch.optim.LBFGS(nn_approximator.parameters())
    loss_values = []
    min_loss = 10000000000
    best_model = deepcopy(nn_approximator)
    for epoch in range(0, epochs):
        loss, residual_loss, initial_loss, boundary_loss = loss_fn(nn_approximator)
        loss_values.append(
            [loss.item(), residual_loss.item(), initial_loss.item(), boundary_loss.item()])

        if min_loss > float(loss):
            min_loss = float(loss)
            print(f"Epoch: {epoch} - Loss: {float(loss):>7f}")
            best_model = deepcopy(nn_approximator)
        elif (epoch) % 100 == 0:
            print(f"Epoch: {epoch} - Loss: {float(loss):>7f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(lambda: loss)

    loss, residual_loss, initial_loss, boundary_loss = loss_fn(nn_approximator)
    loss_values.append(
        [loss.item(), residual_loss.item(), initial_loss.item(), boundary_loss.item()])

    return best_model, np.array(loss_values)