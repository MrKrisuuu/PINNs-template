from train import train_PINN
from Problems.SIR import create_SIR
from Problems.Heat import create_Heat
from plotting import print_loss, plot_loss, plot_1D, plot_1D_in_2D, plot_2D, plot_3D
from PINN import device

import torch


if __name__ == "__main__":
    # SIR
    pinn_SIR, loss_SIR, t_domain_SIR = create_SIR()

    best_pinn_SIR, loss_values_SIR = train_PINN(pinn_SIR, loss_SIR, 1000)

    print_loss(loss_SIR, pinn_SIR)
    plot_loss(loss_values_SIR, title="SIR model", save="SIR_loss")
    t = torch.linspace(t_domain_SIR[0], t_domain_SIR[1], 1001).reshape(-1, 1).to(device)
    t.requires_grad = True
    plot_1D(pinn_SIR, t, title="SIR model", save="SIR_pinn_result", labels=["S", "I", "R"], ylabel="Population")

    # Heat
    pinn_Heat, loss_Heat, x_domain_Heat, y_domain_Heat, t_domain_Heat = create_Heat()

    best_pinn_Heat, loss_values_Heat = train_PINN(pinn_Heat, loss_Heat, 5000)

    print_loss(loss_Heat, pinn_Heat)
    plot_loss(loss_values_Heat, title="Heat model", save="Heat_loss")
    x = torch.linspace(x_domain_Heat[0], x_domain_Heat[1], 101).reshape(-1, 1).to(device)
    x.requires_grad = True
    y = torch.linspace(y_domain_Heat[0], y_domain_Heat[1], 101).reshape(-1, 1).to(device)
    y.requires_grad = True
    t = torch.linspace(t_domain_Heat[0], t_domain_Heat[1], 101).reshape(-1, 1).to(device)
    t.requires_grad = True
    plot_3D(pinn_Heat, x, y, t, name="Heat_pinn_result")




