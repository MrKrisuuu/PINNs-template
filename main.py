from train import train_PINN
from Problems.SIR import create_SIR
from plotting import print_loss, plot_loss, plot_1D
from PINN import device

import torch


if __name__ == "__main__":
    pinn_SIR, loss_SIR, t_domain_SIR = create_SIR()

    best_pinn_SIR, loss_values_SIR = train_PINN(pinn_SIR, loss_SIR, 1000)

    print_loss(loss_SIR, pinn_SIR)
    plot_loss(loss_values_SIR, title="SIR model", save="SIR_loss")
    t = torch.linspace(t_domain_SIR[0], t_domain_SIR[1], 1001).reshape(-1, 1).to(device)
    t.requires_grad = True
    plot_1D(pinn_SIR, t, title="SIR model", save="SIR_pinn_result", labels=["S", "I", "R"], ylabel="Population")

