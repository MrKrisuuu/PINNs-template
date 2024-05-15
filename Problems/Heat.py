from get_points import get_initial_points, get_interior_points, get_boundary_points
from PINN import PINN, device, f, dfdt, dfdx, dfdy
from Problems.Loss import Loss

import torch


def create_Heat():
    t_domain_Heat = [0, 0.5]
    x_domain_Heat = [0, 1]
    y_domain_Heat = [0, 1]

    pinn_Heat = PINN(3, 1).to(device)

    loss_Heat = Loss_Heat(
        t_domain_Heat,
        x_domain_Heat,
        y_domain_Heat,
        n_points=8000
    )

    return pinn_Heat, loss_Heat, x_domain_Heat, y_domain_Heat, t_domain_Heat


class Loss_Heat(Loss):
    def residual_loss(self, pinn):
        x, y, t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())

        loss = dfdt(pinn, x, y, t) - dfdx(pinn, x, y, t, order=2) - dfdy(pinn, x, y, t, order=2)

        return loss.pow(2).mean()

    def initial_loss(self, pinn):
        x, y, t = get_initial_points(*self.args, n_points=self.n_points, device=pinn.device())

        # loss = f(pinn, x, y, t) - torch.exp(-(torch.sqrt((x-0.5)**2 + (y-0.5)**2)*7)**2)

        loss = f(pinn, x, y, t) - x

        return loss.pow(2).mean()

    def boundary_loss(self, pinn: PINN):
        down, up, left, right = get_boundary_points(*self.args, n_points=self.n_points, device=pinn.device())

        x_down, y_down, t_down = down
        x_up, y_up, t_up = up
        x_left, y_left, t_left = left
        x_right, y_right, t_right = right

        loss_down = dfdy(pinn, x_down, y_down, t_down)
        loss_up = dfdy(pinn, x_up, y_up, t_up)
        loss_left = dfdx(pinn, x_left, y_left, t_left)
        loss_right = dfdx(pinn, x_right, y_right, t_right)

        return loss_down.pow(2).mean() + \
               loss_up.pow(2).mean() + \
               loss_left.pow(2).mean() + \
               loss_right.pow(2).mean()