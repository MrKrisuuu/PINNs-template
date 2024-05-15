from get_points import get_initial_points, get_interior_points
from PINN import PINN, device, f, dfdt
from Problems.Loss import Loss

def create_SIR():
    t_domain_SIR = [0, 10]

    pinn_SIR = PINN(1, 3).to(device)

    loss_SIR = Loss_SIR(
        t_domain_SIR
    )

    return pinn_SIR, loss_SIR, t_domain_SIR


class Loss_SIR(Loss):
    def residual_loss(self, pinn):
        t = get_interior_points(*self.args, n_points=self.n_points, device=pinn.device())

        (b, y) = (2, 1)

        S = dfdt(pinn, t, output_value=0) + b * f(pinn, t, output_value=1) * f(pinn, t, output_value=0)
        I = dfdt(pinn, t, output_value=1) - b * f(pinn, t, output_value=1) * f(pinn, t, output_value=0) + y * f(pinn, t, output_value=1)
        R = dfdt(pinn, t, output_value=2) - y * f(pinn, t, output_value=1)

        loss = S.pow(2) + I.pow(2) + R.pow(2)

        return loss.mean()

    def initial_loss(self, pinn):
        t = get_initial_points(*self.args, n_points=self.n_points, device=pinn.device())

        S = f(pinn, t, output_value=0) - 0.9
        I = f(pinn, t, output_value=1) - 0.1
        R = f(pinn, t, output_value=2) - 0.0

        loss = S.pow(2) + I.pow(2) + R.pow(2)

        return loss.mean()