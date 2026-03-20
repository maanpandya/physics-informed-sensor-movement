"""
Gaussian Process model and Physics-Informed Loss (PIL) utilities.

The GPModel class wraps GPyTorch's ExactGP with a Squared Exponential (RBF)
kernel. Spatial derivatives of the GP posterior mean are obtained via
finite differences, consistent with the analytical form in the paper.
The PIL measures the point-wise PDE residual of the GP field estimate,
and is used during Stage II to guide sensor movement.
"""

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal


def calculate_physics_informed_loss(temporal_derivatives, spatial_gradients, laplacian_values, v_x, v_y, D):
    """
    Compute the Physics-Informed Loss (PIL) at a set of evaluation points.

    The PIL is the squared PDE residual of the advection-diffusion equation:
        R = du/dt + v_x * du/dx + v_y * du/dy - D * (d²u/dx² + d²u/dy²)

    Args:
        temporal_derivatives: Tensor (N,) — ∂u/∂t estimate at each point.
        spatial_gradients:    Tensor (N, 2) — [∂u/∂x, ∂u/∂y] at each point.
        laplacian_values:     Tensor (N,) — ∇²u estimate at each point.
        v_x, v_y:             Advection velocity components.
        D:                    Diffusion coefficient.

    Returns:
        total_loss:      Scalar sum of squared residuals.
        squared_residuals: Tensor (N,) of per-point squared residuals.
    """
    residuals = (temporal_derivatives
                 + v_x * spatial_gradients[:, 0]
                 + v_y * spatial_gradients[:, 1]
                 - D * laplacian_values)
    squared_residuals = torch.square(residuals)
    return torch.sum(squared_residuals), squared_residuals

class GPModel(ExactGP):
    """
    Exact Gaussian Process model with a Squared Exponential (RBF) kernel.

    Hyperparameters (length-scale, output-scale, noise) are optimised by
    maximising the exact marginal log-likelihood via Adam.
    """

    def __init__(self, train_x, train_y, likelihood, length_scale=1.0, outputscale=1.0):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.kernel = ScaleKernel(RBFKernel())
        self.kernel.base_kernel.lengthscale = length_scale
        self.kernel.outputscale = outputscale
        self.mean_module = gpytorch.means.ConstantMean()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.kernel(x)
        return MultivariateNormal(mean_x, covar_x)

    def fit(self, train_x, train_y, num_iter=100, learning_rate=0.01, verbose=False):
        """Train the GP by maximising the exact marginal log-likelihood."""
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(num_iter):
            optimizer.zero_grad()
            output = self(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            if verbose and i % 50 == 0:
                print(f"Iteration {i + 1}/{num_iter} - Loss: {loss.item():.4f}")

    def fit_with_pil(self, train_x, train_y, u_prev, time_step_size,
                     anchor_points, v_x, v_y, D, lambda_pil,
                     num_iter=100, learning_rate=0.01, only_pil=True):
        """
        Train the GP with a combined NLL + PIL objective.

        Args:
            u_prev:         Previous-step GP mean at anchor points (for ∂u/∂t).
            time_step_size: Δt for the temporal derivative approximation.
            anchor_points:  Collocation points where the PIL is evaluated.
            v_x, v_y, D:   Estimated PDE parameters.
            lambda_pil:     Weighting factor for the PIL term.
            only_pil:       If True, optimise only the PIL (ignores NLL).
        """
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        for i in range(num_iter):
            optimizer.zero_grad()
            output_current = self(train_x)
            nll_loss = -mll(output_current, train_y)
            pil_loss = self.compute_pil_loss(u_prev, time_step_size,
                                             anchor_points, v_x, v_y, D)
            total_loss = nll_loss + lambda_pil * pil_loss
            
            if only_pil:
                pil_loss.backward()
            else:
                total_loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Iteration {i+1}/{num_iter} - Total Loss: {total_loss.item():.4f} "
                      f"- NLL: {nll_loss.item():.4f} - PIL: {pil_loss.item():.4f}")

    def compute_pil_loss(self, u_prev, time_step_size, anchor_points, v_x, v_y, D):
        u_current = self.predict(anchor_points, requires_grad=True)[0]
        temporal_derivatives = self.compute_temporal_derivative(u_current, u_prev, time_step_size)
        spatial_gradients, laplacian_values = self.compute_finite_difference_derivatives(anchor_points)
        pil_loss = calculate_physics_informed_loss(temporal_derivatives, spatial_gradients, laplacian_values,
                                                   v_x, v_y, D)[0]
        return pil_loss

    def predict(self, test_x, requires_grad=False):
        """Return posterior mean and variance at test_x."""
        self.eval()
        self.likelihood.eval()
        if requires_grad:
            with gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self(test_x))
                mean = observed_pred.mean
                var = observed_pred.variance
        else:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self(test_x))
                mean = observed_pred.mean
                var = observed_pred.variance
        return mean, var

    def compute_finite_difference_derivatives(self, pts, delta=0.05):
        """
        Estimate first- and second-order spatial derivatives of the GP mean
        via central finite differences with step size `delta`.

        Returns:
            grad_u:  Tensor (N, 2) — [∂u/∂x, ∂u/∂y].
            lap_u:   Tensor (N,)   — averaged second-order spatial derivative.
        """
        pts = pts.view(-1, 2)
        
        x_plus = pts.clone()
        x_plus[:,0] += delta
        x_minus = pts.clone()
        x_minus[:,0] -= delta

        y_plus = pts.clone()
        y_plus[:,1] += delta
        y_minus = pts.clone()
        y_minus[:,1] -= delta

        mean_c = self.predict(pts, requires_grad=False)[0]
        mean_xp = self.predict(x_plus, requires_grad=False)[0]
        mean_xm = self.predict(x_minus, requires_grad=False)[0]
        mean_yp = self.predict(y_plus, requires_grad=False)[0]
        mean_ym = self.predict(y_minus, requires_grad=False)[0]

        dudx = (mean_xp - mean_xm) / (2.0 * delta)
        dudy = (mean_yp - mean_ym) / (2.0 * delta)

        d2udx2 = (mean_xp - 2.0*mean_c + mean_xm) / (delta**2)
        d2udy2 = (mean_yp - 2.0*mean_c + mean_ym) / (delta**2)
        lap_u = (d2udx2 + d2udy2)*0.5

        grad_u = torch.stack([dudx, dudy], dim=-1)
        return grad_u, lap_u

    def compute_temporal_derivative(self, u_current, u_prev, time_step_size):
        return (u_current - u_prev) / time_step_size

def initialize_gp_model(sensor_locations, sensor_data, length_scale=1.0, outputscale=1.0, noise=1e-6):
    """
    Instantiate a GPModel with a fixed low noise level suitable for
    near-noiseless analytical field observations.

    Args:
        sensor_locations: Tensor (M, 2) — training input locations.
        sensor_data:      Tensor (M,)   — training observations.
        length_scale:     Initial RBF length-scale.
        outputscale:      Initial kernel output-scale.
        noise:            Fixed observation noise variance.

    Returns:
        gp_model:   Initialised GPModel instance.
        likelihood: Associated GaussianLikelihood.
    """
    likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(noise))
    likelihood.noise_covar.noise = noise
    likelihood.noise_covar.raw_noise.requires_grad_(False)
    gp_model = GPModel(sensor_locations, sensor_data, likelihood, length_scale, outputscale)
    return gp_model, likelihood