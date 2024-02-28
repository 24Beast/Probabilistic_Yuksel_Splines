# Importing Libraries
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


# Helper Functions
def normalize(x):
    return (x - x.mean()) / x.std()


def getToyData(n=100, shuffle=True, eps=1e-1):
    # Normalize data
    t = torch.arange(0, n * 0.1, 0.1)
    x, y = torch.cos(t) + torch.pi * t, torch.sin(t)
    x = normalize(x) + (torch.rand(n) * eps) - 1
    y = normalize(y) + (torch.rand(n) * eps)
    plt.plot(x, y)
    plt.title("Original Data Distribution")
    plt.show()
    t = (t - t.min()) / (t.max() - t.min())
    points = torch.stack((x, y), dim=1)
    if shuffle:
        shuffle_order = torch.randperm(n)
        t = t[shuffle_order]
        points = points[shuffle_order]
    return t, points


def plot(points, title):
    x, y = points[:, 0], points[:, 1]
    plt.plot(x, y)
    plt.scatter(x, y, c="red", s=0.5)
    plt.title(title)
    plt.show()


# Defining Class
class YukselSpline(nn.Module):
    def __init__(self, num_points, num_dims=2):
        """
        Currently only handling 2D cases.
        Parameters
        ----------
        num_points : int
            Number of points defining the spline. Can only be of the form 2n+1.

        Returns
        -------
        None.

        """
        super(YukselSpline, self).__init__()
        self.num_dimensions = 2
        self.num_points = num_points
        self.num_splines = num_points - 2
        self.initWeights()
        self.initLayers()

    def initWeights(self):
        # Defining Variables for Quadratic Bezier Splines
        self.t_exponents = torch.tensor([i for i in range(3)])
        self.B = torch.tensor([[1, 0, 0], [-2, 2, 0], [1, -2, 1]]).float()
        self.D = torch.zeros((self.num_dimensions,self.num_dimensions))
        for i in range(self.num_dimensions):
            self.D[i,i] = 1

    def initLayers(self):
        P = torch.rand(
            (
                self.num_points,
                self.num_dimensions
                + (self.num_dimensions * (self.num_dimensions + 1) // 2),
            )
        )
        P = (P - P.mean()) / P.std()
        self.P = nn.Parameter(P)

    def getSigmaMatrix(self, sigma):
        num_t = len(sigma)
        sigma_new = torch.zeros((num_t, self.num_dimensions, self.num_dimensions))
        index_pos = 0
        for i in range(self.num_dimensions):
            for j in range(i, self.num_dimensions):
                sigma_new[:, i, j] = sigma[:, index_pos]
                sigma_new[:, j, i] = sigma[:, index_pos]
                index_pos += 1
        return sigma_new
    
    def getSigmaArray(self, sigma_mat):
        num_t = len(sigma_mat)
        d = len(sigma_mat[0])
        new_sigma = torch.zeros((num_t, d*(d+1)//2))
        pos = 0
        for i in range(d):
            for j in range(i,d):
                new_sigma[:, pos] = sigma_mat[:,i,j]
                pos+=1
        return new_sigma

    def layerPass(self, x):
        p = self.P[:, :2] * 1
        sigma = self.P[:, 2:] * 1
        sigma_mat = self.getSigmaMatrix(sigma)
        new_sigma_mat = sigma_mat.transpose(1,2).matmul(sigma_mat)
        new_sigma = self.getSigmaArray(new_sigma_mat) 
        return p, new_sigma

    def forward(self, x):
        t = x.clone()
        P, sigma = self.layerPass(x)
        return self.transform(P, sigma, t)

    def transform(self, P, sigma, t_val):
        mu = self.transformMu(P, t_val)
        sigma = self.transformSigma(sigma, t_val)
        return mu, sigma
    
    def getPoints(self):
        return self.layerPass(0)[0]

    def transformMu(self, P, t_val):
        """
        Parameters
        ----------
        points : torch.tensor
            Torch tensor of the form (Batch.
        t_val : torch.tensor
            Time data. Assumption t_val ranges from 0-1.

        Returns
        -------
        mu : torch.tensor
            Predicted Positions.

        """
        P_new = P[:3]
        # Use transform spline here
        mu = torch.zeros((len(t_val), self.num_dimensions))
        for i in range(self.num_splines - 1):
            d = ((t_val * (self.num_splines - 1)) - i) * 0.5
            d = d.unsqueeze(1)
            choice = (0 <= d) * (d <= 0.5)
            choice = choice * 1
            mu = (1 - choice) * mu
            P_temp = P_new.clone()
            P_new[0] = (P_temp[0] + P_temp[2]) / 4 + P_temp[1] / 2
            P_new[2] = P[i + 3]
            P_new[1] = 2 * (P_temp[2] - (P_new[0] + P_new[2]) / 4)
            f_prev = self.transformSpline(P_temp, d + 0.5)
            f_curr = self.transformSpline(P_new, d)
            mu += choice * (
                (
                    ((np.cos(np.pi * d) ** 2) * f_prev)
                    + ((np.sin(np.pi * d) ** 2) * f_curr)
                )
            )
        return mu

    def transformSigma(self, sigma, t_val):
        """
        Parameters
        ----------
        points : torch.tensor
            Torch tensor of the form (Batch.
        t_val : torch.tensor
            Time data. Assumption t_val ranges from 0-1.

        Returns
        -------
        sigma : torch.tensor
            Predicted Variance.

        """
        P_new = sigma[:3]
        # Use transform spline here
        sigma_new = torch.zeros(
            (len(t_val), (self.num_dimensions * (self.num_dimensions + 1) // 2))
        )
        for i in range(self.num_splines - 1):
            d = ((t_val * (self.num_splines - 1)) - i) * 0.5
            d = d.unsqueeze(1)
            choice = (0 <= d) * (d <= 0.5)
            choice = choice * 1
            sigma_new = (1 - choice) * sigma_new
            P_temp = P_new.clone()
            P_new[0] = ((P_temp[0] + P_temp[2]) / 16) + (P_temp[1] / 4)
            P_new[2] = sigma[i + 3]
            P_new[1] = 4 * (P_temp[2] - (P_new[0] + P_new[2]) / 16)
            f_prev = self.transformSplineVariance(P_temp, d + 0.5)
            f_curr = self.transformSplineVariance(P_new, d)
            sigma_new += choice * (
                (
                    ((np.cos(np.pi * d) ** 4) * f_prev)
                    + ((np.sin(np.pi * d) ** 4) * f_curr)
                )
            )
        return sigma_new

    def transformSpline(self, p, t_val):
        t_val = t_val[:, 0].repeat((3, 1)).T
        t_exp = self.t_exponents.repeat((len(t_val), 1))
        t = t_val.pow(t_exp)
        f = t.matmul(self.B).matmul(p)
        return f

    def transformSplineVariance(self, sig, t_val):
        t_val = t_val[:, 0].repeat((3, 1)).T
        t_exp = self.t_exponents.repeat((len(t_val), 1))
        t = t_val.pow(t_exp)
        coeff = t.matmul(self.B)
        out = (coeff**2).matmul(sig)
        return out

    def lossFunc(self, mu, sigma, observed_points):
        """
        Parameters
        ----------
        mu : torch.tensor
            Predicted Means of control points. 
            (Number of control points, Number of Dimensions).
        sigma : torch.tensor
            Predicted variances of control points.
            (Number of control points, d*d(+1)//2)
            d = number of dimensions.
        observed_points : torch.tensor
            Observed points.

        Returns
        -------
        loss : torch.tensor
            -Log(probability).

        """
        sigma = self.getSigmaMatrix(sigma)
        num_dims = len(sigma)
        vals = observed_points - mu
        if len(vals.shape) == 2:
            vals = vals.reshape((*mu.shape, 1))
        vals_transpose = vals.transpose(1, 2)
        numerator = (-0.5) * torch.matmul(vals_transpose, torch.matmul(sigma, vals))
        denominator = (torch.det(sigma) ** 0.5) * ((2 * torch.pi) ** (num_dims / 2))
        loss = -1 * torch.sum(numerator / (denominator+1e-6))
        return loss


if __name__ == "__main__":
    NUM_POINTS = 17
    NUM_EPOCHS = 1000
    NUM_SAMPLES = 200
    B_SIZE = 25

    t, points = getToyData(NUM_SAMPLES)
    num_dims = points.shape[1]

    model = YukselSpline(NUM_POINTS, num_dims)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    num_batch = NUM_SAMPLES // B_SIZE

    for epoch in range(1, NUM_EPOCHS + 1):
        # print(f"{model.P=}")
        running_loss = 0.0
        for b in range(num_batch):
            t_curr = t[b * B_SIZE : (b + 1) * B_SIZE]
            y = points[b * B_SIZE : (b + 1) * B_SIZE]

            optimizer.zero_grad()
            y_pred, sigma = model(t_curr)
            
            # loss = model.lossFunc(y_pred, sigma, y)
            loss = loss_fn(y_pred, y)
            loss.backward()

            optimizer.step()

            # print(f"batch_loss = {loss.item()}")
            running_loss += loss.item()

        avg_loss = running_loss / num_batch
        # print(f"{model.P=}")
        print(f"\rCurrent Epoch {epoch}: Loss = {avg_loss}", end="")
        if epoch % 100 == 0:
            print()
            order = torch.argsort(t)
            y_pred = model(t)[0].detach().numpy()
            plt.plot(points[order, 0], points[order, 1])
            P = model.getPoints().detach().numpy()
            plt.scatter(P[:, 0], P[:, 1], c="g", label="Control Points", marker="D")
            plt.legend()
            plot(y_pred[order], f"Predicted at {NUM_POINTS=}, {epoch=}")
    order = torch.argsort(t)
    y_pred = model(t)[0].detach().numpy()
    plt.plot(points[order, 0], points[order, 1])
    P = model.getPoints().detach().numpy()
    plt.scatter(P[:, 0], P[:, 1], c="g", label="Control Points", marker="D")
    plt.legend()
    plot(y_pred[order], f"Final Predictions at {NUM_POINTS=}")
