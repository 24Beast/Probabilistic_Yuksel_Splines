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
    def __init__(self, num_points, num_dims):
        """
        Parameters
        ----------
        num_points : int
            Number of points defining the spline. Can only be of the form 2n+1.
        input_size : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(YukselSpline, self).__init__()
        self.num_dimensions = num_dims
        self.num_points = num_points
        self.num_splines = num_points - 2
        self.initWeights()
        self.initLayers()

    def initWeights(self):
        # Defining Variables for Quadratic Bezier Splines
        self.t_exponents = torch.tensor([i for i in range(3)])
        self.B = torch.tensor([[1, 0, 0], [-2, 2, 0], [1, -2, 1]]).float()

    def initLayers(self):
        # Potential Model Layers
        self.linear1 = nn.Linear(1, 16)
        self.linear2 = nn.Linear(16, 64)
        self.linear3 = nn.Linear(64, 256)
        self.linear4 = nn.Linear(256, 64)
        self.linear5 = nn.Linear(64, 16)
        self.linear6 = nn.Linear(16, self.num_dimensions)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def layerPass(self, x):
        out = self.linear1(x)
        out = self.sigmoid(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        out = self.relu(out)
        out = self.linear5(out)
        out = self.relu(out)
        out = self.linear6(out)
        out = torch.cumsum(out, axis=1)
        return out

    def forward(self, x):
        t = x.clone()
        pos = torch.arange(1, self.num_points + 1) / 1.0
        pos = pos.reshape((self.num_points, 1))
        P = self.layerPass(pos)
        return self.transform(P, t)

    def transform(self, P, t_val):
        """
        Parameters
        ----------
        points : torch.tensor
            Torch tensor of the form (Batch.
        t_val : torch.tensor
            Time data. Assumption t_val ranges from 0-1.

        Returns
        -------
        probs : torch.tensor
            Probabilities in form of gaussian (Mean, Variance).

        """
        P_new = P[:3]
        # Use transform spline here
        probs = torch.zeros((len(t_val), self.num_dimensions))
        for i in range(self.num_splines - 1):
            d = ((t_val * (self.num_splines - 1)) - i) * 0.5
            d = d.unsqueeze(1)
            choice = (0 <= d) * (d <= 0.5)
            choice = choice * 1
            probs = (1 - choice) * probs
            P_temp = P_new.clone()
            P_new[0] = (P_temp[0] + P_temp[2]) / 4 + P_temp[1] / 2
            P_new[2] = P[i + 3]
            P_new[1] = 2 * (P_temp[2] - (P_new[0] + P_new[2]) / 4)
            f_prev = self.transformSpline(P_temp, d + 0.5)
            f_curr = self.transformSpline(P_new, d)
            probs += choice * (
                (
                    ((np.cos(np.pi * d) ** 2) * f_prev)
                    + ((np.sin(np.pi * d) ** 2) * f_curr)
                )
            )

        return probs

    def transformSpline(self, p, t_val):
        t_val = t_val[:, 0].repeat((3, 1)).T
        t_exp = self.t_exponents.repeat((len(t_val), 1))
        t = t_val.pow(t_exp)
        f = t.matmul(self.B).matmul(p)
        """
        print(f"{t_val.shape=}")
        print(f"{f.shape=}")
        """
        return f

    def getPoints(self):
        pos = torch.arange(1, self.num_points + 1) / 1.0
        pos = pos.reshape((self.num_points, 1))
        P = self.layerPass(pos)
        return P


if __name__ == "__main__":
    NUM_POINTS = 15
    NUM_EPOCHS = 10000
    NUM_SAMPLES = 200
    B_SIZE = 25
    SHUFFLE = True
    ERROR_FACTOR = 5e-2

    t, points = getToyData(NUM_SAMPLES, SHUFFLE, ERROR_FACTOR)
    num_dims = points.shape[1]

    model = YukselSpline(NUM_POINTS, num_dims)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

    num_batch = NUM_SAMPLES // B_SIZE

    for epoch in range(1, NUM_EPOCHS + 1):
        # print(f"{model.P=}")
        running_loss = 0.0
        for b in range(num_batch):
            t_curr = t[b * B_SIZE : (b + 1) * B_SIZE]
            y = points[b * B_SIZE : (b + 1) * B_SIZE]

            optimizer.zero_grad()
            y_pred = model(t_curr)

            loss = loss_fn(y_pred, y)
            loss.backward()

            optimizer.step()

            # print(f"batch_loss = {loss.item()}")
            running_loss += loss.item()

        avg_loss = running_loss / num_batch
        # print(f"{model.P=}")
        if epoch % 100 == 0:
            print(f"\rCurrent Epoch {epoch}: Loss = {avg_loss}", end="")
        if epoch % 500 == 0:
            print()
            order = torch.argsort(t)
            y_pred = model(t).detach().numpy()
            plt.plot(points[order, 0], points[order, 1])
            P = model.getPoints().detach().numpy()
            plt.scatter(P[:, 0], P[:, 1], c="g", label="Control Points", marker="D")
            plt.legend()
            plot(y_pred[order], f"Predicted at {NUM_POINTS=}, {epoch=}")
    order = torch.argsort(t)
    y_pred = model(t).detach().numpy()
    plt.plot(points[order, 0], points[order, 1])
    P = model.getPoints().detach().numpy()
    plt.scatter(P[:, 0], P[:, 1], c="g", label="Control Points", marker="D")
    plt.legend()
    plot(y_pred[order], f"Final Predictions at {NUM_POINTS=}")
