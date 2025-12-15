import torch

class Objective:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

class Quadratic(Objective):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x.square().sum()

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return x
