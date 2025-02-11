import torch
from torch.optim import Optimizer

class SinkGD(Optimizer):
    def __init__(self, params, lr=0.02, num_iterations=5, eps=1e-8):
        """
        SinkGD Optimizer (Inspired by Sinkhorn Normalization)
        
        Args:
        - params: Model parameters
        - lr: Learning rate
        - num_iterations: Number of alternating normalization steps
        - eps: Small constant for numerical stability
        """
        defaults = dict(lr=lr, num_iterations=num_iterations, eps=eps)
        super(SinkGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            num_iterations = group['num_iterations']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data

                # Only apply to 2D weight matrices (skip biases)
                if len(grad.shape) != 2:
                    p.data -= lr * grad
                    continue
                
                # Apply Sinkhorn-like alternating row/column normalization
                for _ in range(num_iterations):
                    # Row-wise normalization
                    row_norms = grad.norm(dim=1, keepdim=True) + eps
                    grad = grad / row_norms * (row_norms.mean())

                    # Column-wise normalization
                    col_norms = grad.norm(dim=0, keepdim=True) + eps
                    grad = grad / col_norms * (col_norms.mean())

                # Apply gradient update
                p.data -= lr * grad

        return loss