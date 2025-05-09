from typing import Dict, Union

import torch


class BaseTorchTensor:
    def __init__(self, device: Union[str, torch.device] = "cpu"):
        """Initialize solver.

        Parameters
        ----------
        device : str or torch.device
            Device to use for computations
        """
        self.device = torch.device(device)

    def _ensure_tensor(self, x: Union[torch.Tensor, float]) -> torch.Tensor:
        """Ensure input is a tensor on the correct device.

        Parameters
        ----------
        x : torch.Tensor or float
            Input to convert

        Returns
        -------
        torch.Tensor
            Tensor on the correct device
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)
        elif x.device != self.device:
            x = x.to(self.device)
        return x
