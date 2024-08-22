"""
GETENERGYTENSOR: Converts the metric into the stress energy tensor

INPUTS:
metric - A metric struct

tryGPU - A flag on whether or not to use GPU computation (0=no, 1=yes)

diffOrder - Order of finite difference, either 'second' or 'fourth'

OUTPUTS:
energy - energy tensor struct
"""
def get_energy_tensor(metric, try_gpu: bool = False, diff_order: str = 'fourth'):

