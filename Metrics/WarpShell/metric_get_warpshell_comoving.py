from datetime import datetime
import numpy as np
from Metrics import metric


def metric_get_warpshell_comoving(gridSize: np.array(np.double), worldCenter: np.array(np.double), m: np.double, R1: float,
                                  R2: float, R_buff: float = 0.0, sigma: float = 0.0, smoothFactor: float = 1.0,
                                  vWarp: float = 0.0, doWarp: bool = False, gridScaling: np.array(np.double) =
                                  np.array([1, 1, 1, 1])):
    metric_val: metric.Metric = metric.Metric()
    metric_val.type = "metric"
    metric_val.name = "Comoving Warp Shell"
    metric_val.scaling = gridScaling
    metric_val.coords = "cartesian"
    metric_val.index = "covariant"
    metric_val.date = datetime.today().strftime('%d-%m-%Y')

    #declare radius array
    world_size = np.sqrt((gridSize(2) * gridScaling(2) - worldCenter(2))**2 + (gridSize(3) * gridScaling(3) - worldCenter(3))**2
                         + (gridSize(4) * gridScaling(4) - worldCenter(4))**2)
    r_sample_res = 10**5
    r_sample = np.linspace(0, world_size * 1.2, r_sample_res)

    #construct rho profile
    rho = np.zeros(1, len(r_sample)) + m / (4 / 3 * np.pi * (R2**3 - R1**3)) * (r_sample > R1 & r_sample < R2)
    metric_val.params_rho = rho

    [~, maxR] = min(diff(rho > 0))
    maxR = rsample(maxR)

    #construct mass profile
    M = cumtrapz(rsample, 4 * pi. * rho. * rsample. ^ 2)

    #construct pressure profile
    P = TOVconstDensity(R2, M, rho, rsample)
    Metric.params.P = P

    #smooth functions
    rho = smooth(smooth(smooth(smooth(rho, 1.79 * smoothFactor), 1.79 * smoothFactor), 1.79 * smoothFactor), 1.79 * smoothFactor);
    rho = rho';
    Metric.params.rhosmooth = rho

    P = smooth(smooth(smooth(smooth(P, smoothFactor), smoothFactor), smoothFactor), smoothFactor);
    P = P';
    Metric.params.Psmooth = P

    #reconstruct mass profile
    M = cumtrapz(rsample, 4 * pi. * rho. * rsample. ^ 2)
    M(M < 0) = max(M)

    #save varaibles
    metric_val.params_M = M
    metric_val.params_r_Vec = rsample

    #set shift linevector
    shiftRadialVector = compactSigmoid(rsample, R1, R2, sigma, Rbuff)
    shiftRadialVector = smooth(smooth(shiftRadialVector, smoothFactor), smoothFactor)

    #construct metric using spherical symmetric solution:
    #solve for B
    B = (1 - 2 * G * M / r_sample / c ^ 2)**(-1)
    B[1] = 1

    #solve for a
    a = alphaNumericSolver(M, P, maxR, r_sample)

    #save variables to the metric.params:
    # solve for A from a
    metric_val.params_A = -np.exp(2 * a)
    metric_val.params_B = B
