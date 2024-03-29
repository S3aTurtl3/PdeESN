import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import torch.sparse

from matplotlib import cm
from collections import OrderedDict
import math


def lorenz(t, X, sigma, beta, rho):
    """Returns output of lorenz equations for a given coordinate and set of 
    dynamical parameters
    X: tuple containing x, y, z coord
    Sigma, beta, rho: the set of dynamical parameters"""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v + u # u=x, v=y, w=z
    return up, vp, wp


def gen_time_series(dt, seconds, initial = (0, 1, 1.05), sig=10, bet=8/3, r=28, tensor= False):
  '''solves the lorenz system with the given parametrs; 
  outputs time series (3-tuple containing lists of x, y, z coords)
  dt: size of timestep
  seconds: the interval of actual time the sequecnce develops over; length of
        resulting time series = seconds//dt
  initial: initial state (x, y, z tuple)
  sig, bet, r (optional): dynamical parameters to use;
                            default is (10, 2.667, 28)'''
  # Integrate the Lorenz equations. HOw to set initial contidion???
  length = int(seconds/dt)
  soln = solve_ivp(lorenz, (0, seconds), initial, args=(sig, bet, r),
                  dense_output=True)
  # Interpolate solution onto the time grid, t.
  t = np.linspace(0, seconds, length)
  pts = soln.sol(t)
  series = np.stack(pts)
  x = series[0].astype('float32')
  y = series[1].astype('float32')
  z = series[2].astype('float32')
  orig = np.stack([x, y, z], axis=1).astype('float32')
  if tensor:
    return torch.from_numpy(orig[:]).unsqueeze(0).cuda(non_blocking=True)
  else:
    return orig

def plot_3D(series, n):
  '''Plots a given time series in 3D
  series: a 2D numpy array where rows are timesteps and cols are spacial
          coordinates x, y, z'''
  x = series[:, 0]
  y = series[:, 1]
  z = series[:, 2]
  # Use Matplotlib 3D projection to plot
  WIDTH, HEIGHT, DPI = 1000, 750, 100
  fig = plt.figure(facecolor='k', figsize=(WIDTH/DPI, HEIGHT/DPI))
  ax = fig.gca(projection='3d')
  ax.set_facecolor('k')
  fig.subplots_adjust(left=0, right=1, bottom=0, top=1) 
  # Make the line multi-coloured by plotting it in segments of length s which
  # change in colour across the whole time series.
  s = 10
  cmap = plt.cm.winter
  for i in range(0,n-s,s):
      ax.plot(x[i:i+ s +1], y[i:i+s+1], z[i:i+s+1], color=cmap(i/n), alpha=0.4)
  # Remove all the axis clutter, leaving just the curve.
  ax.set_axis_off()

  plt.savefig('lorenz.png', dpi=DPI)
  plt.show()


def gen_conditions(numInits, maxCoord):
  '''gen_conditions(numInits, maxCoord) --> Tensor
  Generates a set of random (positive) initial conditions for lorenz systems
  Attempts to create representative sample, creating initial conditions having all 
  possible combinations of magnitudes for their coordinates. Thus, initial conditions 
  that are listed early on in the tensor will tend to have coordinates of lower magnitude,
  while conditions that are listed later on in the tensor will tend to have coordinates of
  higher magnitude.

  Parameters
  ----------
  numInits : int
      the number of initial conditions to generate
  maxCoord : int
      Maximum value that initial conditions'''
  step = math.ceil(maxCoord/numInits)

  # coordinates of initial states are gradually allowed to have greater magnitudes
  conditions = torch.randint(0, step, (1, 3))
  for i in range(step, maxCoord, step):
    conditions = torch.cat((conditions, torch.randint(0, i+step, (1, 3))), dim=0)

  return conditions


# Generate Training Batches
def gen_batches(paramCombos, conditions, cropLen=50*1000, numCrops=1, dt=0.02):
  '''
  Generates batches of lorenz systems to be analyzed using the ESN-Manifold learning method

  Parameters
  ----------
  paramCombos : Tensor
      a 2D array where each row itemizes a set of dynamical parameters to be used in
      generating a lorenz system. num rows = num lorenz systems; col1: sigmas; col2: betas; col3: rhos
  conditions : Tensor
      a 2D tensor containing initial conditions to be used in generating training sequences; each row 
      contains an initial condition. Initial conditions consist of 3 coordinates x, y, z
  cropLen : int
      the length of each training sequence
  numCrops : int
      number of segments to split each lorenz system into (so as to create more training examples)
  dt : float
      the real value of each unit of discretized time'''
  seriesLen = numCrops*cropLen
  maxT = seriesLen*dt 

  difLorenz = [] # contains the training data for modeling each sigma

  for i in range(paramCombos.shape[0]):
    posSig = round(paramCombos[i, 0], 6)
    posBet = round(paramCombos[i, 1], 6)
    posRho = round(paramCombos[i, 2], 6)
    print(f'Generating Lorenz with parameters: {paramCombos[i]}')
    examples = []
    for con in conditions:
      # generate lorenz system with selected sigma
      seq = gen_time_series(dt, maxT, sig=posSig, bet=posBet, r=posRho, initial=con, tensor=True).squeeze(0) # 3-tuple where rows are time, and the cols are x, y, z
      # split the time series into (consecutive) segments
      splitCoords = seq.reshape(numCrops, cropLen, 3) # batch dim x ts x space (x, y, z)
      examples.append(splitCoords)
    batch = torch.cat(examples).unsqueeze(0)
    difLorenz.append(batch)

  return torch.cat(difLorenz)   # shape: space (X, Y, Z) x batch dim (segments of X/Y/Z sequence) x ts