import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from torch.nn import functional as F
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
import torch.sparse
import math
from random import randint

from matplotlib import cm
from collections import OrderedDict


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


def plot_3D(series):
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
      ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color=cmap(i/n), alpha=0.4)
  # Remove all the axis clutter, leaving just the curve.
  ax.set_axis_off()

  plt.savefig('lorenz.png', dpi=DPI)
  plt.show()


# Generate initilal Conditions
def gen_conditions(numInits, maxCoord):
  step = math.ceil(maxCoord/numInits)

  # initial states are allowed to get gradually further away; max coord being 50
  conditions = torch.randint(0, step, (1, 3))
  for i in range(step, maxCoord, step):
    conditions = torch.cat((conditions, torch.randint(0, i+step, (1, 3))), dim=0)

  return conditions


# Generate Training Batches
def gen_batches(paramCombos, conditions, cropLen=50*1000, numCrops=1, dt=0.02):
  '''
  paramCombos: a 2D array where each row itemizes a set of dynamical parameters to be used in
              generating a lorenz system. num rows = num lorenz systems; col1: sigmas; col2: betas; col3: rhos'''
  seriesLen = numCrops*cropLen
  maxT = seriesLen*dt 

  difLorenz = [] # contains the training data for modeling each sigma

  for i in range(paramCombos.shape[0]):
    posSig = round(paramCombos[i, 0], 6)
    posBet = round(paramCombos[i, 1], 6)
    posRho = round(paramCombos[i, 2], 6)
    print(paramCombos[i])
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