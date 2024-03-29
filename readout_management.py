import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.decomposition import PCA
import numpy as np
import torch.sparse
import json



def useForWout(weightArr, network):
  '''useForWout(weightArr, network) --> None
  Sets the given network's readout layer to the provided set of weights

  Parameters
  ----------
  weightArr : numpy array
      matrix of weights to be used as the readout layer for the website
  network : ESN
      an object of the ESN class
  '''
  with torch.no_grad():
    weightTns = torch.from_numpy(weightArr).cuda()
    network.Wout.weight = nn.Parameter(weightTns)


def plot_prediction(predicted, actual, var, predHoriz = 0,  saveDir="", desc="", ymin=None, ymax=None):
  '''plot_prediction(predicted, actual, var, desc="", predHoriz=0, ymin=None, ymax=None) --> None
  Plots the networks prediction of a lorenz system's evolution against the actual system
  Note: plot's only a single dimension of the time series' embedding dimension

  Saves images under the filename

  Parameters
  ----------
  predicted : Tensor
      the timeseries prediction; in the shape timesteps x space embedding dimension
  actual : Tensor
      the correct lorenz timeseries; in the shape timesteps x space embedding dimension
  var : int
      the dimension of the timeseries embedding to be plotted
  desc : string (optional)
      a description to include in the file name of the image when saving it
  predHoriz : int (optional)
      indicates a the x coordinate of a vertical line that can be included to 
      denote the prediction horizon of the prediction; default 0
  ymin, ymax : float
      indicates the scale of the y-axis in the plot
  '''
  # Plot the time series
  testFig = plt.figure(figsize=(20,2))
  plt.plot(actual[:, var].cpu().detach().t(), label='Lorenz')
  plt.plot(predicted[:, var].cpu().detach().t(), label='Prediction')
  plt.ylim((ymin, ymax))
  plt.xlabel('Time Steps')
  varStrs = ['X', 'Y', 'Z']
  plt.ylabel(f'{varStrs[var]} Coordinate')
  testFig.legend()
  if predHoriz != 0:
    plt.vlines(predHoriz-1, min(actual[:, var]), max(actual[:, var]))
  
  # Save the figure
  if saveDir:
    testPth = saveDir + f'/PredComparisons/predCompare_Var{var}' + desc
    testFig.savefig(testPth)


def try_examp(model, readoutWeights, paramCombos, weightI, batches, exampI=0, wash=2000, predLen=2000, saveDir="", desc=""): # save implementation for when you add multi-init learning
  '''try_examp(model, readoutWeights, paramCombos, weightI, batches, exampI=0, wash=2000, predLen=2000) --> None
  Finds and plots the prediction horizon of one of the ESNs from a Parameter Extraction Test
  '''
  batch = batches[weightI]
  testWeight = readoutWeights[weightI]
  useForWout(testWeight, model)
  pred, horis = model.findPredHoriz(batch, wash, predLen)
  print(f'Param Combo: {paramCombos[weightI]}')
  print(f'Prediction horizon per example in batch: {horis}')
  prediction = pred[exampI].squeeze(0)
  horizon = horis[exampI]
  testSeq = batch[exampI][wash:wash+predLen]
  plot_prediction(prediction, testSeq, 0, horizon, saveDir, desc)
  plot_prediction(prediction, testSeq, 1, horizon, saveDir, desc)
  plot_prediction(prediction, testSeq, 2, horizon, saveDir, desc)


#approx L2 distance
def apx_L2_Dist(readout1, readout2):
  dist = np.sum(np.abs(readout1-readout2))/np.sum(np.abs(readout2))
  return dist

#Angle Between
def cos_angle_bet(readout1, readout2):
  flat1 = readout1.reshape(-1)
  flat2 = readout2.reshape(-1)
  cosA = np.dot(flat1, flat2)/(np.linalg.norm(flat1)*np.linalg.norm(flat2))
  return cosA


def plot_self_sim(readouts, saveDir="", desc = ""):
  '''plots a self similarity matrix, using dot product distance as a metric
  for comparison
  
  Parameters
  ----------
  readouts : Tensor
      a tensor containing multiple ESN readout layer weights; in the shape numReadouts x output size of readout x input size of readout'''
  selfSim = []
  #Self-Similarity
  for readout in readouts:
    row = []
    for otherOut in readouts:
      row.append(cos_angle_bet(readout, otherOut)) 
    row = np.stack(row)
    selfSim.append(row)

  simMat = np.stack(selfSim)

  #Plot Self Similarity
  figure = plt.figure()
  ax = figure.add_subplot(1, 1, 1)
  matrix = ax.imshow(simMat)

  # add color bar
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  plt.colorbar(matrix, cax=cax)

  if saveDir:
    ParMatPth = saveDir + '/SimMats/CosSim' + desc
    figure.savefig(ParMatPth)


def readout_manifold(method, readouts, numComponents, whichComponents, colors, saveDir="", desc=""):
  '''represents readouts on a 2D manifold'''
  assert method in ['pca', 'isomap', 'mds']
  if method == 'pca':
    mthd = PCA(n_components=numComponents)
  elif method == 'isomap':
    mthd = Isomap(n_components=numComponents)
  else:
    mthd = MDS(n_components=numComponents)
  numReadouts = readouts.shape[0]
  flattened = readouts.reshape((numReadouts, -1))
  components = mthd.fit_transform(flattened)
  fig = plt.figure()
  plt.scatter(components[:,whichComponents[0]], components[:,whichComponents[1]], c=colors)
  plt.xlabel(f"Component {whichComponents[0]}")
  plt.ylabel(f"Component {whichComponents[1]}")
  plt.colorbar()

  if saveDir:
    # save components
    dataPth = saveDir + f'/ManiComponents/{method}components{desc}.npy'
    np.save(dataPth, components)
    # save figure
    figPth = saveDir + f'/{method}/{method}' + desc
    fig.savefig(figPth)
    




class FileManager:
  '''Creates a json file itemizing all the filenames related to the execution of an ESN test'''
  def __init__(self, savePth):
    '''FileManager(savePth) --> FileManager
    Keeps track of all the filenames related to the execution of an ESN test. 
    Saves a json file itemizing all the filenames related to the execution of
    an ESN test

    Parameters
    ----------
    savePth: str
          the jsonFile in which to store the json file with the list of filenames
    testName : str
          descriptive name of the test from which these files originate'''
    self.files = {}
    if savePth[-5:] != '.json':
      raise ValueError('savePth does not have valid file extension (expecting .json)')
    self.nameLog = savePth

  def addFile(self, role, filePth):
    '''FileManager.addFile(role, filePth)
    adds the provided file name to the running list of file names
    
    Parameters
    ----------
    role : string
        the role that the contents of this file play in the experiment
        Valid Roles: ESN, DATA, DYNAMS, READOUTS, PCACOMPS
    fileName : string
        the path 
    '''
    if not role in ['ESN', 'DATA', 'DYNAMS', 'READOUTS', 'PCACOMPS']:
      raise ValueError(f'role {role} is not valid ("role" should be among the options: ESN, DATA, DYNAMS, INITSTATES, READOUTS, PCACOMPS)')
    self.files[role] = filePth
  
  def saveFileList(self):
    '''FileManager.saveFileList()
    saves the fileNames in a json file, where the keys represent file roles and the
    values are the filenames'''
    with open(self.nameLog, "w") as nameLog:
      json.dump(self.files, nameLog, indent = 4, sort_keys=True)


  def locationsOf(self, roles):
    '''FileManager.locationsOf(roles) --> List
    returns a dictionary of a subset of the filenames stored under the FileManager,
    as specified by roles. They keys in the resulting dictionary are the elements of roles,
    and the values are the corresponding filenames 
    
    Parameters
    ----------
    roles : list or tuple of str
        a list of strings denoting the roles of filenames's saved under the FileManager.
        Valid Roles: ESN, DATA, DYNAMS, READOUTS, PCACOMPS
    '''
    desiredFiles = {}
    if not self.files:
      # if self.files is empty, load from json file
      with open(self.nameLog, "r") as nameLog:
        self.files = json.load(nameLog)
    # dictionary of output desired filenames
    for role in roles:
      desiredFiles[role] = self.files[role]

    return desiredFiles