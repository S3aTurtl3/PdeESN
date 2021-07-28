import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch.sparse



def useForWout(weightArr, network):
  with torch.no_grad():
    weightTns = torch.from_numpy(weightArr).cuda()
    network.Wout.weight = nn.Parameter(weightTns)


def plot_prediction(predicted, actual, var, desc="", predHoriz = 0, ymin=None, ymax=None):
  #both pred and actual are tensors must be squeezed
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
  

  toSave = input('Save this img? ')
  if toSave.lower() == 'yes please':
    desc = input('desc: ')
    testPth = esnDir + 'LorenzLinReg/' + f'predCompareVar{var}' + desc
    testFig.savefig(testPth)


def try_examp(model, readoutWeights, weightI, batches, wash=2000, predLen=2000): # save implementation for when you add multi-init learning
  # Select a readout layer and the batch of examples it was trained on
  batch = batches[weightI]
  testWeight = readoutWeights[weightI]
  useForWout(testWeight, model)
  pred, horis = model.findPredHoriz(batch, wash, predLen)
  print(horis)
  prediction = pred[weightI].squeeze(0)
  horizon = horis[weightI]
  testSeq = batch[weightI][wash:wash+predLen]
  plot_prediction(prediction, testSeq, 0, predHoriz=horizon)
  plot_prediction(prediction, testSeq, 1, predHoriz=horizon)
  plot_prediction(prediction, testSeq, 2, predHoriz=horizon)


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


def plot_self_sim(readouts):
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

  toSaveMat = input("save matrix?")
  if toSaveMat == "yes please!":
    desc = input(f"description (e.g. sig{sigma}bet{int(beta)}rho{rho}lorenz: ")
    ParMatPth = esnDir + 'LorenzLinReg/' + 'MFSCosMat' + f'sig{sigma}bet{int(beta)}rho{rho}lorenz'
    figure.savefig(ParMatPth)