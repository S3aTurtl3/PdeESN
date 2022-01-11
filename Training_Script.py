from os import listdir
from ESN import load_model, ESN
import numpy as np
import torch
from readout_management import FileManager

def train_to_extract(data, dataDesc, washLen, saveDir, esnParams, fileMan, loadMan=None):
  '''train_to_extract(data,  washLen, saveDir, esnParams, loadMan=None) --> Tensor
  Trains an ESN on each batch of the data, and returns a Tensor (in the shape numBatches X embeddingDimension x reservoirSize) containing the readout 
  layers of each ESN

  data      : Tensor
      data to be analyzed using the ESN-Manifold learning method
      (in the shape numBatches X batchLen X timeSequenceLen X embeddingDimension)
  dataDesc  : str
      description of the data accordning to the naming convention (encoded in base64)
  washLen   : int
      number of timesteps to be used as washout during training
  saveDir   : string
      the directory in which to save
  esnParams : tuple of numbers
      esn hyperparameters
  fileMan   : FileManager
      file manager responsible for recording the names of key files used by this function 
  loadMan   : FileManager (optional)
      a file manager from which relevant files should be loaded instead of being generated
  '''
  # If loadMan is specified, load ESN from there 
  if loadMan is not None:
    # load previous ESN
    paths = loadMan.locationsOf(['ESN', 'READOUTS'])
    esnPth = paths['ESN']
    outsPth = paths['READOUTS']
    model = load_model(esnPth)
    allOutNP = np.load(outsPth)
    fileMan.addFile("ESN", esnPth)
    fileMan.addFile("READOUTS", outsPth)
    return allOutNP, model
  else:
    # Create and save new ESN
    torch.manual_seed(2)
    model = ESN(*esnParams)
    model.cuda()
    path = model.save_model(saveDir)
    fileMan.addFile("ESN", path)
    # Train
    data = data.cuda()
    readouts = model.linRegTrain(data, washLen)

    #Save the list of readout layer weights
    allOutPth = saveDir + f'/Readouts/ReadoutWsh{washLen}_From{dataDesc}.npy'
    np.save(allOutPth, readouts)
    fileMan.addFile("READOUTS", allOutPth)
    return readouts, model