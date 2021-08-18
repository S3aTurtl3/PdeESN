from os import listdir
import numpy as np
import base64
from Lorenz import gen_conditions, gen_batches

def extraction_test_data(numConditions, maxCoord, seriesLen, sigRange, betRange, rhoRange, increments, numCrops, saveDir, fileMan, loadMan=None):
  '''extraction_test_data(numConditions, maxCoord, seriesLen, sigRange, betRange, rhoRange, increments, numCrops, saveDir, fileMan, loadMan=None) --> Tensor
  Generates and saves batches (in the shape numBatches X batchLen X timeSequenceLen X embeddingDimension) of lorenz systems to be analyzed using the ESN-Manifold learning method
  numConditions
  
  Parameters
  ----------
  numConditions   : int
      the number of initial conditions to be used in generating examples for a 
      batch of training data
  maxCoord  : int
      the maximum possible value for coordinates in an initial condition (used in 
      generating examples for a batch of training data) 
  seriesLen : int
      the length of each sequence in the data
  numCrops  : int
      number of segments to split each lorenz system into (so as to create more training examples)
  sigRange  : two-tuple of floats (rounded to 6 decimal places)
      specifies the range over which to vary the dynamical parameter Sigma
  betRange  : two-tuple of floats (rounded to 6 decimal places)
      specifies the range over which to vary the dynamical parameter Beta
  rhoRange  : two-tuple of floats (rounded to 6 decimal places)
      specifies the range over which to vary the dynamical parameter Rho
  increments  : three-tuple of floats
      specifies the increment between items in each set of dynamical parameters; first, second,
      and third floats correspond to stepsizes for ranges of sigma, beta, and rho, respectively
  fileMan     : FileManager
      file manager responsible for recording the names of key files used by this function 
  loadMan     : FileManager (optional)
      a file manager from which relevant files should be loaded instead of being generated
  '''
  data = None
  paramCombos = None
  minSig = round(sigRange[0], 6) # later change so you use a list instead of all these vars...
  maxSig = round(sigRange[1], 6)
  minBet = round(betRange[0], 6)
  maxBet = round(betRange[1], 6)
  minRho = round(rhoRange[0], 6)
  maxRho = round(rhoRange[1], 6)


  # If loadMan is specified, load from there
  if loadMan is not None:
    paths = loadMan.locationsOf(['DATA', 'DYNAMS'])
    dataPth = paths["DATA"]
    dynamPth = paths["DYNAMS"]
    fileMan.addFile("DATA", dataPth)
    fileMan.addFile("DYNAMS", dynamPth)
    data = np.load(dataPth)
    paramCombos = np.load(dynamPth)
    return data, paramCombos

  # If this data has already been generated, load from there
  dataDesc = f'numInits{numConditions}_maxCoord{maxCoord}_seriesLen{seriesLen}_sigmas{(minSig, maxSig)}_betas{(minBet, maxBet)}_rhos{(minRho, maxRho)}_increments{increments}_numCrops{numCrops}'
  dynamDesc = f'sigmas{(minSig, maxSig)}_betas{(minBet, maxBet)}_rhos{(minRho, maxRho)}_increments{increments}'
  dataDesc = base64.b64encode(dataDesc.encode('utf-8')).decode('utf-8')
  dynamDesc = base64.b64encode(dynamDesc.encode('utf-8')).decode('utf-8')
  existingDataFiles = listdir(f'{saveDir}/Data')
  existingDynamFiles = listdir(f'{saveDir}/Dynams')
  for name in existingDataFiles:
    if dataDesc in name:
      dataPth = f'{saveDir}/Data/{name}'
      fileMan.addFile("DATA", dataPth)
      data = np.load(dataPth)
      exists = True
      break
  if data is not None:
    for name in existingDynamFiles:
      if dynamDesc in name:
        dynamPth = f'{saveDir}/Dynams/{name}'
        fileMan.addFile("DYNAMS", dynamPth)
        paramCombos = np.load(dynamPth)
        break
    if paramCombos is not None:
      return data, paramCombos
      
  
  # Create initial conditions to be used in generating examples for a batch of training data
  conditions = gen_conditions(numConditions, maxCoord)

  # Generate combinations of dynamical parameters to be used producing the lorenz sequences in each batch (1 set of dynamics per batch)
  sigmas = np.arange(minSig, maxSig, increments[0]) # automate this reshaping process to work no matter how many dynam params (not nested for loops)...
  betas = np.arange(minBet, maxBet, increments[1])
  rhos = np.arange(minRho, maxRho, increments[2])

  betaByExamp = np.repeat(np.expand_dims(betas, axis=1), betas.shape[0], axis=1)
  rhoByExamp = np.tile(rhos, (rhos.shape[0], 1))
  
  paramList = []
  for i in range(sigmas.shape[0]):
    sigmaByExamp = np.tile(sigmas[i],(betas.shape[0], rhos.shape[0]))
    paramList.append(np.stack((sigmaByExamp, betaByExamp, rhoByExamp), axis=2).reshape(-1, 3))
    
  paramCombos = np.concatenate(paramList)

  # Generate batches of training data
  data = gen_batches(paramCombos, conditions, cropLen=seriesLen, numCrops=numCrops, dt=0.02)

  # Save data
  dataPth = saveDir + f'/Data/Data{dataDesc}.npy'
  dynamPth = saveDir + f'/Dynams/Dynam{dynamDesc}.npy'
  np.save(dataPth, data.cpu().numpy())
  np.save(dynamPth, paramCombos)
  fileMan.addFile("DATA", dataPth)
  fileMan.addFile("DYNAMS", dynamPth)
  return data, paramCombos
