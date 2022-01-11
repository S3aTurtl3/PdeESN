from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from readout_management import FileManager, plot_self_sim, readout_manifold

def extract_and_analyze(esn, readouts, paramCombos, method, numComponents, saveDir, fileMan):
  '''extract_and_analyze(esn, readouts, paramCombos, method, numComponents, saveDir, fileMan)
  Analyzes a set of readouts in order to extract the dynamical parameters
  associated with them
  
  Parameters
  ----------
  esn         : ESN
      the ESN used in the dynamical parameter extraction method 
  readouts    : numpy array
      readout layers corresponding to the provided ESN
  paramCombos : numpy array
      the dynamical parrameters corresponding to with each readout in 'readouts';
      both arrays must have the same shape
  method      : str
      manifold learning method to use; either 'isomap', 'pca', 'mds', or 'spectral'
  numComponents : int
      the number of components to be generated using the specified manifold 
      learning method
  saveDir     : str
      the directory in which to save relevant files
  fileMan     : FileManager
      file manager responsible for recording the names of key files used by this function'''
  desc = fileMan.getLogName()
  # self-similarity matrix
  if readouts.shape[0] != paramCombos.shape[0]:
    raise ValueError(f"dim 0 of 'readouts' (shape is {readouts.shape}) should be the same as dim 0 of 'paramCombos' (shape is {paramCombos.shape}).")
  simPth = plot_self_sim(readouts, saveDir, desc) # how to fix so terminal doesn't make pop-up windows..!!!
  fileMan.addFile("SIMMAT", simPth)
  
  # manifold learning cross sections, colored by dynamical parameters
  components, maniPth = readout_manifold(readouts, paramCombos, method, numComponents, saveDir, fileMan)

  # label figures with the filename of the file manager
  # return table of best/worst prediction horizon vs parameter (save in PredComparisons)
  pass