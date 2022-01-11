from readout_management import FileManager
from Data_Generation_Script import extraction_test_data
from  Training_Script import train_to_extract
from Extraction_and_Analysis_Script import extract_and_analyze
import torch


saveDir = 'INSERT TARGET DIRECTORY'


# Get Data
manny =  FileManager(saveDir + '/FileLogs/testingTrainPlotScript.json')
data, paramCombos, dataDesc = extraction_test_data(6, 50, 50*1000, (10, 10.3), (2.566667, 2.766667), (27.9, 28.1), (3, 3, 3), 1, saveDir, manny)

# Train
manny.saveLog()
if isinstance(data, np.ndarray):
  data = torch.from_numpy(data).cuda()
else:
  data.cuda()
manny.saveLog()
allOutNP, model = train_to_extract(data, dataDesc, 2000, saveDir, (), manny)

# Analyze
extract_and_analyze(model, allOutNP, paramCombos, 'mds', 3, saveDir, manny)
manny.saveLog()