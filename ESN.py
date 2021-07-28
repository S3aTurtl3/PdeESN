import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from sklearn.linear_model import LinearRegression, Ridge
import torch.sparse



def load_model(pth):
  '''Returns a new ESN instance that uses the same input layer and reservoir initialization of
  another Echo State Network; readout layer weights are not loadded into the model.
  pth: path to the pre-trained network'''
  # Determine network's configuration
  params = []
  hyperParamNames = ['InSize', 'outSize', 'res', 'avDeg', 'sig', 'rad', 'leak']
  for i, paramName in enumerate(hyperParamNames):
    startI = pth.index(paramName)
    startI += len(paramName)
    if i == len(hyperParamNames) - 1:
      endI = -1
    else:
      endI = pth.index(hyperParamNames[i+1])
    param = pth[startI:endI]
    params.append(eval(param))
  # Initialize the network using the same hyperparameters and weights as the given model
  network = ESN(*params)
  weights = torch.load(pth)
  weights.pop('Wout.weight', None) # readout layer weights are not loaded
  network.load_state_dict(weights, strict=False)
  network.eval()
  network.cuda()
  return network

class ESN(nn.Module):
    def __init__(self, input_size=3, output_size=3, reservoir_size=300, avg_degree=6, sigma=1., radius=0.1, leakage=1.):
        '''Creates an echo state network (ESN) with leaky integrator perceptrons
        input_size:     the size of the embedding dimension of the input system state
        output_size:    the size of the embedding dimension of the output system state
        reservoir_size: number of perceptrons in the reservoir
        avg_degree:     average degree of connectivity; determines reservoir sparcity
        sigma:          scaling factor; inputs to the ESN are scaled by this value
        radius:         spectral radius
        leakage:        percentage (between 0-1) leaking rate of the perceptrons (a leaking rate of 1 results in no leakage)'''
        super(ESN, self).__init__()
        self.input_size = input_size                
        self.output_size = output_size              
        self.reservoir_size = reservoir_size
        self.avg_degree = avg_degree
        self.sigma = sigma
        self.radius = radius
        self.leakage = leakage
        self.abs_eigs = 0                   # arbitrary initial value
        self.thresh = np.array([5, 10, 5])  # max divergance threshold for predictions on the Lorenz System

        self.sparsity = self.avg_degree/self.reservoir_size

        #Create the reservoir, input layer, and readout layer 
        self._init_reservoir(self.reservoir_size, self.sparsity, self.radius)
        self._init_Win(self.reservoir_size, self.input_size, self.sigma)
        self.Wout = nn.Linear(self.reservoir_size, self.output_size, bias=False)
        # Initialize the hidden state
        self.register_buffer("h0", torch.zeros(reservoir_size))
        self.allOuts = []

    def _init_reservoir(self, size, sparsity, radius):
        '''initializes the reservoir of an ESN. The reservoir is represented by
        a sparse adjacency matrix...(etc...reference the build notes)
        size: number of perceptrons in the reservoir
        spasity: the sparcity of the reservoir'''
        A = torch.Tensor(size**2)
        A.uniform_(-1, 1)
        if sparsity < 1:
            zero_weights = torch.randperm(size**2)
            zero_weights = zero_weights[:int(size**2 * (1 - sparsity))]
            A[zero_weights] = 0
        A = A.view(size, size)
        abs_eigs = (torch.eig(A)[0] ** 2).sum(1).sqrt()
        self.abs_eigs = abs_eigs
        self.register_buffer('A', A * (radius / torch.max(abs_eigs)))

    def _init_Win(self, reservoir_size, input_size, sigma):
        '''initilizes the input layer of the ESN'''
        self.register_buffer('Win', (self.radius / torch.max(self.abs_eigs)*(2*torch.rand(reservoir_size, input_size) - 1)))

    def forward(self, input_sequence, washout_len, noise_std=0., returnHistory=False):
        '''Forward pass through the ESN w/o feedback loop;
        Returns tensor contaning ESN forecasts at each timestep. (OR the history of reservoir hidden states)
        System values are given as input to the ESN at each timestep
        input_sequence:   time series with the following shape: batch_size x series_len x embedding dimension
        washout_len:      identifies how many of the first few timesteps should be used as a washout
        noise_std:        std of the noise (normaly distributed around 0) to add to the input; default 0
        returnHistory:    when True, function outputs the history of reservoir hidden states; when False,
                      function ouputs a tensor contaning ESN forecasts at each timestep.'''
        batch_size = input_sequence.shape[0]
        seq_len = input_sequence.shape[1]
        # Pass inputs through the input layer
        input = F.linear(input_sequence + noise_std*torch.randn_like(input_sequence), self.Win)

        # System values are given as input to the ESN at each timestep; hidden states are updated
        h = self.h0.unsqueeze(0).expand(batch_size, -1)
        h_list = [] # history of reservoir hidden states
        for i in range(seq_len):
            # Update the hidden state
            h = (1-self.leakage)*h + self.leakage*torch.tanh(F.linear(h, self.A) + input[:,i])
            if i >= washout_len -1:
                h_list.append(h)
        
        # Returns history of hidden states
        h_list = torch.stack(h_list, dim=1)
        if returnHistory:
          return h_list

        # Passes hidden states through the readout layer to get sequence of predictions
        return self.Wout(h_list)


    def diverging(self, predPt, realPt):
      '''Returns a numpy array of booleans, each corresponding to a feature of predPt; 
      Each boolean indicates whether the corresponding feature of predPt diverges 
      from realPt by more than the acceptable threshold (self.thresh)

      predPt: the point predicted by the ESN
      realPt: what the correct ESN output should have been'''
      divergence = np.greater(np.absolute(np.subtract(predPt, realPt)), self.thresh)
      return divergence

    def findPredHoriz(self, input_sequence, washout_len, prediction_len = 0):
      '''Returns...
        * number of consecutive timesteps for which ESN predictions are accurate within a threshold (self.thresh).
          i.e. it returns the ESN's prediction
        * sequence of predictions outputted by the ESN (the length of this sequence is determined either by the
          batch example on which the ESN achieves the greatest prediction horizon OR by user input)

      A prediction horizon is the number of consecutive timesteps (excluding the washout) 
      for which the ESN makes accurate (errors are within a threshold (self.thresh)) 
      predictions of a system's state before the threshold for acceptable error is first exceeded.
      After the washout, the ESN makes predictions in a feedback loop, like that of the evaluate() method.

      input_sequence:   time series with the following shape: batch_size x series_len x embedding dimension
      washout_len:      identifies how many of the first few timesteps should be used as a washout
      prediction_len:   (optional) number of timesteps the esn should forecast; if unspecified, 
                                  forward passes stop when all prediciton horizons have been found'''
      horizons = torch.zeros(input_sequence.shape[0]) # stores ESN prediction horizon for each provided time series

      # Identify prediction stopping point
      stopAtHorizon = False
      # if no prediction length was specified, forward passes stop when all prediciton horizons have been found
      if prediction_len == 0:
        prediction_len = input_sequence.shape[1] - washout_len
        stopAtHorizon = True
     

      batch_size = input_sequence.shape[0]
      # Pass inputs through the input layer
      input = F.linear(input_sequence, self.Win)
      
      # System values are given as input to the ESN only during the washout; after the washout,
      # the ESN uses its previous predictions as input.
      h = self.h0.unsqueeze(0).expand(batch_size, -1)
      out_list = []
      out = input_sequence[:, washout_len + 1]
      for i in range(washout_len+prediction_len-1):
        if i < washout_len:
            # Update the hidden state
            h = (1-self.leakage)*h + self.leakage*torch.tanh(F.linear(h, self.A) + input[:,i])
        else:
            if i == washout_len:
                out = self.Wout(h)
                out_list.append(out)
            # Predictions are made either until... 
                # (A) the ESN's forecast is of a specified length (prediction_len)
                # (B) (if no length was specified) the ESN's predictions deviate from the actual system state 
                  #(in input_sequence) by more than the values in self.thresh
            divergence = self.diverging(out.detach().cpu().numpy(), input_sequence[:, i].cpu().numpy())
            if divergence.any():
              divergenceByExamp = np.any(divergence, axis=1)
              indices = np.where(divergenceByExamp == True)
              # Record the location of the first instance of divergence in the sequence
              for index in indices[0]:
                if horizons[index] == 0:
                  horizons[index] = len(out_list)
              # If no prediction length was specifed, forward passes stop when all prediciton horizons have been found
              if stopAtHorizon and horizons.all():
                break
            # Update the hidden state
            h = (1-self.leakage)*h + self.leakage*torch.tanh(F.linear(h, self.A) + F.linear(out, self.Win))
            # Pass hidden state through the readout layer
            out = self.Wout(h)
            out_list.append(out)
      
      outSequence = torch.stack(out_list, dim=1)   
        
      return (outSequence, horizons)


    def evaluate(self, input_sequence, washout_len,  prediction_len):
        '''Returns sequence of predictions outputted by the ESN; After the washout, the ESN makes predictions in a feedback loop,
        using the predictions of previous timesteps as input for future timesteps.
        
        input_sequence:   time series with the following shape: batch_size x series_len x embedding dimension
        washout_len:      identifies how many of the first few timesteps should be used as a washout
        prediction_len:   number of timesteps the esn should forecast'''
        input = F.linear(input_sequence, self.Win)

        batch_size = input_sequence.shape[0]

        h = hidStart
        out_list = []
        # System values are given as input to the ESN during the washout period; 
        # After the washout the ESN makes predictions in a feedback loop
        for i in range(washout_len+prediction_len-1):
            # Update the hidden state during the washout period
            if i < washout_len:
                h = (1-self.leakage)*h + self.leakage*torch.tanh(F.linear(h, self.A) + input[:,i])
            else:
                if i == washout_len:
                    out = self.Wout(h)
                    out_list.append(out)
                # Update the hidden state, using the previous prediction as input
                h = (1-self.leakage)*h + self.leakage*torch.tanh(F.linear(h, self.A) + F.linear(out, self.Win))
                # Pass hidden state through the readout layer
                out = self.Wout(h)
                out_list.append(out)
        
        return torch.stack(out_list, dim=1)

    def linRegTrain(self, data, wash=2000):
      # data is of shape num batches x batch dim x ts x space
      self.allOuts = []

      for batch in data:
        hidStates = self.forward(batch[:, :-1], wash, returnHistory=True) # minus last timestep so don't go over dim(desired); shape: batch dim x (ts - wash) x reservoir size
        hidStates = hidStates.view(-1, hidStates.shape[-1]) # res = reservoir size (set in the ESN section...should encapsulate this process in an object...)
        print("shape of hid: ", hidStates.size()) 
        desired = batch[:, wash:]
        desired = desired.reshape(-1, desired.shape[-1])
        
        ridge_reg = Ridge(alpha=1e-2, fit_intercept=False)
        ridge_reg.fit(hidStates.cpu(), desired.cpu())
        altWeightArr = np.float32(ridge_reg.coef_)
        self.allOuts.append(altWeightArr)
      return np.stack(self.allOuts)

    def save_model(self, dir):
      '''Saves the network weights to the specified directory location
      dir: the directory in which to save the network weights'''
      pth = f'{dir}/ESNconfigInSize{self.input_size}outSize{self.output_size}res{self.reservoir_size}avDeg{self.avg_degree}sig{self.sigma}rad{self.radius}leak{self.leakage}'
      torch.save(self.state_dict(), pth)

    
      