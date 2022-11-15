# Import necessary libraries

import pandas as pd
import numpy as np
import collections.abc
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union
#from typeguard import check_argument_types
import os
import contextlib
import wave

import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from itertools import permutations
import torch_optimizer as optim

from transformers import Wav2Vec2FeatureExtractor, WavLMForAudioFrameClassification
from librosa import load
#import soundfile as sf
import argparse
import logging
from scipy.signal import medfilt

from transformers import Wav2Vec2FeatureExtractor, WavLMForAudioFrameClassification
import torch
import time


start = time.time()
print("Started")

# Load the feature_extracture, pre-trained model 

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sd')
model = WavLMForAudioFrameClassification.from_pretrained('microsoft/wavlm-base-plus-sd')


# Freeze all parameters
param_before = []
for p in model.parameters():
  param_before.append(p)
  p.requires_grad=False

# Unfreeze the final classifier layer weights
model.classifier.weight.requires_grad = True
model.classifier.bias.requires_grad = True


# bat=[]
# Loss_train = []
# weights = []

class SimpleBrain(sb.Brain):

  def compute_forward(self, batch, stage):

    # Compute the forward pass
    batch[0] = batch[0].to('cuda')
    c = self.modules.model(batch[0])

    return c[0]

    
  def compute_objectives(self, predictions, batch, stage):

    batch[1] = batch[1].to('cuda')
    label_o=batch[1]


    Loss_total = []

    # Following loop is to compute the PIT Loss for all the examples in the batch

    for i in range(len(label_o)):

      bce = sb.nnet.losses.bce_loss

      label = torch.squeeze(label_o[i:i+1,:,:])
      label = (label>0.5).long()
      # Compute all possible premutations of the labels
      label_perms = [label[..., list(p)] for p in permutations(range(label.shape[-1]))]

      prediction = predictions[i]
      # Upsample the prediction 
      prediction = prediction.repeat_interleave(320,dim=0)

      # Compute the loss against each possible permutation
      Loss = []
      for j in range(len(label_perms)):
        Loss.append(bce(prediction,label_perms[j]))

      # Index for which loss is minimum
      min_index = Loss.index(min(Loss))

      # Append the loss which is least to update weights
      Loss_total.append(bce(prediction,label_perms[min_index]))

    # Take the mean over all examples in the batch  
    loss = torch.stack(Loss_total).mean()
    Loss_train.append(loss)

    return loss


def main(device='cuda'):


    data=[]

    # Load the input and targets

    for filename in os.listdir('twospk-train/train_testaudio/test-pt(5sec)'):

      a = torch.load('twospk-train/train_testaudio/test-pt(5sec)/'+filename)

      target = torch.load('twospk-target/targets_test_five/targets_test_five/content/outputs/'+filename)
      target = torch.narrow(target,0,0,79680)



    print("Training starting")

    # Load model hyperparameter file

    hparams_file = "hyp.yaml"
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Trainer initialisation

    brain = SimpleBrain({"model": model},hparams['opt_class'],hparams,run_opts={'device':device},)
    
    # Training Loop
    
    brain.fit(range(hparams["N_epochs"]), data,train_loader_kwargs=hparams["dataloader_options"])
    print(time.time()-start," sec")
    print("The end")


    # Get predictions for test audio

    for test_file in os.listdir('twospk-train/train_testaudio/test-pt(5sec)'):
      
      a = torch.load("twospk-train/train_testaudio/test-pt(5sec)/"+test_file)
      a = a.reshape(1,80000)
      a = torch.from_numpy(a)
      batch = [a]
      output = brain.compute_forward(batch,stage=2)
      output = output[0]
      print(output.shape)
      torch.save(output,'outputs/out_test_five/'+test_file.split('.')[0]+'.pt')

    
if __name__ == "__main__":
    main()
    
def test_error(device):
    main(device)
    
