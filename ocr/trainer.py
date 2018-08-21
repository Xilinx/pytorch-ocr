#   Copyright (c) 2018, Xilinx, Inc.
#   All rights reserved.
# 
#   Redistribution and use in source and binary forms, with or without 
#   modification, are permitted provided that the following conditions are met:
#
#   1.  Redistributions of source code must retain the above copyright notice, 
#       this list of conditions and the following disclaimer.
#
#   2.  Redistributions in binary form must reproduce the above copyright 
#       notice, this list of conditions and the following disclaimer in the 
#       documentation and/or other materials provided with the distribution.
#
#   3.  Neither the name of the copyright holder nor the names of its 
#       contributors may be used to endorse or promote products derived from 
#       this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#   OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
#   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
#   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import time
import random
import os
from datetime import datetime

from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch.nn.init as init
from torch import autograd, zeros
from torch import zeros
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from ocr import Logger
from ocr import OCRModule
from ocr import OCRCTCLoss
from ocr import GtPreprocessor 
from ocr import ImagePreprocessor 
from ocr import OCRTrainDataset, OCRValDataset
from ocr import PreprocessingRunner
from ocr import BatchGenerator
from ocr import GreedyDecoder

class PytorchOCRTrainer(object):

    def __init__(self, trainer_params, args):
        
        #Init arguments
        self.args = args
        self.trainer_params = trainer_params
        self.experiment_start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.checkpoints_dir_path = os.path.join(self.output_dir_path, 'checkpoints')
        self.setup_experiment_output()
        self.logger = Logger(args, trainer_params, self.output_dir_path)

        #Randomness
        random.seed(trainer_params.random_seed)
        torch.manual_seed(trainer_params.random_seed)
        torch.cuda.manual_seed_all(trainer_params.random_seed)

        #Init data preprocessors and dataset
        self.image_preprocessor = ImagePreprocessor(trainer_params)
        self.gt_preprocessor = GtPreprocessor()
        self.train_dataset = OCRTrainDataset(args.imgs, args.sortedtrain)
        self.val_dataset = OCRValDataset(args.imgs, args.valid)
        
        #Run train preprocessing
        self.train_preprocessing_runner = PreprocessingRunner(trainer_params,
                                                        self.train_dataset, 
                                                        self.image_preprocessor, 
                                                        self.gt_preprocessor)
        self.train_preprocessing_runner.run()

        #Run val preprocessing with same preprocessors
        self.val_preprocessing_runner = PreprocessingRunner(trainer_params,
                                                        self.val_dataset,
                                                        self.image_preprocessor, 
                                                        self.gt_preprocessor)
        self.val_preprocessing_runner.run()

        #Setup dataloaders
        self.batch_generator = BatchGenerator(args.no_cuda)
        self.val_dataloader = DataLoader(self.val_dataset, 
                                           batch_size=trainer_params.batch_size, 
                                           sampler=SequentialSampler(self.val_dataset), 
                                           num_workers=trainer_params.num_workers, 
                                           drop_last=False,
                                           collate_fn=self.batch_generator.collate_batch)

        self.train_seq_dataloader = DataLoader(self.train_dataset, 
                                           batch_size=trainer_params.batch_size, 
                                           sampler=SequentialSampler(self.train_dataset), 
                                           num_workers=trainer_params.num_workers, 
                                           drop_last=True,
                                           collate_fn=self.batch_generator.collate_batch)
        
        self.train_random_dataloader = DataLoader(self.train_dataset, 
                                           batch_size=trainer_params.batch_size, 
                                           sampler=RandomSampler(self.train_dataset), 
                                           num_workers=trainer_params.num_workers, 
                                           drop_last=False,
                                           collate_fn=self.batch_generator.collate_batch)

        #Init starting epoch
        self.starting_epoch = 1

        #Set the model output size to number of tokens after processing the dataset
        output_size = self.gt_preprocessor.number_of_tokens 

        #Init model
        self.model = OCRModule(trainer_params, output_size)
        self.criterion = OCRCTCLoss(trainer_params)
        self.decoder = GreedyDecoder(self.gt_preprocessor)

        #Load model
        if args.input: 
            self.logger.log.info("Loading checkpoint model {}".format(args.input))
            package = torch.load(self.args.input, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(package['state_dict'])

        if not self.args.no_cuda:
            self.model = self.model.cuda()

        #Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), 
            	                   lr = trainer_params.lr)
        self.scheduler = trainer_params

        #Load from checkpoint to resume or retrain, if any
        if args.input and args.pretrained_policy == "RESUME":
                self.logger.log.info("Resuming training from {}".format(args.input))
                self.optimizer.load_state_dict(package['optim_dict'])
                self.starting_epoch = package['epoch']
                if not args.no_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()
            
        if args.input and args.pretrained_policy == "RETRAIN":
                self.logger.log.info("Retraining from {}".format(args.input))
                pass

        #Perform BN FC fusion, if requested
        if args.init_bn_fc_fusion:
            if not trainer_params.prefused_bn_fc:
                self.model.batch_norm_fc.init_fusion()
                self.trainer_params.prefused_bn_fc = True
            else:
                raise Exception("BN and FC are already fused.")

        #Recap
        self.logger.log_params()
        self.logger.log_model_info(self.model)

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, trainer_params):
        if trainer_params.lr_schedule == 'STEP':
            self._scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=trainer_params.lr_step, gamma=trainer_params.lr_gamma)
        elif trainer_params.lr_schedule == 'FIXED':
            self._scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[lambda epoch: 1])
        else:
            raise Exception("Unknown lr schedule {}".format(trainer_params.lr_schedule))

    @property
    def experiment_name(self):
        if self.args.pretrained_policy == "RESUME" and self.args.input:
            checkpoints_path = os.path.dirname(self.args.input)
            name = os.path.basename(os.path.split(checkpoints_path)[1])
            return name
        else:
            return '{}_{}{}_W{}B{}A{}IA{}_FC_W{}B{}'.format(
                    self.experiment_start_time, 
                    self.trainer_params.neuron_type,
                    self.trainer_params.layer_size,
                    self.trainer_params.recurrent_weight_bit_width,
                    self.trainer_params.recurrent_bias_bit_width if self.trainer_params.recurrent_bias_enabled else "OFF",  
                    self.trainer_params.recurrent_activation_bit_width,
                    self.trainer_params.internal_activation_bit_width,
                    self.trainer_params.fc_weight_bit_width,
                    self.trainer_params.fc_bias_bit_width)

    @property
    def output_dir_path(self):
        if self.args.pretrained_policy == "RESUME" and self.args.input:
            return os.path.normpath(os.path.join(os.path.dirname(self.args.input), os.pardir))
        else:
            return os.path.join(self.args.experiments, self.experiment_name)

    def setup_experiment_output(self):
        if (self.args.pretrained_policy == "RESUME" and self.args.input) or self.args.dry_run:
            pass
        else:
            os.mkdir(self.output_dir_path)
            os.mkdir(self.checkpoints_dir_path)  

    def train_dataloader(self, epoch):
        if epoch < self.trainer_params.seq_to_random_threshold:
            return self.train_seq_dataloader
        else:
            return self.train_random_dataloader

    def on_training_batch_end(self, loss, accuracy, batch_size, epoch, batch, num_batches, batch_time):
        batch_wer, batch_cer = accuracy
        self.logger.training_accuracy.wer.update(batch_wer, batch_size)
        self.logger.training_accuracy.cer.update(batch_cer, batch_size)
        self.logger.loss.update(loss, batch_size)
        self.logger.log_training_batch(lr=self.optimizer.param_groups[0]['lr'], 
                                       batch_size=batch_size,
                                       epoch=epoch,
                                       batch=batch,
                                       num_batches=num_batches,
                                       batch_time=batch_time)

    def on_val_batch_end(self, accuracy, batch_size):
        batch_wer, batch_cer = accuracy
        self.logger.val_accuracy.wer.update(batch_wer, batch_size)
        self.logger.val_accuracy.cer.update(batch_cer, batch_size)

    def on_training_epoch_begin(self):
        self.logger.loss.reset()
        self.logger.training_accuracy.reset()
        self.logger.val_accuracy.reset()

    def on_training_epoch_end(self, epoch):
        self.logger.log_training_epoch(epoch)
        is_updated = self.logger.val_accuracy.update_best(epoch)
        if is_updated["wer"] and not self.args.dry_run: 
            self.checkpoint(epoch, best="wer")
        if is_updated["cer"] and not self.args.dry_run: 
            self.checkpoint(epoch, best="cer")
        if epoch % self.trainer_params.checkpoint_interval == 0 and not self.args.dry_run:
            self.checkpoint(epoch)        

    def on_training_end(self):
        if not self.args.dry_run:
            self.checkpoint(self.trainer_params.epochs)

    def on_val_epoch_end(self, epoch):
        self.logger.log_val_epoch(epoch)

    def checkpoint(self, epoch, best=None):
        postfix = "best_{}".format(best) if best is not None else epoch
        file_path = os.path.join(self.checkpoints_dir_path, 'epoch_{}.tar'.format(postfix))
        self.logger.log.info("Saving checkpoint model to {}".format(file_path))
        torch.save(
            OCRModule.serialize(
                self.trainer_params,
                self.model, 
                optimizer=self.optimizer, 
                epoch=epoch + 1,  #so that at resume it starts from the +1 epoch
                best_val_wer=self.logger.val_accuracy.best_wer,
                best_val_cer=self.logger.val_accuracy.best_cer),
            file_path)

    def compute_batch_accuracy(self, model_output_tensor, seq_len_tensor, gt_tensor, gt_len_tensor):
        # unflatten ground truths
        gt_list = []
        offset = 0
        for size in gt_len_tensor:
            gt_list.append(gt_tensor[offset:offset + size])                
            offset += size

        output_list = self.decoder.decode(model_output_tensor, True)
        gt_string_list = self.decoder.convert_to_strings(gt_list, False)
        assert(len(output_list) == len(gt_string_list))

        #Log random output vs gt sample
        _, sample_decoded_transcript = output_list[0] 
        _, sample_decoded_reference = gt_string_list[0]
        self.logger.log.info('Sample Decoded Transcript: {}'.format(sample_decoded_transcript))
        self.logger.log.info('Sample Decoded Reference:  {}'.format(sample_decoded_reference))

        batch_wer = 0
        batch_cer = 0

        for i in range(len(gt_string_list)):
            encoded_transcript, decoded_transcript = output_list[i] 
            encoded_reference, decoded_reference = gt_string_list[i]
            batch_wer += self.decoder.wer(decoded_transcript, decoded_reference) / float(max(len(decoded_reference.split()), len(decoded_transcript.split())))
            batch_cer += self.decoder.cer(decoded_transcript, decoded_reference) / float(max(len(decoded_reference), len(decoded_transcript)))
        
        #Put the values relative to 100%
        batch_wer *= 100
        batch_cer *= 100

        return batch_wer, batch_cer
    
    def train_batch(self, data):
        #Get training data from data tuple
        image_tensor, seq_len_list, seq_len_tensor, gt_tensor, gt_len_tensor = data
        training_batch_size = image_tensor.size(1)
        
        #Setup variables
        image_tensor = Variable(image_tensor, requires_grad=False)
        seq_len_tensor = Variable(seq_len_tensor, requires_grad=False)
        gt_tensor = Variable(gt_tensor, requires_grad=False)
        gt_len_tensor = Variable(gt_len_tensor, requires_grad=False)

        #Compute forward of the model
        output = self.model(image_tensor, seq_len_list)

        #Computes forward of the loss 
        loss = self.criterion(output, seq_len_tensor, gt_tensor, gt_len_tensor)        
        loss_value = loss.data[0]

        #Compute gradients
        self.optimizer.zero_grad()
        loss.backward()

        #Clip gradients
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.trainer_params.max_norm)

        #Propagates updates
        self.optimizer.step()
                
        if not self.args.no_cuda:
            torch.cuda.synchronize()

        #Compute training accuracy for current batch
        training_batch_accuracy = self.compute_batch_accuracy(output.data, 
                                                              seq_len_tensor.data, 
                                                              gt_tensor.data, 
                                                              gt_len_tensor.data)  
        return loss_value, training_batch_accuracy, training_batch_size

    def val_batch(self, data):
        #Get validation data from data tuple
        image_tensor, seq_len_list, seq_len_tensor, gt_tensor, gt_len_tensor = data
        val_batch_size = image_tensor.size(1)

        #Setup variables
        image_tensor = Variable(image_tensor, volatile=True)
        seq_len_tensor = Variable(seq_len_tensor, volatile=True)
        gt_tensor = Variable(gt_tensor, volatile=True)
        gt_len_tensor = Variable(gt_len_tensor, volatile=True)

        #Compute forward of the model
        output = self.model(image_tensor, seq_len_list)

        #Compute valid accuracy
        val_batch_accuracy = self.compute_batch_accuracy(output.data, 
                                                         seq_len_tensor.data, 
                                                         gt_tensor.data, 
                                                         gt_len_tensor.data)
        return val_batch_accuracy, val_batch_size

    def train_model(self):
        for epoch in range(self.starting_epoch, self.trainer_params.epochs):
            #Epoch starts
            self.on_training_epoch_begin()
            
            self.scheduler.step()
            
            #Set current mode to training
            self.model.train() 

            #Go through train batches
            for i, (data) in enumerate(self.train_dataloader(epoch)):
                batch_start_time = time.time()
                loss_value, train_batch_accuracy, training_batch_size = self.train_batch(data)
                self.on_training_batch_end(loss=loss_value, 
                                           accuracy=train_batch_accuracy, 
                                           batch_size=training_batch_size,
                                           epoch=epoch,
                                           batch=i,
                                           num_batches=len(self.train_dataloader(epoch)),
                                           batch_time=time.time() - batch_start_time)
            
            #Perform eval
            self.eval_model(epoch)

            #Epoch ends
            self.on_training_epoch_end(epoch)

        #training ends
        self.on_training_end()

    def eval_model(self, epoch=None):
        #Switch to eval and go through val set
        self.model.eval()

        #Go through val dataset to accumulate cer and wer
        for (data) in self.val_dataloader:
            val_batch_accuracy, val_batch_size = self.val_batch(data)
            self.on_val_batch_end(val_batch_accuracy, val_batch_size) 

        self.on_val_epoch_end(epoch)

    def export_model(self, simd_factor, pe):
        self.model.eval()
        self.model.export(os.path.join(self.output_dir_path, 'r_model_fw_bw.h'), simd_factor, pe)

    def export_test_image(self, height, index=100):
        tensor, gt = self.train_dataset.image_gt_list[index]
        print("Exporting image with GT: {}".format(self.decoder.convert_to_strings([gt], False)))
        tensor = tensor.expand(1, -1, height).transpose(0, 1)
        np.savetxt("test_image.txt", tensor.contiguous().numpy().reshape(-1,1), fmt='%.10f')
        
        var = Variable(tensor, volatile=True)
        output_tensor = self.model(var, [tensor.size()[0]])
        print("Image resognized as: {}".format(self.decoder.decode(output_tensor.data, True)))


            
            



            

        

