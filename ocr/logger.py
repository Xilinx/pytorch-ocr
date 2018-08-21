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

import json
import sys
import logging
import os

import torch
from tensorboardX import SummaryWriter

from model import OCRModule

class BatchAverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0 #per batch value normalized over batch size
        self.sum = 0 #sum over batches
        self.count = 0 #number of batches
        self.avg = 0 #average over batches

    def update(self, batch_val, batch_size):
        self.val = batch_val / batch_size #val normalized by batch size
        self.sum += batch_val
        self.count += batch_size
        self.avg = self.sum / self.count

class OCRAccuracy(object):
    def __init__(self, best_wer=(0, 100), best_cer=(0, 100)):
        self.cer = BatchAverageMeter()
        self.wer = BatchAverageMeter()
        self.best_cer = best_cer #epoch, value
        self.best_wer = best_wer 

    def reset(self):
        self.cer.reset()
        self.wer.reset()

    def update_best(self, epoch):
        is_updated = {"wer": False, "cer": False}
        if self.wer.avg < self.best_wer[1]:
            self.best_wer = (epoch, self.wer.avg)
            is_updated["wer"] = True
        if self.cer.avg < self.best_cer[1]:
            self.best_cer = (epoch, self.cer.avg)
            is_updated["cer"] = True
        return is_updated

class Logger(object):
    def __init__(self, args, trainer_params, output_dir_path):
        self.args = args
        self.trainer_params = trainer_params
        self.output_dir_path = output_dir_path
        self.log = logging.getLogger('log')
        self.log.setLevel(logging.INFO)
        
        #Stout logging
        out_hdlr = logging.StreamHandler(sys.stdout)
        out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        out_hdlr.setLevel(logging.INFO)
        self.log.addHandler(out_hdlr)
        
        #Txt and tensorboard logging
        if not args.dry_run:
            file_hdlr = logging.FileHandler(os.path.join(output_dir_path, 'log.txt'))
            file_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
            file_hdlr.setLevel(logging.INFO)
            self.log.addHandler(file_hdlr)
            self.tb_writer = SummaryWriter(output_dir_path)

        #Init tracked values
        self.loss = BatchAverageMeter()
        self.training_accuracy = OCRAccuracy()
        self.val_accuracy = OCRAccuracy(self.starting_best_val_wer, self.starting_best_val_cer)

    @property
    def recurrent_bias_string(self):
        if self.trainer_params.recurrent_bias_enabled:
            return '{} {} bit(s)\t'.format(
              self.trainer_params.recurrent_bias_quantization,
              self.trainer_params.recurrent_bias_bit_width)
        else:
            return "OFF\t" 

    @property
    def input_quantization_string(self):
        if self.trainer_params.quantize_input:
            return "ON"
        else:
            return "OFF"

    @property
    def starting_best_val_wer(self):
        if self.args.input and self.args.pretrained_policy == "RESUME":
            package = torch.load(self.args.input, map_location=lambda storage, loc: storage)
            try:
                return package['best_val_wer']
            except KeyError:
                self.log.info("No best_val_wer found in checkpoint, set to 100%")
                return (0, 100)
        elif self.args.input and self.args.pretrained_policy == "RETRAIN":
            package = torch.load(self.args.input, map_location=lambda storage, loc: storage)
            try:
                return (0, package['best_val_wer'][1]) #epoch, value
            except KeyError:
                self.log.info("No best_val_wer found in checkpoint, set to 100%")
                return (0, 100)
        else:
            return (0, 100) #epoch, value
    
    @property
    def starting_best_val_cer(self):
        if self.args.input and self.args.pretrained_policy == "RESUME":
            package = torch.load(self.args.input, map_location=lambda storage, loc: storage)
            try:
                return package['best_val_cer']
            except KeyError:
                self.log.info("No best_val_cer found in checkpoint, set to 100%")
                return (0, 100)
        elif self.args.input and self.args.pretrained_policy == "RETRAIN":
            package = torch.load(self.args.input, map_location=lambda storage, loc: storage)
            try:
                return (0, package['best_val_cer'][1]) #epoch, value
            except KeyError:
                self.log.info("No best_val_cer found in checkpoint, set to 100%")
                return (0, 100)
        else:
            return (0, 100) #epoch, value

    def log_params(self):
        self.log.info(self.trainer_params)
        self.log.info(self.args.__dict__)
        if not self.args.dry_run:
            self.tb_writer.add_text('Trainer Params', str(self.trainer_params))
            self.tb_writer.add_text('Args', str(self.args))
            with open(os.path.join(self.output_dir_path, 'trainer_params.json'), 'a') as fp:
                json.dump(self.trainer_params, fp) 
            with open(os.path.join(self.output_dir_path, 'args.txt'), 'a') as fp:
                json.dump(self.args.__dict__, fp) 

    def log_model_info(self, model):
        architecture = "Architecture: {}".format(model)
        num_parameters = "Number of parameters: {}".format(OCRModule.get_param_size(model))
        self.log.info(architecture)
        self.log.info(num_parameters)
        if not self.args.dry_run:
            self.tb_writer.add_text('Architecture', architecture)
            self.tb_writer.add_text('Num Parameters', num_parameters)

    def log_training_batch(self, lr, epoch, batch, num_batches, batch_size, batch_time):
        self.log.info('Input Quantization: {}\t'
              'Recurrent Weights: {} {} bit(s)\t'
              'Recurrent Activations: {} {} bit(s)\t' 
              'Recurrent Tanh/Sigmoid: {} bit(s)\t'
              'Recurrent Bias: {}'.format(
              self.input_quantization_string,
              self.trainer_params.recurrent_weight_quantization,
              self.trainer_params.recurrent_weight_bit_width,
              self.trainer_params.recurrent_activation_quantization,
              self.trainer_params.recurrent_activation_bit_width,
              self.trainer_params.internal_activation_bit_width,
              self.recurrent_bias_string))
        
        self.log.info('FC Weights: {} {} bit(s)\t'
              'FC Bias: {} {} bit(s)\t'.format(
              self.trainer_params.fc_weight_quantization,
              self.trainer_params.fc_weight_bit_width,
              self.trainer_params.fc_bias_quantization,
              self.trainer_params.fc_bias_bit_width))
        
        self.log.info('Epoch: [{}][{}/{}]\t'
              'Batch Size: {}\t'
              'Batch Time {}\t'
              'LR {}\t'
              'Batch Loss {}\t'.format(
              epoch, 
              (batch + 1), 
              num_batches, 
              batch_size,
              batch_time, 
              lr,
              self.loss.val))

        self.log.info('Best Val Avg WER: {wer:.3f}%, at Epoch: {wer_epoch}\t'
              'Best Val Avg CER: {cer:.3f}%, at Epoch: {cer_epoch}\t'.format(
                  wer_epoch=self.val_accuracy.best_wer[0], 
                  cer_epoch=self.val_accuracy.best_cer[0],
                  wer=self.val_accuracy.best_wer[1], 
                  cer=self.val_accuracy.best_cer[1]))
        
        self.log.info('Batch Training WER: {wer:.3f}%\t'
              'Batch Training CER: {cer:.3f}%\t \n'.format(
                  wer=self.training_accuracy.wer.val, 
                  cer=self.training_accuracy.cer.val))
        
        if not self.args.dry_run:
            tb_iteration = epoch * (batch + 1)
            self.tb_writer.add_scalar('loss', self.loss.val, tb_iteration)
            self.tb_writer.add_scalar('iteration_train_wer', self.training_accuracy.wer.val, tb_iteration) 
            self.tb_writer.add_scalar('iteration_train_cer', self.training_accuracy.cer.val, tb_iteration) 

    def log_val_epoch(self, epoch):
        self.log.info('Average Epoch Val WER: {wer:.3f}%\t'
              'Average Epoch Val CER: {cer:.3f}%\t \n'.format(wer=self.val_accuracy.wer.avg, cer=self.val_accuracy.cer.avg))
        if not self.args.dry_run and epoch is not None:
            self.tb_writer.add_scalar('epoch_val_wer', self.val_accuracy.wer.avg, epoch) 
            self.tb_writer.add_scalar('epoch_val_cer', self.val_accuracy.cer.avg, epoch)

    def log_training_epoch(self, epoch):
        self.log.info('Training Summary Epoch: [{}]\t'
              'Average Loss {loss:.3f}\t'.format(epoch, loss=self.loss.avg))
        self.log.info('Average Epoch Training WER: {wer:.3f}%\t'
              'Average Epoch Training CER: {cer:.3f}%\t'.format(wer=self.training_accuracy.wer.avg, cer=self.training_accuracy.cer.avg))
        if not self.args.dry_run:
            self.tb_writer.add_scalar('epoch_avg_train_wer', self.training_accuracy.wer.avg, epoch) 
            self.tb_writer.add_scalar('epoch_avg_train_cer', self.training_accuracy.cer.avg, epoch)
