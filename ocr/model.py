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

from functools import partial
import os

import math
import numpy
import torch
import torch.nn as nn

from quantization.modules.quantized_linear import QuantizedLinear
from quantization.modules.rnn import QuantizedGRU, QuantizedLSTM
from quantization.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class FusedBatchNorm1dLinear(nn.Module):
    def __init__(self, trainer_params, batch_norm, linear):
        super(FusedBatchNorm1dLinear, self).__init__()
        self.batch_norm = batch_norm
        self.linear = linear
        self.trainer_params = trainer_params
    
    def forward(self, x):
        if self.trainer_params.prefused_bn_fc:
            x = self.linear(x)
        else:
            x = self.batch_norm(x)
            x = self.linear(x)
        return x

    #To be called after weights have been restored in trainer.py
    def init_fusion(self):   
        bn_weight_var = torch.mul(self.batch_norm.weight.data, torch.rsqrt(self.batch_norm.running_var + self.batch_norm.eps))
        bias_coeff = self.batch_norm.bias.data - torch.mul(self.batch_norm.running_mean, bn_weight_var)
        self.linear.bias.data = torch.addmv(self.linear.bias.data, self.linear.weight.data, bias_coeff)
        self.linear.weight.data = self.linear.weight.data * bn_weight_var.expand_as(self.linear.weight.data)

class OCRModule(nn.Module):
    def __init__(self, trainer_params, output_size):
        super(OCRModule, self).__init__()
        self.trainer_params = trainer_params
        self.recurrent_layer = self.recurrent_layer_type(input_size=self.trainer_params.target_height,
                                                         hidden_size=self.trainer_params.layer_size,
                                                         num_layers=1,
                                                         batch_first=False,
                                                         bidirectional=self.trainer_params.bidirectional,
                                                         bias=self.trainer_params.recurrent_bias_enabled)
                                                 
        self.batch_norm_fc = FusedBatchNorm1dLinear(
            trainer_params,
            nn.BatchNorm1d(self.reduce_factor * self.trainer_params.layer_size),
            QuantizedLinear(
                      bias=True,
                      in_features=self.reduce_factor * self.trainer_params.layer_size,
                      out_features=output_size,
                      bias_bit_width=self.trainer_params.fc_bias_bit_width, 
                      bias_q_type=self.trainer_params.fc_bias_quantization, 
                      weight_bit_width=self.trainer_params.fc_weight_bit_width, 
                      weight_q_type=self.trainer_params.fc_weight_quantization)
        )

        self.output_layer = nn.Sequential(SequenceWise(self.batch_norm_fc))

    @property
    def direction_factor(self):
        if self.trainer_params.bidirectional:
            return 2
        else:
            return 1

    @property
    def reduce_factor(self):
        if self.trainer_params.bidirectional and self.trainer_params.reduce_bidirectional == 'CONCAT':
            return 2
        else:
            return 1

    @property
    def recurrent_layer_type(self):
        if self.trainer_params.neuron_type == 'QGRU':
            func = QuantizedGRU
        elif self.trainer_params.neuron_type == 'QLSTM':
            func = QuantizedLSTM
        elif self.trainer_params.neuron_type == 'LSTM':
            func = nn.LSTM
        elif self.trainer_params.neuron_type == 'GRU':
            func = nn.LSTM
        else:
            raise Exception("Invalid neuron type.")

        if self.trainer_params.neuron_type == 'QLSTM' or self.trainer_params.neuron_type == 'QGRU':
            func = partial(func, bias_bit_width=self.trainer_params.recurrent_bias_bit_width, 
                                 bias_q_type=self.trainer_params.recurrent_bias_quantization, 
                                 weight_bit_width=self.trainer_params.recurrent_weight_bit_width, 
                                 weight_q_type=self.trainer_params.recurrent_weight_quantization, 
                                 activation_bit_width=self.trainer_params.recurrent_activation_bit_width, 
                                 activation_q_type=self.trainer_params.recurrent_activation_quantization,
                                 internal_activation_bit_width=self.trainer_params.internal_activation_bit_width)
        return func

    def forward(self, x, seq_len):
        if self.trainer_params.mask_padded:
            x = pack_padded_sequence(x, seq_len, batch_first=False)
        x, h = self.recurrent_layer(x) 
        if self.trainer_params.mask_padded:
            x, _ = pad_packed_sequence(x)

        if self.trainer_params.bidirectional:
            if self.trainer_params.reduce_bidirectional == 'SUM':
                x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
            elif self.trainer_params.reduce_bidirectional == 'CONCAT':
                #do nothing, x is already in the proper shape
                pass
            else:
                raise Exception('Unknown reduce mode: {}'.format(self.trainer_params.reduce_bidirectional))

        x = self.output_layer(x)
        # x has size SEQUENCE (= max width) x BATCH x FEATURES (= output size = num of tokens)
        return x

    def export(self, output_path, simd_factor, pe):
        if self.trainer_params.neuron_type == 'QLSTM':
            assert(self.trainer_params.target_height % simd_factor == 0)
            assert(self.trainer_params.layer_size % simd_factor == 0)
            assert((simd_factor >= 1 and pe == 1) or (simd_factor == 1 and pe >= 1))
            
            ih_simd = self.trainer_params.target_height / simd_factor 
            hh_simd = self.trainer_params.layer_size / simd_factor 
            
            lstm_weight_ih = self.recurrent_layer.hls_lstm_weight_ih_string(ih_simd, pe)
            lstm_weight_hh = self.recurrent_layer.hls_lstm_weight_hh_string(hh_simd, pe)

            lstm_weight_decl_list = map(list, zip(*lstm_weight_ih))[0] + map(list, zip(*lstm_weight_hh))[0]
            lstm_weight_string_list = map(list, zip(*lstm_weight_ih))[1] + map(list, zip(*lstm_weight_hh))[1]
            
            if self.trainer_params.recurrent_bias_enabled:
                lstm_bias = self.recurrent_layer.hls_lstm_bias_strings(pe)
                lstm_bias_decl_list = map(list, zip(*lstm_bias))[0]
                lstm_bias_string_list = map(list, zip(*lstm_bias))[1]
            
            fc_weight_decl, fc_weight_string = self.batch_norm_fc.linear.hls_weight_string(self.reduce_factor)
            fc_bias_decl, fc_bias_string = self.batch_norm_fc.linear.hls_bias_string(self.reduce_factor)
            
            def define(name, val):
                return "#define {} {}\n".format(name, val)

            with open(output_path, 'w') as f: 
                print("Exporting model to {}".format(output_path))
                f.write("#pragma once" + '\n')
                
                f.write(define("PE", pe))
                f.write(define("SIMD_INPUT", ih_simd))
                f.write(define("SIMD_RECURRENT", hh_simd))
                f.write(define("NUMBER_OF_NEURONS", self.trainer_params.layer_size))
                f.write(define("NUMBER_OF_NEURONS_TYPEWIDTH", int(math.ceil(math.log(self.trainer_params.layer_size, 2.0)) + 1)))
                f.write(define("HEIGHT_IN_PIX", self.trainer_params.target_height))
                f.write(define("HEIGHT_IN_PIX_TYPEWIDTH", int(math.ceil(math.log(self.trainer_params.target_height, 2.0)) + 1)))
                f.write(define("MAX_NUMBER_COLUMNS_TEST_SET", 732))
                f.write(define("MAX_NUMBER_COLUMNS_TEST_SET_TYPEWIDTH", 11))
                f.write(define("SIZE_OF_OUTPUT_BUFFER", 96))
                data_width = 64
                f.write(define("DATAWIDTH", data_width))
                input_bit_width = self.trainer_params.recurrent_activation_bit_width if self.trainer_params.quantize_input else 8
                f.write(define("PIXELWIDTH", input_bit_width)) 
                f.write(define("PACKEDWIDTH", int(data_width * input_bit_width / 2)))
                f.write(define("WEIGHTWIDTH", self.trainer_params.recurrent_weight_bit_width))
                f.write(define("BIASWIDTH", self.trainer_params.recurrent_bias_bit_width))
                f.write(define("FCWEIGHTWIDTH", self.trainer_params.fc_weight_bit_width))
                f.write(define("FCBIASWIDTH", self.trainer_params.fc_bias_bit_width))
                f.write(define("OUTPUTACTIVATIONHIDDENLAYERWIDTH", self.trainer_params.recurrent_activation_bit_width))
                f.write(define("OUTPUTACTIVATIONOUTPUTLAYERWIDTH", 16))
                    
                for decl in lstm_weight_decl_list:
                    f.write(decl + '\n')
                if self.trainer_params.recurrent_bias_enabled:
                    for decl in lstm_bias_decl_list:
                        f.write(decl + '\n')
                f.write(fc_weight_decl + '\n')
                f.write(fc_bias_decl + '\n')
                
                f.write(fc_weight_string + '\n')
                for string in lstm_weight_string_list:
                    f.write(string + '\n')
                f.write(fc_bias_string + '\n')
                if self.trainer_params.recurrent_bias_enabled:
                    for string in lstm_bias_string_list:
                        f.write(string + '\n')
        else:
            raise Exception("Export not supported for {}".format(self.trainer_params.neuron_type))

    @staticmethod
    def serialize(trainer_params, model, optimizer, epoch, best_val_wer, best_val_cer):
        package = {
            'trainer_params': trainer_params,
            'state_dict': model.state_dict(),
            'optim_dict' : optimizer.state_dict(),
            'epoch' : epoch,
            'best_val_wer': best_val_wer,
            'best_val_cer': best_val_cer
        }
        return package
    
    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params
    
