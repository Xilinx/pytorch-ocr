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


import argparse
import json
import os

import torch
import numpy as np
from ocr import PytorchOCRTrainer

torch.backends.cudnn.enabled = False
torch.set_printoptions(precision=10)

class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

def ascii_encode_dict(data):
    ascii_encode = lambda x: x.encode('ascii')
    return dict(map(ascii_encode, pair) if isinstance(pair[1], unicode) else pair for pair in data.items())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCR training')
    parser.add_argument('--params', '-p', default="default_trainer_params.json", help='Path to params JSON file. Default ignored when resuming.')
    parser.add_argument('--experiments', '-e', default="experiments", help='Path for experiments. Ignored when resuming.')
    parser.add_argument('--input', '-i', help='Path to input checkpoint.')
    parser.add_argument('--pretrained_policy', default="RESUME", help='RESUME/RETRAIN.')
    parser.add_argument('--init_bn_fc_fusion', default=False, action='store_true', help='Init BN FC fusion.')
    parser.add_argument('--eval', default=False, action='store_true', help='Perform only evaluation on val dataset.')
    parser.add_argument('--export', default=False, action='store_true', help='Perform only export of quantized weights.')
    parser.add_argument('--no_cuda', default=False, action='store_true', help='Run on CPU.')
    parser.add_argument('--export_test_image', default=False, action='store_true', help='Export pre-quantized and reshaped test image.')
    parser.add_argument('--valid', default="db_files_uw3-500/valid.txt", help='Input path for val file.')
    parser.add_argument('--sortedtrain', default="db_files_uw3-500/sortedTrain.txt", help='Input path for train file.')
    parser.add_argument('--imgs', default="db_files_uw3-500/imgs", help='Input path for images dir.')
    parser.add_argument('--dry_run', default=False, action='store_true', help='Do not write any output file.')
    parser.add_argument('--simd_factor', default=1, type=int, help='SIMD factor for export.')
    parser.add_argument('--pe', default=1, type=int, help='Number of PEs for export.')

    #Overrides
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--layer_size', type=int)
    parser.add_argument('--neuron_type', type=str)
    parser.add_argument('--target_height', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_schedule', type=str)
    parser.add_argument('--lr_step', type=int)
    parser.add_argument('--lr_gamma', type=float)
    parser.add_argument('--max_norm', type=float)
    parser.add_argument('--seq_to_random_threshold', type=int)
    parser.add_argument('--bidirectional', type=bool)
    parser.add_argument('--reduce_bidirectional', type=str)
    parser.add_argument('--recurrent_bias_enabled', type=bool)
    parser.add_argument('--checkpoint_interval', type=int)
    parser.add_argument('--recurrent_weight_bit_width', type=int)
    parser.add_argument('--recurrent_weight_quantization', type=str)
    parser.add_argument('--recurrent_bias_bit_width', type=int)
    parser.add_argument('--recurrent_bias_quantization', type=str)
    parser.add_argument('--recurrent_activation_bit_width', type=int)
    parser.add_argument('--recurrent_activation_quantization', type=str)
    parser.add_argument('--internal_activation_bit_width', type=int)
    parser.add_argument('--fc_weight_bit_width', type=int)
    parser.add_argument('--fc_weight_quantization', type=str)
    parser.add_argument('--fc_bias_bit_width', type=int)
    parser.add_argument('--fc_bias_quantization', type=str)
    parser.add_argument('--quantize_input', type=bool)
    parser.add_argument('--mask_padded', type=bool)

    args = parser.parse_args()

    #Set paths relative to main.py
    path_args = ['params', 'experiments', 'input', 'valid', 'sortedtrain', 'imgs']
    for path_arg in path_args:
        path = getattr(args, path_arg)
        if path is not None and not os.path.isabs(path):
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
            setattr(args, path_arg, abs_path)

    #Avoid creating new folders etc. 
    if args.eval or args.export or args.export_test_image:
        args.dry_run = True

    #force cpu when exporting weights
    if args.export or args.export_test_image:
        args.no_cuda = True

    if args.input and args.pretrained_policy == "RESUME" and args.params == "default_trainer_params.json":
        package = torch.load(args.input, map_location=lambda storage, loc: storage)
        trainer_params = package['trainer_params']
    else:
        with open(args.params) as d:
            trainer_params = json.load(d, object_hook=ascii_encode_dict)
    trainer_params = objdict(trainer_params)

    #Overrides
    if args.epochs is not None:
        trainer_params.epochs = args.epochs
    if args.internal_activation_bit_width is not None:
        trainer_params.internal_activation_bit_width = args.internal_activation_bit_width

    trainer = PytorchOCRTrainer(trainer_params, args)

    if args.export_test_image:
        trainer.export_test_image(trainer_params.target_height)
        exit(0)

    if args.export:
        trainer.export_model(args.simd_factor, args.pe)
        exit(0)

    if args.eval:
        trainer.eval_model()
    else:
        trainer.train_model()






