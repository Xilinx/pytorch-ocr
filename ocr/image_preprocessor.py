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

from __future__ import division # / is always float division in both Python 2 and 3

import sys

from PIL import Image
import PIL.ImageOps 
from torchvision.transforms.functional import resize, to_tensor
import math 
import torch
import numpy as np
from quantization.function.quantization_scheme import ActivationQuantizationScheme

DELTA = 0.1

class ImagePreprocessor(object):
    def __init__(self, trainer_params):
        self.target_height = trainer_params.target_height
        self.input_quantization_scheme = ActivationQuantizationScheme(trainer_params.recurrent_activation_bit_width, trainer_params.recurrent_activation_quantization)

    def get_scaling_ratio(self, original_height):
        return (self.target_height + DELTA) / original_height

    def invert_image(self, image):
        return PIL.ImageOps.invert(image)

    def resize_and_normalize_image(self, image):
        original_width, original_height = image.size
        scaling_ratio = self.get_scaling_ratio(original_height)
        target_width = int(math.ceil(original_width * scaling_ratio))
        size = (int(target_width), int(self.target_height))
        resized_image = image.resize(size, Image.BILINEAR) #This call also normalizes the input image
        return resized_image

    def image_to_squeezed_transposed_tensor(self, image):
        #C x H x W
        tensor_image = to_tensor(image)
        #Swap dimensions 1 (H) and 2 (W) to get C x W x H
        transposed_tensor = torch.transpose(tensor_image, 1, 2)
        #C = 1 so we can remove it
        squeezed_tensor = torch.squeeze(transposed_tensor)
        return squeezed_tensor

    def shift_scale_tensor_image(self, tensor_image):
        return (tensor_image - 0.5) * 2 #put it between -1 and 1

    def quantize_tensor_image(self, tensor_image):
        out = self.input_quantization_scheme.q_forward(tensor_image)
        return out.data
