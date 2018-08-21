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

import os

import torch
from torch.utils.data import Dataset
from PIL import Image

class OCRDataset(Dataset):
    def __init__(self, images_dir_abs_path, image_ground_truth_file_abs_path):
        self.images_dir_abs_path = images_dir_abs_path
        self.image_ground_truth_file_abs_path = image_ground_truth_file_abs_path
        self.image_gt_list = []
        self.setup_dataset_list()

    def setup_dataset_list(self):
        with open(self.image_ground_truth_file_abs_path) as train_file:
            for line in train_file:
                (rel_image_path, ground_truth_string) = line.split(',', 1) #1 is maxsplit
                abs_image_path = os.path.join(self.images_dir_abs_path, rel_image_path + '.png')
                image = Image.open(abs_image_path).convert('L') 
                self.image_gt_list.append((image, ground_truth_string))
        self.order_by_width()

    def order_by_width(self):
        self.image_gt_list = sorted(self.image_gt_list, key=lambda (image, gt): image.size[0])

    def update_images(self, function):
        self.image_gt_list = [(function(image), gt) for (image, gt) in self.image_gt_list]

    def update_gts(self, function):
        self.image_gt_list = [(image, function(gt)) for (image, gt) in self.image_gt_list]

    def call_on_each_image(self, function):
        for image, gt in self.image_gt_list:
            function(image)

    def call_on_each_gt(self, function):
        for image, gt in self.image_gt_list:
            function(gt)

    def __len__(self):
        return len(self.image_gt_list)

    def __getitem__(self, index):
        return self.image_gt_list[index]


class OCRTrainDataset(OCRDataset):
    def __init__(self, images_dir_abs_path, train_ground_truth_file_abs_path):
        super(OCRTrainDataset, self).__init__(images_dir_abs_path, train_ground_truth_file_abs_path)
        


class OCRValDataset(OCRDataset):
    def __init__(self, images_dir_abs_path, val_ground_truth_file_abs_path):
        super(OCRValDataset, self).__init__(images_dir_abs_path, val_ground_truth_file_abs_path)

        
