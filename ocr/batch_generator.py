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

import torch

class BatchGenerator(object):
    def __init__(self, no_cuda):
        self.no_cuda = no_cuda
        
    #Assume tensor is Seq/Width x Height
    def pad_tensor(self, tensor, max_seq_len):
        original_seq_len, height = tensor.size()
        padded_seq_len = max_seq_len - original_seq_len
        
        if padded_seq_len:
            padding = torch.zeros(padded_seq_len, height)
            padded_tensor = torch.cat((tensor, padding), 0).contiguous()
            return padded_tensor
        
        else:
            return tensor

    #Assume image tensors are Seq/Width x Height
    def get_max_seq_len(self, batch_list):   
        max_seq_len = 0
        
        for image, gt in batch_list:
            seq_len, height = image.size()
            
            if seq_len > max_seq_len:
                max_seq_len = seq_len
        
        return max_seq_len

    def get_tensor_lists(self, batch_list, max_seq_len):
        image_tensor_list = [] 
        seq_len_list = []
        gt_list = []
        gt_len_list = []
        
        for image, gt in batch_list:
            #Non padded sequence length
            seq_len_list.append(image.size()[0])
            
            #gt is already a list of tokens and CTC wants a SINGLE tensor for the whole batch
            gt_list.extend(gt)
            
            #In order to distinguish the tokens belonging to different samples, we are also passing
            #to CTC a tensor of the length of the gt for each sample in the batch
            gt_len_list.append(len(gt))
            image_tensor_list.append(self.pad_tensor(image, max_seq_len))
        
        return (image_tensor_list, seq_len_list, gt_list, gt_len_list)

    def generate_data_tensors(self, lists_tuple):
        image_tensor_list, seq_len_list, gt_list, gt_len_list = lists_tuple
        
        # CTC wants in input a 1D tensor containing the non-padded size of each output sequence of the model 
        seq_len_tensor = torch.IntTensor(seq_len_list).contiguous()
       
        # stack on dim 1, i.e. batch, so that we have SEQUENCE_LEN x BATCH x FEATURES (target_height)
        image_tensor = torch.stack(image_tensor_list, dim=1).contiguous()
        gt_tensor = torch.IntTensor(gt_list).contiguous()
        gt_len_tensor = torch.IntTensor(gt_len_list).contiguous()
        
        if not self.no_cuda:
            image_tensor = image_tensor.cuda()
        
        return (image_tensor, seq_len_list, seq_len_tensor, gt_tensor, gt_len_tensor)

    def collate_batch(self, batch_list):
        batch_list.sort(key=lambda (image, gt): image.size()[0], reverse=True)
        max_seq_len = self.get_max_seq_len(batch_list)
        lists_tuple = self.get_tensor_lists(batch_list, max_seq_len)
        return self.generate_data_tensors(lists_tuple)


    