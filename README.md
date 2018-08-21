# Quantized LSTMs for OCR 

This Pytorch-based repository allows to train a full-precision or quantized bidirectional LSTM to perform OCR on the included dataset. A quantized trained model can be accelerated on the LSTM-PYNQ overlay found here: [https://github.com/Xilinx/LSTM-PYNQ](https://github.com/Xilinx/LSTM-PYNQ)

## Requirements

An Nvidia GPU with a CUDA+CuDNN installation is suggested but not required, training with quantization is supported on CPUs as well.

* Python environment, including Numpy and Pillow (tested with Python 2.7, Pillow 4.2.1, Numpy 1.13.3).
* [Pytorch](https://pytorch.org) (tested with version 0.3.1)
* [pytorch-quantization](https://github.com/xilinx/pytorch-quantization) (tested with master branch)
* [Warp-CTC with Pytorch bindings](https://github.com/SeanNaren/warp-ctc) (tested with commit aba791f)
* [python-levenshtein](https://pypi.python.org/pypi/python-Levenshtein/0.12.0) (tested with version 0.12.0)
* [TensorboardX](https://pypi.python.org/pypi/tensorboardX) (tested with version 1.1)
* [Tensorboard](https://github.com/tensorflow/tensorboard) (optional)

## Suggested Setup

Assuming your OS is a recent linux based distribution, such as Ubuntu 16.04, you can follow the following steps.

### CUDA

* Current repos have been tested against CUDA 9.0. Compatibility with earlier or later CUDA releases is not guaranteed.
* Follow the instructions [here](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal) on how to install CUDA and get the meta-package `cuda-libraries-dev-9-0`.

### Python

* Get a complete Python distribution installer from Anaconda [here](https://repo.anaconda.com/archive/Anaconda2-5.1.0-Linux-x86_64.sh).

* Run the installer in a bash shell with:

```bash
bash Anaconda2-5.1.0-Linux-x86_64.sh
```

and set everything to default. More information about the installation process can be found [here](https://conda.io/docs/user-guide/install/linux.html).

### Pytorch

* assuming a CUDA 9.0 environment (replace `cuda90` otherwise), install Pytorch 0.3.1 through *conda* (the Anaconda package manager) with:

```bash
conda install pytorch=0.3.1 torchvision cuda90 -c pytorch
```

More installer combinations are available [here](https://pytorch.org/).

### Pytorch Quantization

* Install a recent CMake version with:

```bash
conda install cmake
```

* Clone the repo:
```
git clone https://github.com/xilinx/pytorch-quantization
```

* Build the library by running `python build.py` .
* Add the library to your python's path with `export PYTHONPATH=PYTHONPATH:/path/to/pytorch-quantization`, with */path/to/pytorch-quantization* the full path to where you cloned the repo (with no trailing slashes, as shown here).

### Pytorch OCR

* Install the required conda dependencies with:

```bash
conda install -c anaconda pillow 
```

* Install the required pip dependencies with:

```bash
pip install scikit-learn python-levensthein tensorboardX
```

* Compile Warp-CTC with:

```bash
git clone https://github.com/SeanNaren/warp-ctc
cd warp-ctc && git checkout aba791f
mkdir build && cd build && cmake .. && make
```

* Then install Warp-CTC Pytorch bindings with:

```bash
export CUDA_HOME=/usr/local/cuda
cd warp-ctc/pytorch_binding
python setup.py install
```

* Finally, clone the actual repo:
* 
```
git clone https://github.com/xilinx/pytorch-ocr
```

* And create a folder for experiments inside:

```bash
cd pytorch-ocr && mkdir experiments
```


### Tensorboard (optional)

Besides logging to stdout, the training scripts generates (through TensorboardX) a visual trace of loss and accuracy that can been visualized with Tensorboard. 

To visualize it, first install tensorboard with:

```bash
conda install -c anaconda tensorboard
```

And then run it on the experiments folder, such as:

```bash
tensorboard --logdir=pytorch-ocr/experiments 
```

The UI will default to port 6006, which must be open to incoming TCP connections.

## Training

Training support a set of *hyperparameters* specified in *.json* file, plus a set of *arguments* specified on the command line for things such as I/O locations, picking between training or evaluation, exporting weights etc. Both are specified in their respective sections below. You will also find a few examples for different types of runs.

### Architecture

The supported architecture is composed of a single recurrent layer, a single fully connected layer, a batch normalization step (between the recurrent and the fully connected layers), and respectively a CTC decoder+loss layer for training and a greedy decoder for evaulation.

The naming for experiments has the following convention. For an architecture such as *QLSTM128_W2B8A4I32_FC_W32B32* we have:

* *Q* stands for quantized neuron.
* *LSTM* is the type of recurrent neuron.
* *128* is the number of neurons.
* *W2* is the number of bits for weights.
* *B8* is the number of bits for the recurrent bias.
* *A4* is the number of bits for the recurrent activation (= hidden state).
* *I32* is the number of bits for the internal activations (= sigmoid and tanh).
* *FC_W32A32* are the number of bits allocated to the weights and activations of the fully connected layer.


### Arguments

Hyperparameters, arguments, output logs and a tensorboard trace are persisted to disk as a reference during training, unless `--dry_run` is specified.

Training supports *resuming* or *retraining* from a checkpoint with the argument `-i` (to specify the input checkpoint path) and the argument `--pretrained_policy` (to specify `RESUME` or `RETRAIN`). 
Resuming ignores the `default_trainer_params.json` file and reads the training hyperparameters from within the checkpoint, unless a different `.json` file is specified with the `-p` argument. 
Retraining ignores the hyperparamters found within the checkpoint, as well as the optimizer state, and reads the hyperparameters from the `.json` file, either the default one or one specified with the `-p` argument.

Export an appropriate pretrained model to an HLS-friendly header with argument `--export`, and optional arguments `--simd_factor` to specify a scaling factor for the unrolling *within* a neuron (`1` is full unrolling, `2` is half unrolling etc.) and `--pe` to specify the number of processing elements allocated to compute a neuron.

More arguments and their usage can be found by invoking the helper.

### Hyperparameters 

The support hyperparameters with their default values are (taken from *default_trainer_params.json* with added comments):

    "random_seed": 123456, # Seed value to init all the randomness, to enforce reproducibility
    "batch_size" : 32, # Training batch size
    "num_workers" : 0, # CPU workers to prepare training batch, has to be 0 on Python 2.7
    "layer_size" : 128, # Number of neurons in a direction of the recurrent neuron
    "neuron_type" : "QLSTM", # Type of recurrent neurons, tested with LSTM (for Pytorch default backend), or QLSTM (for pytorch-quantization backend)
    "target_height" : 32, # Height to which the dataset images are resized, translates to input size of the recurrent neuron
    "epochs" : 4000, # Number of training epochs
    "lr" : 1e-4, # Starting learning rate
    "lr_schedule" : "FIXED", # Learning rate policy, allowed values are STEP/FIXED
    "lr_step" : 40, # Step size in number of epochs for STEP lr policy
    "lr_gamma" : 0.5, # Gamma value for STEP lr policy
    "max_norm" : 400, # Max value for gradient clipping
    "seq_to_random_threshold": 20, # Number epochs after which the training batches switch from being taken in increasing order of sequence length (where sequence length means image width for OCR) to being sampled randomly
    "bidirectional" : true, # Enable bidirectional recurrent layer
    "reduce_bidirectional": "CONCAT", # How to reduce the two output sequences coming out of a bidirectional (if enabled) recurrent layer, allowed values are SUM/CONCAT 
    "recurrent_bias_enabled": true, # Enable bias in reccurent layer
    "checkpoint_interval": 10, # Internal in number of epochs after which a checkpoint of the model is saved
    "recurrent_weight_bit_width": 32, # Number of bits to which the recurrent layer's weights are quantized
    "recurrent_weight_quantization": "FP", # Quantization strategy for the recurrent layer's weights
    "recurrent_bias_bit_width": 32, # Number of bits to which the recurrent layer's bias is quantized
    "recurrent_bias_quantization": "FP", # Quantization strategy for the recurrent layer's bias
    "recurrent_activation_bit_width": 32, # Number of bits to which the recurrent layer's activations (along both the recurrency and the output path) are quantized
    "recurrent_activation_quantization": "FP", # Quantization strategy for the recurrent layer's activation
    "internal_activation_bit_width": 32, # Number of bits to which the recurrent layer internal non-linearities (sigmoid and tanh) are quantized
    "fc_weight_bit_width": 32, # Number of bits to which the fully connected layer's weights are quantized
    "fc_weight_quantization": "FP", # Quantization strategy for the fully connected layer's weights
    "fc_bias_bit_width": 32, # Number of bits to which the fully connected layer's bias is quantized
    "fc_bias_quantization": "FP", # Quantization strategy for the fully connected layer's bias
    "quantize_input": true, # Quantize the input according to the recurrent_activation bit width and quantization
    "mask_padded": true, # Mask output values coming from padding of the input sequence
    "prefused_bn_fc": false # Signal that batch norm and the fully connected layer have to be considered as fused already, so that the batch norm step is skipped.

Currently dropout is not implemented, which is why the best set of weights (w.r.t. validation accuracy) is tracked and saved to disk at every improvement (with a single epoch granularity).

### Strategy

Training a model that can be reproduced accurately in hardware requires at least one retraining step. The reason is that the batch norm coefficients have to be fused into the quantized fully connected layer, since batch norm is not implemented in hardware.
The suggested strategy goes as follow:

* Train with quantized input, quantized recurrent weights/activations/bias, full-precision internal activations and full-precision fully connected layer, either from scratch or from a pretrained full precision model.
* Perform batch norm-fully connected fusion and retrain with quantized internal activations and quantized fully connected layer.

### Example: Full Precision BiLSTM

To start training a full-precision model with default hyperparameters, simply run:

```bash
python main.py
```

### Example: Quantized W2A4 BiLSTM

```bash
python main.py -p quantized_params/QLSTM128_W2B8A4I32_FC_W32B32.json
```
