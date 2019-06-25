# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
from builtins import range
import os
import sys
import numpy as np

FLAGS = None
np_dtype_string = np.dtype(object)

def np_to_model_dtype(np_dtype):
    if np_dtype == np.bool:
        return "TYPE_BOOL"
    elif np_dtype == np.int8:
        return "TYPE_INT8"
    elif np_dtype == np.int16:
        return "TYPE_INT16"
    elif np_dtype == np.int32:
        return "TYPE_INT32"
    elif np_dtype == np.int64:
        return "TYPE_INT64"
    elif np_dtype == np.uint8:
        return "TYPE_UINT8"
    elif np_dtype == np.uint16:
        return "TYPE_UINT16"
    elif np_dtype == np.float16:
        return "TYPE_FP16"
    elif np_dtype == np.float32:
        return "TYPE_FP32"
    elif np_dtype == np.float64:
        return "TYPE_FP64"
    elif np_dtype == np_dtype_string:
        return "TYPE_STRING"
    return None

def np_to_torch_dtype(np_dtype):
    if np_dtype == np.bool:
        return torch.bool
    elif np_dtype == np.int8:
        return torch.int8
    elif np_dtype == np.int16:
        return torch.int16
    elif np_dtype == np.int32:
        return torch.int
    elif np_dtype == np.int64:
        return torch.long
    elif np_dtype == np.uint8:
        return torch.uint8
    elif np_dtype == np.uint16:
        return None # Not supported in Torch
    elif np_dtype == np.float16:
        return None
    elif np_dtype == np.float32:
        return torch.float
    elif np_dtype == np.float64:
        return torch.double
    elif np_dtype == np_dtype_string:
        return None # Not supported in Torch

def create_libtorch_modelfile(
        models_dir, model_version, io_cnt, max_batch, dtype, shape):

    # For now, only generate model for one input / output pair
    if io_cnt != 1:
        return

    if not tu.validate_for_libtorch_model(dtype, dtype, dtype, shape, shape, shape):
        return

    torch_dtype = np_to_torch_dtype(dtype)

    model_name = tu.get_zero_model_name("libtorch_nobatch" if max_batch == 0 else "libtorch",
                                   io_cnt, dtype)
    # handle for -1 (when variable) since can't create tensor with shape of [-1]
    torch_shape = [abs(s) for s in shape]
    # Create the model (only one io pair)
    class IdentityNet(nn.Module):
        def __init__(self):
            super(IdentityNet, self).__init__()
        def forward(self, input):
            return input
    identityModel = IdentityNet()
    example_input = torch.zeros(torch_shape, dtype=torch_dtype)
    traced = torch.jit.trace(identityModel, example_input)

    model_version_dir = models_dir + "/" + model_name + "/" + str(model_version)

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass # ignore existing dir

    traced.save(model_version_dir + "/model.pt")


def create_libtorch_modelconfig(
        models_dir, model_version, io_cnt, max_batch, dtype, shape):

    # For now, only generate model for one input / output pair
    if io_cnt != 1:
        return

    if not tu.validate_for_libtorch_model(dtype, dtype, dtype, shape, shape, shape):
        return

    # Unpack version policy
    version_policy_str = "{ latest { num_versions: 1 }}"

    # Use a different model name for the non-batching variant
    model_name = tu.get_zero_model_name("libtorch_nobatch" if max_batch == 0 else "libtorch",
                                   io_cnt, dtype)
    config_dir = models_dir + "/" + model_name
    config = '''
name: "{}"
platform: "pytorch_libtorch"
max_batch_size: {}
version_policy: {}
input [
  {{
    name: "INPUT__0"
    data_type: {}
    dims: [ {} ]
  }}
]
output [
  {{
    name: "OUTPUT__0"
    data_type: {}
    dims: [ {} ]
  }}
]
'''.format(model_name, max_batch, version_policy_str,
           np_to_model_dtype(dtype), tu.shape_to_dims_str(shape),
           np_to_model_dtype(dtype), tu.shape_to_dims_str(shape))

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Top-level model directory')
    parser.add_argument('--libtorch', required=False, action='store_true',
                        help='Generate Pytorch LibTorch models')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.libtorch:
        import torch
        from torch import nn

    import test_util as tu

    # Only generating PyTorch identity model with float32, no batching,
    # and one variable size demension. Same models for other frameworks
    # are generated as part of gen_qa_zero_models
    create_libtorch_modelfile(FLAGS.models_dir, 1, 1, 0, np.float32, [-1])
    create_libtorch_modelconfig(FLAGS.models_dir, 1, 1, 0, np.float32, [-1])
