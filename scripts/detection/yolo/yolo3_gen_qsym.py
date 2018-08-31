# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import os
import logging
import mxnet as mx
from mxnet.contrib.quantization import quantize_model

from mxnet import gluon

from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform


COCO_ROOT_DIR = "/lustre/atlas/zhengdc/data/cv/coco/"


def get_yolo_dataiter(batch_size, num_calib_batches, data_shape=416, num_workers=16):
    val_dataset = gdata.PersonDetection(root=COCO_ROOT_DIR, splits='person_val2017', skip_empty=False)
    width, height = data_shape, data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        2 * batch_size * num_calib_batches, False, last_batch='keep', num_workers=num_workers, batchify_fn=batchify_fn,)

    data, _ = next(iter(val_loader))
    return mx.io.NDArrayIter(data, batch_size=batch_size, label_name=None)


def save_symbol(fname, sym, logger=None):
    if logger is not None:
        logger.info('Saving symbol into file at {}'.format(fname))
    sym.save(fname)


def save_params(fname, arg_params, aux_params, logger=None):
    if logger is not None:
        logger.info('Saving params into file at {}'.format(fname))
    save_dict = {('arg:{}'.format(k)): v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:{}'.format(k)): v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
    mx.nd.save(fname, save_dict)


ctx = mx.gpu(0)

logging.basicConfig()
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

prefix, epoch = "yolo3_person", 0
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

batch_size = 64
logger.info('batch size = {} for calibration'.format(batch_size))

calib_mode = 'naive'
#calib_mode = 'entropy'
logger.info('calibration mode set to {}'.format(calib_mode))

num_calib_batches = 20
if calib_mode != 'none':
    logger.info('number of batches = {} for calibration'.format(num_calib_batches))

data_nthreads = 16

#quantized_dtype = "int8"
quantized_dtype = "uint8"

exclude_first_conv = True
excluded_sym_names = ["yolov30_yolooutputv30_conv0_fwd", "yolov30_yolooutputv31_conv0_fwd", "yolov30_yolooutputv32_conv0_fwd"]
calib_layer = lambda name: name.endswith('_output') and (name.find('conv') != -1 or name.find('fc') != -1)
if exclude_first_conv:
    excluded_sym_names += ['darknetv30_conv0_fwd']

if calib_mode == 'none':
    qsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                   ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                   calib_mode=calib_mode, quantized_dtype=quantized_dtype,
                                                   logger=logger)
    sym_name = '{}-{}-{}-symbol.json'.format(prefix, 'quantized', quantized_dtype)
    save_symbol(sym_name, qsym, logger)
    param_name = '{}-{}-{}-{}.params'.format(prefix, 'quantized', quantized_dtype, "0000")
else:
    data = get_yolo_dataiter(batch_size, num_calib_batches)
    cqsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params, label_names=None,
                                                    ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                    calib_mode=calib_mode, calib_data=data,
                                                    num_calib_examples=num_calib_batches * batch_size,
                                                    calib_layer=calib_layer, quantized_dtype=quantized_dtype,
                                                    logger=logger)
    if calib_mode == 'entropy':
        suffix = 'quantized-{}batches-entropy'.format(num_calib_batches)
    elif calib_mode == 'naive':
        suffix = 'quantized-{}batches-naive'.format(num_calib_batches)
    else:
        raise ValueError('unknow calibration mode {} received, only supports `none`, `naive`, and `entropy`'.format(calib_mode))

    sym_name = '{}-{}-{}-symbol.json'.format(prefix, suffix, quantized_dtype)
    save_symbol(sym_name, cqsym, logger)
    param_name = '{}-{}-{}-{}.params'.format(prefix, suffix, quantized_dtype, "0000")

save_params(param_name, qarg_params, aux_params, logger)
