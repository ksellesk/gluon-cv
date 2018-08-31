from __future__ import division
from __future__ import print_function

import sys
import argparse
import logging
logging.basicConfig(level=logging.INFO)
import time
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet import nd
from mxnet import gluon
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric

import cv2

COCO_ROOT_DIR = "/lustre/atlas/zhengdc/data/cv/coco/"

COCO_CLASSES = ['person']


def parse_args():
    parser = argparse.ArgumentParser(description='Eval YOLO networks.')
    parser.add_argument('--network', type=str, default='darknet53',
                        help="Base network name")
    parser.add_argument('--algorithm', type=str, default='yolo3',
                        help='YOLO version, default is yolo3')
    parser.add_argument('--data-shape', type=int, default=416,
                        help="Input data shape")
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    args = parser.parse_args()
    return args

def get_dataset(dataset, data_shape):
    if dataset.lower() == 'voc':
        val_dataset = gdata.VOCDetection(splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        #val_dataset = gdata.PersonDetection(root=COCO_ROOT_DIR, splits='person_train2017', skip_empty=False)
        #val_dataset = gdata.COCODetection(root=COCO_ROOT_DIR, splits='instances_val2017', skip_empty=False)
        val_dataset = gdata.PersonDetection(root=COCO_ROOT_DIR, splits='person_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(data_shape, data_shape))
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric

def get_dataloader(val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size, False, last_batch='keep', num_workers=num_workers, batchify_fn=batchify_fn,)
    return val_loader

def validate(mod, val_data, ctx, classes, size, metric):
    """Test on validation dataset."""
    metric.reset()
    count = 0
    with tqdm(total=size) as pbar:
        tic = time.time()
        num = 0
        for batch in val_data:
            b_inputs = mx.io.DataBatch(data=(batch[0],))
            b_labels = batch[1]

            mod.forward(b_inputs, is_train=False)
            det_ids, det_scores, det_bboxes = mod.get_outputs()

            gt_bboxes = b_labels.slice_axis(axis=-1, begin=0, end=4)
            gt_ids = b_labels.slice_axis(axis=-1, begin=4, end=5)

            print(det_bboxes.shape, det_ids.shape, det_scores.shape, gt_bboxes.shape, gt_ids.shape)

            metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids)
            pbar.update(batch[0].shape[0])
            #pbar.update(1)
            count += 1
            #if count == 10:
            #    break
    return metric.get()

if __name__ == '__main__':
    args = parse_args()

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    prefix, epoch = "yolo3_person-quantized-20batches-naive", 0
    #prefix, epoch = "yolo3_person-quantized", 0
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod.bind(for_training=False,
             data_shapes=[('data', (args.batch_size, 3, args.data_shape, args.data_shape))])
    mod.set_params(arg_params, aux_params)

    # training data
    val_dataset, val_metric = get_dataset(args.dataset, args.data_shape)
    #for idx, i in enumerate(val_dataset):
    #    assert len(i) == 2
    #    print(type(i))
    #    print(type(i[0]))
    #    print(i[0].shape)
    #    print(i[1].shape)
    #    if i[1].shape[0] == 14:
    #        cv2.imwrite("test{}.jpg".format(idx), i[0].asnumpy())
    #    #input()
    val_data = get_dataloader(
        val_dataset, args.data_shape, args.batch_size, args.num_workers)
    #print(val_data)
    #for b in val_data:
    #    assert len(b) == 2
    #    print(type(b))
    #    print(type(b[0]))
    #    print(type(b[1]))
    #    print(b[0].shape)
    #    print(b[1].shape)
    #    input()
    #input()
    #sys.exit()

    classes = val_dataset.classes  # class names

    # training
    names, values = validate(mod, val_data, ctx, classes, len(val_dataset), val_metric)
    for k, v in zip(names, values):
        print(k, v)
