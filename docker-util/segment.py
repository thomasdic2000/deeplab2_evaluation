"""
Script for running inference via docker.
Docker passes parameters through environment variables.
"""
import os
import sys
import copy
import logging

# caffe logging
os.environ['GLOG_minloglevel'] = '2' 
# propviz logging
logging.basicConfig(level=int(os.environ['LOGLEVEL']), stream=sys.stdout)

import scipy
import numpy as np

from propviz.caffe import Segmenter
from propviz import metrics
from propviz.drawing import visualize_detection


def deeplabv2_resnet101_preprocess(img):
    """
    Transforms image into a data blob of the right size and shape.

    img: 500x500x3 image

    Does not currently support batch transformations due to deeplabv2/resnet101
    memory requirements (~7.5GB for single image inference).
    """
    # means are from deeplabv2 prototxt
    #mean = np.array([104.008, 116.669, 122.675], dtype=np.float32).reshape(1,1,3)
    # reverse rgb?
    img = img[:,:,::-1]  
    # mean transform
    #img = img - mean
    # right/bottom zero pad to 513x513x3
    img = np.pad(img, ((0,13), (0,13), (0,0)), 'constant')
    # HxWx3 -> 3xHxW
    img = img.transpose((2, 0, 1)) 
    # simulate minibatch, 3xHxW -> 1x3xHxW, make float
    img = img[np.newaxis,:].astype(np.float32) 

    return img


def main():
    # env vars passed through docker
    net = Segmenter(proto=os.environ['PROTOTXT'],
                    model=os.environ['MODEL'],
                    gpu_id=0,
                    preprocess=deeplabv2_resnet101_preprocess)


    files = open(os.environ['LIST_PART'], 'r').readlines()

    img_sz = 500 # TODO: put this somewhere else

    predictions = np.zeros((len(files), img_sz, img_sz)).astype(np.uint8)
    gt = np.zeros((len(files), img_sz, img_sz)).astype(np.uint8)

    log = open(os.environ['TMP'] + '/log.txt', 'w')

    for count, line in enumerate(files):
        print(line)

        img_fn, label_fn = line.split()

        img,label = scipy.misc.imread(img_fn), scipy.misc.imread(label_fn)

        pred = net.predict(img, os.environ['OUT_LAYER'])

        # crop back to gt size
        pred = pred.data.squeeze()[:img_sz,:img_sz].astype(np.uint8)

        predictions[count,:,:] = pred
        gt[count,:,:] = label


        # providing a tmp dir path will pngs of segmentation results
        if 'TMP' in os.environ.keys():
            img_name = img_fn.split('/')[-1]
            out_fn = os.path.join(os.environ['TMP'], img_name)
            visualize_detection(pred, img, out_fn, 1)

            out_fn = out_fn.replace(".png", "_gt.png")
            visualize_detection(label, img, out_fn, 1)
            
            try:
                log.write('{}, {}, {}\n'.format(out_fn, *metrics.iou(pred, label).values()))
                log.flush()
            except:
                import pdb; pdb.set_trace()

        if 'LIMIT' in os.environ.keys() and count >= int(os.environ['LIMIT']):
            break

    print(metrics.iou(predictions, gt))


if __name__ == '__main__':
    main()
