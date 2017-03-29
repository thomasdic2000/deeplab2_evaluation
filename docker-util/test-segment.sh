#!/usr/bin/env bash

export LOGLEVEL="10"
export PROTOTXT="/vol/experiments/deeplab/roof-and-pool/config/roof-only-resnet101-test.prototxt"
export MODEL="/vol/experiments/deeplab/roof-and-pool/model/roof-only-resnet101/train_iter_15000.caffemodel"
export LIST_PART="/vol/data/roof-only/lists/all-data-test.txt"
export OUT_LAYER="crf_inf_argmax"
export LIMIT=5
export TMP="/vol/tmp/"

/bin/bash segment.sh
