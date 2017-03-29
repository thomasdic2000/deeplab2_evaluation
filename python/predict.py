#!/usr/bin/env python
# based on coce by Martin Kersner, m.kersner@gmail.com


import caffe

from PIL import Image

class Segmenter(caffe.Net):
""" Extension of caffe.Net for deeplab v2 prediction """
    def __init__(self, prototxt_path, weights_path, gpu_id=0, img_size=500):
    	caffe.Net.__init__(self, prototxt_path, weights_path)
	self.set_mode_gpu()
	self.set_device(gpu_id)
        self.img_size = img_size


    def predict(img_path=''):
        img, cur_h, cur_w = preprocess_image(img_path)



    def preprocess_image(img_path):
        input_image = 255 * caffe.io.load_image(img_path)
        # what???
        image = Image.fromarray(np.unit8(input_image))
        image = np.array(image)

        mean_vec = np.array([103.939, 116.779, 123.69])


