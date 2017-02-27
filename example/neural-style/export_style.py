from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import find_mxnet
import mxnet as mx
import numpy as np
import importlib
import logging
logging.basicConfig(level=logging.DEBUG)
import argparse
from collections import namedtuple
from skimage import io, transform
from skimage.restoration import denoise_tv_chambolle

from nstyle import *
from scipy.io import savemat
import os

def main():
    Executor = namedtuple('Executor', ['executor', 'data', 'data_grad'])
    model_module =  importlib.import_module('model_vgg19')
    style, content = model_module.get_symbol()    

    # input
    gpu=-1
    dev = mx.gpu(gpu) if gpu >= 0 else mx.cpu()
    base='/Users/jxy198/Downloads/cubism_images/'
    painting_download_list=base + 'cubism_painting_download_list.txt'
    painting_path=base + 'cubism_paintings/'
    save_path=base + 'cubism_styles/'
    try:
        os.mkdir(save_path)
        os.mkdir(save_path + 'relu1_1')
        os.mkdir(save_path + 'relu2_1')
        os.mkdir(save_path + 'relu3_1')
        os.mkdir(save_path + 'relu4_1')
        os.mkdir(save_path + 'relu5_1')
    except FileExistsError:
        pass
    

    with open(painting_download_list) as f:
        paintings=f.readlines()

    filename=lambda x: x[-2]+'_'+x[-1]        
    paintings = [filename(line.strip().split("/")) for line in paintings]
    for image in paintings:
        if os.path.exists(save_path + 'relu5_1/' + image + '.mat'):
            continue
        else:
            print(image)
        style_np = PreprocessContentImage(painting_path + image, 600)
        size=style_np.shape[2:]
    
        # model
        model_executor = model_module.get_executor(style, content, size, dev)
        model_executor.data[:] = style_np
        model_executor.executor.forward()

        style_array = []
        for i in range(len(model_executor.style)):
            style_array.append(model_executor.style[i].copyto(mx.cpu()))

            
        savemat(save_path + 'relu1_1/' + image + '.mat', {'relu1_1':style_array[0].asnumpy()})
        savemat(save_path + 'relu2_1/' + image + '.mat', {'relu2_1':style_array[1].asnumpy()})
        savemat(save_path + 'relu3_1/' + image + '.mat', {'relu3_1':style_array[2].asnumpy()})
        savemat(save_path + 'relu4_1/' + image + '.mat', {'relu4_1':style_array[3].asnumpy()})
        savemat(save_path + 'relu5_1/' + image + '.mat', {'relu5_1':style_array[4].asnumpy()})

if __name__ == "__main__":
    main()
