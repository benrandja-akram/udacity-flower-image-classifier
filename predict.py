import argparse 
import time
import json
import sys

import torch 
from torch import nn, optim
from torchvision import datasets, models, transforms
import numpy as np
from PIL import Image


def parse_terminal_args():

    parser = argparse.ArgumentParser(description='Classify flowers images !')
    parser.add_argument('input', help='input file to be classified (required)')
    parser.add_argument('checkpoint', help='Classification model (required)')
    parser.add_argument('--top_k', help='top k categories number [default 5].')
    parser.add_argument('--category_names', help='category names file')
    parser.add_argument('--gpu', action='store_true', help='activate cuda')
    args = parser.parse_args()
    return args


def load_classification_model():
    info = torch.load(args.checkpoint)
    model = info['model']
    model.classifier = info['classifier']
    model.load_state_dict(info['state_dict'])
    return model


def process_image(image):
    im = Image.open(image)
    width, height = im.size
    picture_coords = [width, height]
    max_span = max(picture_coords)
    max_element = picture_coords.index(max_span)
    if (max_element == 0):
        min_element = 1
    else:
        min_element = 0
    aspect_ratio=picture_coords[max_element]/picture_coords[min_element]
    new_picture_coords = [0,0]
    new_picture_coords[min_element] = 256
    new_picture_coords[max_element] = int(256 * aspect_ratio)
    im = im.resize(new_picture_coords)   
    width, height = new_picture_coords
    left = (width - 244)/2
    top = (height - 244)/2
    right = (width + 244)/2
    bottom = (height + 244)/2
    im = im.crop((left, top, right, bottom))
    np_image = np.array(im)
    np_image = np_image.astype('float64')
    np_image = np_image / [255,255,255]
    np_image = (np_image - [0.485, 0.456, 0.406])/ [0.229, 0.224, 0.225]
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def classify_image_file(path, topk=5):
    topk=int(topk)
    with torch.no_grad():
        image = process_image(path)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.float()
        model = load_classification_model()
        if (args.gpu):
           image = image.cuda()
           model = model.cuda()
        else:
            image = image.cpu()
            model = model.cpu()
        result = model(image)
        probs, classes = torch.exp(result).topk(topk)
        probs, classes = probs[0].tolist(), classes[0].add(1).tolist()
        results = zip(probs,classes)
        return results
    
    
def open_categories_file():
    if (args.category_names is not None):
        file = args.category_names 
        data = json.loads(open(file).read())
        return data
    return None

def show_prediction_result(results):
    file = open_categories_file()
    i = 0
    for a, b in results:
        i = i + 1
        a = str(round(a,4) * 100.) + '%'
        if (file):
            b = file.get(str(b),'None')
        else:
            b = ' class {}'.format(str(b))
        print("{}.{} ({})".format(i, b, a))
    return None




if __name__ == '__main__':
    args = parse_terminal_args() 
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but no GPU detected")
    if (args.top_k is None):
        top_k = 5
    else:
        top_k = args.top_k
    image_path = args.input
    prediction = classify_image_file(image_path,top_k)
    show_prediction_result(prediction)
