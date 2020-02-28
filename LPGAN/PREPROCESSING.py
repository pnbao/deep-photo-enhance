from .DATA import *
from PIL import Image
from numpy import array

def get_normalize_size_shape_method(img, max_length):
    [ height, width, channels ] = img.shape
    if height >= width:
        longerSize = height
        shorterSize = width
    else:
        longerSize = width
        shorterSize = height

    scale = float(max_length) / float(longerSize)

    outputHeight = int(round(height*scale))
    outputWidth = int(round(width*scale))
    return outputHeight, outputWidth

def cpu_normalize_image(img, max_length):
    outputHeight, outputWidth = get_normalize_size_shape_method(img, max_length)
    outputImg = Image.fromarray(img)
    outputImg = outputImg.resize((outputWidth, outputHeight), Image.ANTIALIAS)
    outputImg = array(outputImg)
    return outputImg
