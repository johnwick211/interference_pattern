from PIL import Image
import numpy as np

def crop_algorithm(filename, min_column=600, max_column=700):
    #OPEN .tif FILE
    image_tiff = Image.open(filename)
    
    imarray = np.array(image_tiff)
    no_of_rows, no_of_columns, rgba = imarray.shape
    
    imarrayT = imarray.transpose((1, 0, 2))

    #CROP THE IMAGE AND SAVE THE CROPPED IMAGE
    new_imarrayT = imarrayT[min_column:max_column]
    new_imarray = new_imarrayT.transpose((1, 0, 2))
    new_image = Image.fromarray(new_imarray, mode='RGBA') # float32
    return new_image