from PIL import Image
import numpy as np
import os

#DETAILS OF THE IMAGES
folder = 'original'
base_name = 'image_'
file_ext = '.tif'
N = 19
filenumbers = range(N)

save_image_folder = 'cropped'

if not os.path.exists(save_image_folder):
    os.makedirs(save_image_folder)

#LOOP ON EVERY IMAGE
for i in filenumbers:
    i = str(i)
    print(i)
    
    #OPEN .tif FILE
    filename = folder + '/' + base_name + i + file_ext
    image_tiff = Image.open(filename)
    
    imarray = np.array(image_tiff)
    no_of_rows, no_of_columns, rgba = imarray.shape
    
    imarrayT = imarray.transpose((1, 0, 2))
      
    #CHOOSE THE HORIZONTAL PIXELS TO KEEP (THE REST ARE CROPPED AWAY)
    min_column = 600
    max_column = 700

    #CROP THE IMAGE AND SAVE THE CROPPED IMAGE
    new_imarrayT = imarrayT[min_column:max_column]
    new_imarray = new_imarrayT.transpose((1, 0, 2))
    new_image = Image.fromarray(new_imarray, mode='RGBA') # float32
    new_image.save(save_image_folder+"/new_image_%s.tif" %i, "TIFF")

