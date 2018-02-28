from PIL import Image
import numpy as np
import os

#DETAILS OF THE IMAGES
folder = 'original'
base_name = 'image_'
file_ext = '.tif'
N = 91
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
    
    #CHOOSE THE VERTICLE PIXELS TO KEEP (THE REST ARE CROPPED AWAY)
    min_row = -11
    max_row = -1

    #CROP THE IMAGE AND SAVE THE CROPPED IMAGE
    new_imarray = imarray[min_row:max_row]
    new_image = Image.fromarray(new_imarray, mode='RGBA') # float32
    new_image.save(save_image_folder+"/new_image_%s.tif" %i, "TIFF")

