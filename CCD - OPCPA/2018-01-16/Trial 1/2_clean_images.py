from PIL import Image
import numpy as np
import os

def _big(fft2, xlen, ylen, minI):
    """
    Parameters:
    fft2: A 2D array of the intensity values after the fourier transform.
    xlen: Number of pixels horizontally
    ylen: Number of pixels vertically
    minI: All intensities above this value are recorded

    Return:
    I: Intensity value that's kept
    loc: Index location of the intensity value in [i, j]
    """

    I = []
    loc = []
    for i in range(ylen):
        for j in range(xlen):
            if fft2[i][j] > minI:
                I.append(fft2[i][j])
                loc.append([i, j])
    return I, loc

def _make_2DGaussian(A, y_sd, x_sd, y0_i, x0_i, xlen, ylen):
    """
    Parameters:
    A: The multiple of a normalised Gaussian
    y_sd: Standard deviation of vertical Gaussian
    x_sd: Standard deviation of horizontal Gaussian
    y0_i: The vertical index of the Gaussian centre
    x0_i: The horizontal index of the Gaussian centre
    xlen: Number of pixels horizontally
    ylen: Number of pixels vertically
    """

    #MAKES A LINEARLY INCREASING ARRAY WITH 0 AT y0_i, x0_i AND WITH SPACING OF 1
    x = np.arange(-x0_i, xlen-x0_i)
    y = np.arange(-y0_i, ylen-y0_i)

    #FUNCTION OF 1D GAUSSIAN
    gaussian = lambda A, sd, u: np.sqrt(A)*(1/(sd*np.sqrt(2*np.pi)))*np.exp(-0.5*(u/sd)**2)

    #MAKES 2D GAUSSIAN CENTRED AT y0_i, x0_i
    gaussian2D = np.zeros((len(y), len(x)))
    for i in range(len(x)):
        for j in range(len(y)):
            gaussian2D[j][i] = gaussian(A, y_sd, y[j])*gaussian(A, x_sd, x[i])
    return gaussian2D

def _make_ellipse(A, y_rad, x_rad, y_i, x_i, xlen, ylen):
    """
    Parameters:
    A: The amplification of the mask
    y_rad: Max vertical distance from eclipse centre
    x_rad: Max horizontal distance from eclipse centre
    y_i: The vertical index of the eclipse centre
    x_i: The horizontal index of the eclipse centre
    xlen: Number of pixels horizontally
    ylen: Number of pixels vertically
    """

    #MAKES A LINEARLY INCREASING ARRAY WITH 0 AT y0_i, x0_i AND WITH SPACING OF 1
    x = np.arange(-x_i, xlen-x_i)
    y = np.arange(-y_i, ylen-y_i)

    #EQUATION FOR ECLIPSE
    ellipse_eq = lambda u, u0, u_rad, v, v0, v_rad: (u/u_rad)**2 + (v/v_rad)**2 - 1

    #CREATE THE ECLIPSE MASK CENTRED AT y_i, x_i. EVERYTHING INSIDE THE ECLIPSE GETS AMPLIFIED BY a
    ellipse = np.ones((len(y), len(x)))
    for i in range(xlen):
        for j in range(ylen):
            ellipse_val = ellipse_eq(x[i], x_i, x_rad, y[j], y_i, y_rad)
            if ellipse_val < 0:
                ellipse[j][i] = A
    return ellipse

def _applymask(mask_func, fft2, points_loc, xlen, ylen, x_sd, y_sd, A):
    """
    Parameters:
    mask_func: The mask function defined above
    fft2: A 2D array of the intensity values after the fourier transform.
    points_loc: The location where the highest intensity values are in fourier space.
    xlen: Number of pixels horizontally
    ylen: Number of pixels vertically
    """

    #ALGORITHM TO APPLY MASK TO AMPLIFY THE HIGHEST INTENSITY VALUES
    mask_list = []
    for y_i, x_i in points_loc:
        mask = mask_func(A, y_sd, x_sd, y_i, x_i, xlen, ylen)
        mask_list.append(mask)
        
    mask = sum(mask_list)
    newfft2 = mask*fft2

    #RETURNS A NEW 2D ARRAY IN FOURIER SPACE AND THE 2D ARRAY OF THE MASK
    return newfft2, mask

def clean_image(image, mask_func, x_sd, y_sd, A, r = 100):
    #FFT THE IMAGE
    fft2 = np.fft.fft2(image)
    absfft2 = np.abs(fft2)
    
    #FIND THE LOCATIONS THAT HAVE HIGH INTENSITY IN FFT IMAGE
    ylen, xlen = absfft2.shape
    points, points_loc = _big(absfft2, xlen, ylen, 1e6) #points_loc = [[y_i, x_i], ....]

    #REMOVE SOME OF THE HIGH INTENSITY POINTS
#    new_points_loc = []
#    for u in range(len(points_loc)):
#        i, j = points_loc[u]
#        if i<r and j<r:
#            None
#        else:
#            new_points_loc.append(points_loc[u])
#    points_loc = new_points_loc
#    print(points_loc)
    
    #APPLY MASK TO EVERY HIGH INTENSITY POINT ON FOURIER IMAGE
    newfft2, mask = _applymask(mask_func, fft2, points_loc, xlen, ylen, x_sd, y_sd, A)
    absnewfft2 = abs(newfft2)
    newimage = np.fft.ifft2(newfft2)
    absnewimage = abs(newimage)

    return fft2, absfft2, newfft2, absnewfft2, newimage, absnewimage, mask

#INFORMATION OF THE IMAGES
folder = 'cropped'
base_name = 'new_image_'
file_ext = '.tif'
N = 19
filenumbers = range(N)

save_image_folder = 'clean2'

if not os.path.exists(save_image_folder):
    os.makedirs(save_image_folder)

#LOOP ALL IMAGES
for k in filenumbers:
    k = str(k)
    print(k)
    
    #OPEN .tif FILE
    file_path = folder + '/' + base_name + k + file_ext
    image = np.asarray(Image.open(file_path).convert('L'))
    
    #DEFINE PROPERTIES FOR MASKING
    ylen, xlen = image.shape
    x_sd = xlen/100.
    y_sd = ylen/100.
    A = 1000.

    #DEFINE THE MASK FUNCTION
    mask_func = _make_2DGaussian
    #mask_func = _make_ellipse

    #APPLY MASK TO THE IMAGE TO CLEAN (CAN REPEATEDLY CLEAN THE IMAGE)
    fft2, absfft2, newfft2, absnewfft2, newimage, absnewimage, mask = clean_image(image, mask_func, x_sd, y_sd, A)
    #fft2, absfft2, newfft2, absnewfft2, newimage, absnewimage, mask = clean_image(newimage, mask_func, x_sd, y_sd, A)
    #fft2, absfft2, newfft2, absnewfft2, newimage, absnewimage, mask = clean_image(newimage, mask_func, x_sd, y_sd, A)

    #CREATE A NEW ARRAY OF THE CLEANED COLOURED IMAGE TO SAVE AS .tif FILE
    absnewimage = absnewimage/absnewimage.max()*255 
    absnewimage = absnewimage.astype(int)
    new_array = np.zeros((ylen,xlen,4), dtype='uint8')
    for i in range(ylen):
        for j in range(xlen):
            new_array[i][j] = [0,absnewimage[i][j],0,255]

    #CONVERT ARRAY INTO IMAGE AND SAVE IT.
    new_image = Image.fromarray(new_array, mode='RGBA') # float32
    new_image.save(save_image_folder+"/new_image_%s.tif" %k, "TIFF")



