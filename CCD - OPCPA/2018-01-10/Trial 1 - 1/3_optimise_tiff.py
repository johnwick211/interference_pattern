from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.animation as an
from matplotlib.widgets import Slider
from scipy import optimize
import scipy.fftpack as scfft
import pickle

def init():
    pattern.set_data([], [])
    fitted_pattern.set_data([], [])
    title11.set_text(' ')
    title12.set_text(' ')
    return pattern, fitted_pattern, title11, title12


def animate(j):
    pattern.set_data(x, intensity_values[j % no_of_images])
    fitted_pattern.set_data(x, intensity_values_fit[j % no_of_images])
    title11.set_text("Measured/Average Intensity Pattern. i = %i" % j)
    title12.set_text("Fitted Intensity Pattern. i = %i" % j)
    return pattern, fitted_pattern, title11, title12


def makefile(filename):
    # Create a file if it doesn't already exist
    with open(filename, 'w') as f:
        f.close()
    return


def largest_intensity(intensity_values):
    """
    Finds the largest intensity value within all intensity data
    """
    max_intensity = 0
    for array in intensity_values:
        if max(array) > max_intensity:
            max_intensity = max(array)
    return max_intensity


def smallest_intensity(intensity_values):
    """
    Finds the smallest intensity value within all intensity data
    """
    min_intensity = largest_intensity(intensity_values)
    for array in intensity_values:
        if min(array) < min_intensity:
            min_intensity = min(array)
    return min_intensity


def format_fn(tick_val, tick_pos):
    """
    Used in plotdata function to have y-axis ticks in 'pi' intervals.
    """
    if tick_val == np.pi:
        return "$\pi$"
    elif tick_val == -np.pi:
        return "$-\pi$"
    elif tick_val == 0:
        return "$0$"
    else:
        return "$%i\pi$" % (tick_val / np.pi)


def optimise_mean(file_path, vertical_fringes=True):
    """
    The start of the fitting algorithm.
    The data is first mean-averaged and then forwarded to the main fitting algorithm.
    """
    # OPEN .tif FILE
    image_tiff = Image.open(file_path)
    imarray = np.array(image_tiff)

    no_of_rows, no_of_columns, rgb = imarray.shape

    I = []
    
    # CONVERTS TO (no_of_columns, no_of_rows, rgb) shape
    if vertical_fringes == True:
        imarrayT = imarray.transpose((1, 0, 2))
        # CHOOSE THE VERTICLE PIXELS TO AVERAGE
        indexes_to_average = range(no_of_rows)
        x = np.arange(0, no_of_columns, 1)
        for i in range(no_of_columns):
            imarray_to_average = list(map(lambda j: imarrayT[i][j], indexes_to_average))
            I.append(np.mean(imarray_to_average))
        
    else:
        imarrayT = imarray
        indexes_to_average = range(no_of_columns)
        #x = np.linspace(0, (no_of_rows-1)*100, no_of_rows)
        x = np.arange(0, no_of_rows, 1)
        for i in range(no_of_rows):
            imarray_to_average = list(map(lambda j: imarrayT[i][j], indexes_to_average))
            I.append(np.mean(imarray_to_average))

    return optimise(I,x)

def optimise_mode(file_path, vertical_fringes=True):
    # OPEN .tif FILE
    image_tiff = Image.open(file_path)
    imarray = np.array(image_tiff)

    no_of_rows, no_of_columns, rgb = imarray.shape
    
    I = []

    # CONVERTS TO (no_of_columns, no_of_rows, rgb) shape
    if vertical_fringes == True:
        imarrayT = imarray.transpose((1, 0, 2))
        # CHOOSE THE VERTICLE PIXELS TO AVERAGE
        indexes_to_average = range(no_of_rows)
        x = np.arange(0, no_of_columns, 1)
        for i in range(no_of_columns):
            imarray_to_average = list(map(lambda j: imarrayT[i][j][1], indexes_to_average))
            I.append(hist_peak(imarray_to_average))
    else:
        imarrayT = imarray
        indexes_to_average = range(no_of_columns)
        x = np.arange(0, no_of_rows, 1)
        for i in range(no_of_rows):
            imarray_to_average = list(map(lambda j: imarrayT[i][j][1], indexes_to_average))
            I.append(hist_peak(imarray_to_average))
        
    return optimise(I,x)

def optimise(I, x):
    """
    Fits the data with an optimisation algorithm module.
    Fits wavenumber, then phase, then wavenumber, then phase...
    This is repeated 100 times to make best fit.
    """
    #PROPETIES OF THE WAVE
    offset = np.mean(I)
    #amplitude = max(I)-min(I)
    amplitude = 3*np.std(I)/(2**0.5)

    # OPTIMISATION ALGORITHM REQUIRES AN INITIAL GUESS
    guess_k = np.pi/200
    guess_phase = np.pi/4

    counter = 0
    k_val = guess_k
    phase_val = guess_phase
    while counter < 10:
        counter += 1

        # FUNCTIONS TO FIT
        optimize_func_k = lambda u: osc_func_k(u, x, amplitude, offset, k_val) - I
        u = optimize.leastsq(optimize_func_k, [phase_val], maxfev=1000)[0]
        phase_val = u[0]

        optimize_func_phase = lambda u: osc_func_phase(u, x, amplitude, offset, phase_val) - I
        u = optimize.leastsq(optimize_func_phase, [k_val], maxfev=1000)[0]
        k_val = u[0]

    u = [amplitude, k_val, phase_val, offset]
    # RETURNS THE OPTIMISED PARAMETERS IN THE FITTED FUNCTION
    return u, I, x

def fit(I, x):
    """
    Tried fitting the data by fourier transforming it.
    Fringes are too wide for fourier transform to work.
    """
    N = len(x)
    T = max(x) / float(N)

    F_I = scfft.fft(I)
    f1 = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)

    g1_plot_val = 2.0 / N * np.abs(F_I[:N // 2])

    ind = np.where(g1_plot_val > 1)[0]

    temp = []
    for i in ind:
        if i != 0:
            temp.append(g1_plot_val[i])
    temp = np.array(temp)

    # OPTIMISATION ALGORITHM REQUIRES AN INITIAL GUESS
    guess_k = np.pi / 50.
    guess_offset = np.mean(I)
    guess_amplitude = max(I)-min(I)
    guess_phase = np.pi/4
    u = [guess_amplitude, guess_k, guess_phase, guess_offset]

    maxfreq =  f1[np.where(temp == max(temp))[0][0]]
    print maxfreq

    yy = u[0]*np.cos(maxfreq * 2*np.pi*x)**2 + u[-1] - u[0]/2

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(x, I)
    plt.plot(x,yy)

    plt.subplot(1, 2, 2)
    plt.plot(f1, 2.0 / N * np.abs(F_I[:N // 2]))
    plt.axis([0, 0.05, 0, 200])

    plt.show()
    return

def hist_peak(array, o=1, bins='auto'):
    n, bins = np.histogram(array, bins=bins)
    if o == 2:
        n, bins, patches = plt.hist(array, bins=bins)
    return bins[_max_value_index(n)]

def _max_value_index(array):
    if type(array) is list:
        array = np.array(array)
    loc = np.where(array == max(array))[0][0]
    return loc

def oscillate_func(u, x, A, offset):
    return A * (np.cos(u[0] * x + u[1])) ** 2 + offset - A/2

def osc_func_noA(u, x, offset):
    return u[2] * (np.cos(u[0] * x + u[1])) ** 2 + offset - u[2]/2

def osc_func_k(u, x, A, offset, k):
    return A * (np.cos(k * x + u[0])) ** 2 + offset - A / 2

def osc_func_phase(u, x, A, offset, phase):
    return A * (np.cos(u[0] * x + phase)) ** 2 + offset - A / 2

def oscillate_func_1(u, x):
    return u[0] * (np.cos(u[1] * x + u[2])) ** 2 + u[-1] - u[0]/2

def oscillate_func_2(u, x):
    return (np.cos(u[0] * x + u[1])) ** 2

def plotdata(num, fig):
    """
    Plots the data.
    """
    num_up = int(num/2) + 1

    ax21 = plt.subplot2grid((10, 3), (0, 2), rowspan=4)
    ax21.plot(voltages[:num_up], k_values[:num_up], 'gx', label='Increasing voltage')
    ax21.plot(voltages[:num_up], k_values[:num_up], 'g-')
    ax21.plot(voltages[num_up-1:], k_values[num_up-1:], 'kx', label='Decreasing voltage')
    ax21.plot(voltages[num_up-1:], k_values[num_up-1:], 'k-')
    k_plot0, = ax21.plot([], [], 'g-', linewidth=3)
    k_plot1, = ax21.plot([], [], 'k-', linewidth=3)
    ax21.set_title('Optimised wavenumbers with voltage')
    ax21.set_ylabel('$Wavenumber$ $(not$ $m^{-1})$', fontsize=18)
    ax21.set_xlabel('$Voltage$ $(V)$', fontsize=18)
    ax21.legend(loc='best')
    
    ax22 = plt.subplot2grid((2, 10), (1, 0), colspan=6)
    # ax22.yaxis.set_major_formatter(FuncFormatter(format_fn))
    # ax22.yaxis.set_major_locator(MultipleLocator(base=np.pi))
    ax22.plot(voltages[:num_up], displacement[:num_up], 'gx', label='Increasing voltage')
    ax22.plot(voltages[:num_up], displacement[:num_up], 'g-')
    ax22.plot(voltages[num_up-1:], displacement[num_up-1:], 'kx', label='Decreasing voltage')
    ax22.plot(voltages[num_up-1:], displacement[num_up-1:], 'k-')
    phase_solved0, = ax22.plot([], [], 'g-', linewidth=3)
    phase_solved1, = ax22.plot([], [], 'k-', linewidth=3)
    #ax22.set_title("Displacement")
    ax22.set_ylabel('$Displacement$ $(\mu m)$', fontsize=18)
    ax22.set_xlabel('$Voltage$ $(V)$', fontsize=18)
    ax22.legend(loc='best')

    ax23 = plt.subplot2grid((10, 3), (6, 2), rowspan=4)
    ax23.plot(voltages[:num_up], original_phases[:num_up], 'gx', label='Increasing voltage')
    ax23.plot(voltages[:num_up], original_phases[:num_up], 'g-')
    ax23.plot(voltages[num_up-1:], original_phases[num_up-1:], 'kx', label='Decreasing voltage')
    ax23.plot(voltages[num_up - 1:], original_phases[num_up - 1:], 'k-')
    phase_orig0, = ax23.plot([], [], 'g-', linewidth=3)
    phase_orig1, = ax23.plot([], [], 'k-', linewidth=3)
    ax23.set_title('Fitted phases')
    ax23.set_ylabel('$Phase$', fontsize=18)
    ax23.set_xlabel('$Voltage$ $(V)$', fontsize=18)
    ax23.legend(loc='best')

    ax = [ax21, ax22, ax23]
    k_plot = [k_plot0, k_plot1]
    phase_solved = [phase_solved0, phase_solved1]
    phase_orig = [phase_orig0, phase_orig1]
    return ax, k_plot, phase_solved, phase_orig


# INFO TO FIND IMAGES
folder = 'clean2'
base_name = 'new_image_'
file_ext = '.tif'
no_of_images = 91
filenumbers = np.arange(no_of_images)

#'1' FOR MODE, '2' FOR MEAN
mode_or_mean = 2
#Pickle file name
picklefile = 'mean2.p'

animation_speed = 400

#max_voltage = 90
#voltage_increments = 10.
#
no_of_points_forward = (no_of_images + 1) / 2
no_of_points_backward = no_of_points_forward - 1

# voltages = list(range(int(no_of_points_forward))) + list(range(int(no_of_points_backward - 1), -1, -1))
# voltages = np.array(voltages)*voltage_increments

voltages = [0.1, 2.3, 3.9, 6, 8.3, 10, 11.9, 13.9, 16, 18.1, 20.1, \
            22, 24.1, 26, 28, 30, 32, 33.9, 36.1, 38.1, 40, 42, 44.1, \
            46.2, 48.2, 50, 52.1, 54.1, 56, 58.3, 60, 62.3, 64.2, 66.1, \
            68.1, 70.3, 72, 74.3, 76, 78.1, 80, 82.5, 84, 86.1, 88.1, 90.4, \
            88, 85.9, 84, 82.2, 80.4, 78.3, 75.7, 73.8, 72.2, 70.3, 68.2, 66, \
            63.6, 61.9, 60.1, 58, 55.8, 54.1, 52, 49.9, 47.7, 45.9, 44.1, 41.6, \
            40.1, 38.1, 35.8, 33.8, 31.9, 29.9, 27.9, 25.9, 23.9, 21.9, 19.9, 18, \
            16, 13.7, 11.9, 10, 8, 6, 4, 1.6, 0.2]

# LISTS OF PARAMETERS FROM FITTING
amp_values = []
k_values = []
phase_values = []
offset_values = []
intensity_values = []
intensity_values_fit = []

# APPLY FITTING TO ALL IMAGES
for u in filenumbers:
    u = str(u)
    print(u)

    file_path = folder + '/' + base_name + u + file_ext

    if mode_or_mean == 1:
        wave_optimised_params, intensity, x = optimise_mode(file_path, vertical_fringes=True)
    else:
        wave_optimised_params, intensity, x = optimise_mean(file_path, vertical_fringes=True)

    amp_values.append(wave_optimised_params[0])
    k_values.append(wave_optimised_params[1])
    phase_values.append(wave_optimised_params[2])
    offset_values.append(wave_optimised_params[3])
    intensity_values.append(intensity)
    intensity_values_fit.append(oscillate_func_1(wave_optimised_params, x))

original_phases = np.array(phase_values)
phase_values = np.array(phase_values)

#CONVERT PHASE SHIFT INTO DISTANCE
for u in range(1, len(filenumbers)):
    diff = phase_values[u-1] - phase_values[u]
    diff_sign = np.sign(diff)
    while abs(diff) > np.pi/2:
        phase_values[u] = phase_values[u] + diff_sign*np.pi
        diff = abs(phase_values[u] - phase_values[u - 1])

#ONE FRINGE SPACING IS HALF A WAVELENGTH AND IS 1 PI PHASE SHIFT
wavelength = 532.e-9
distance_per_fringe_spacing = wavelength/2.
no_of_fringe_spacing_moved = phase_values/np.pi
displacement = no_of_fringe_spacing_moved * distance_per_fringe_spacing

#CONVERT TO MICROMETERS
displacement = displacement*1e6

#SAVE INFORMATION INTO A FILE
#makefile(picklefile)
datafile = open(picklefile, 'wb')
pickle.dump([x, voltages, k_values, original_phases, intensity_values, intensity_values_fit, displacement], datafile)
datafile.close()

# PLOT THE INFORMATION INTO GRAPHS
fig = plt.figure(num=1, figsize=(16, 9))

ax00 = plt.subplot2grid((2, 10), (0, 0), colspan=6)
plt.subplots_adjust(left=0.15, bottom=0.25)
pattern00, = ax00.plot(x, intensity_values[0], 'g-', linewidth=3, label='measurements')
fitted_pattern00, = ax00.plot(x, intensity_values_fit[0], 'b-', linewidth=3, label='fitted')
title00 = ax00.set_title("i = %i, k = %0.5f, displacement = %0.4f, orig_phase = %0.4f" % (0, k_values[0], displacement[0], original_phases[0]))
ax00.set_ylabel("$Intensity$", fontsize=18)
ax00.set_xlim(x[0], x[-1])
ax00.set_ylim(smallest_intensity(intensity_values), largest_intensity(intensity_values))
ax00.legend(loc='best')

axcolor = 'lightgoldenrodyellow'
ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03], axisbg=axcolor)
shte = Slider(ax_slider, 'i', 0, no_of_images-1, valinit=0, valfmt='%0.0f')

ax_list, k_plot, phase_solved_plot, phase_orig_plot = plotdata(num=no_of_images, fig=fig)

def update(val):
    """
    This function is used when the slider bar is moved.
    val = value of the slider
    """
    val = int(val)
    if val > no_of_points_forward:
        k_plot[0].set_data(voltages[:no_of_points_forward], k_values[:no_of_points_forward])
        k_plot[1].set_data(voltages[no_of_points_forward-1:val+1], k_values[no_of_points_forward-1:val+1])
        phase_solved_plot[0].set_data(voltages[:no_of_points_forward], displacement[:no_of_points_forward])
        phase_solved_plot[1].set_data(voltages[no_of_points_forward-1:val+1], displacement[no_of_points_forward-1:val+1])
        phase_orig_plot[0].set_data(voltages[:no_of_points_forward], original_phases[:no_of_points_forward])
        phase_orig_plot[1].set_data(voltages[no_of_points_forward-1:val+1], original_phases[no_of_points_forward-1:val+1])
    else:
        k_plot[0].set_data(voltages[:val+1], k_values[:val+1])
        k_plot[1].set_data([],[])
        phase_solved_plot[0].set_data(voltages[:val+1], displacement[:val+1])
        phase_solved_plot[1].set_data([],[])
        phase_orig_plot[0].set_data(voltages[:val+1], original_phases[:val+1])
        phase_orig_plot[1].set_data([],[])
    pattern00.set_data(x, intensity_values[val])
    fitted_pattern00.set_data(x, intensity_values_fit[val])
    title00.set_text("i = %i, k = %0.5f, displacement = %0.4f, orig_phase = %0.4f" % (val, k_values[val], displacement[val], original_phases[val]))
    plt.draw()

shte.on_changed(update)

#anim = an.FuncAnimation(fig, animate, init_func=init, frames=no_of_images, interval=animation_speed)  # , blit=True)

plt.show()