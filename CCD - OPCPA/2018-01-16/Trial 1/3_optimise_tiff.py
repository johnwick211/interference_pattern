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
    fitted_pattern.set_data(x, intensity_values_avg[j % no_of_images])
    title11.set_text("Measured/Average Intensity Pattern. i = %i" % j)
    title12.set_text("Fitted Intensity Pattern. i = %i" % j)
    return pattern, fitted_pattern, title11, title12


def makefile(filename):
    # Create a file if it doesn't already exist
    with open(filename, 'w') as f:
        f.close()
    return


def largest_intensity(intensity_values):
    max_intensity = 0
    for array in intensity_values:
        if max(array) > max_intensity:
            max_intensity = max(array)
    return max_intensity


def smallest_intensity(intensity_values):
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
    #PROPETIES OF THE WAVE
    offset = np.mean(I)
    amplitude = max(I)-min(I)

    # OPTIMISATION ALGORITHM REQUIRES AN INITIAL GUESS
    guess_k = np.pi/200
    guess_phase = np.pi/4

    counter = 0
    k_val = guess_k
    phase_val = guess_phase
    while counter < 100:
        counter += 1

        # FUNCTIONS TO FIT
        optimize_func_k = lambda u: osc_func_k(u, x, amplitude, offset, k_val) - I
        u = optimize.leastsq(optimize_func_k, [phase_val])[0]
        phase_val = u[0]

        optimize_func_phase = lambda u: osc_func_phase(u, x, amplitude, offset, phase_val) - I
        u = optimize.leastsq(optimize_func_phase, [k_val])[0]
        k_val = u[0]

    u = [amplitude, k_val, phase_val, offset]
    # RETURNS THE OPTIMISED PARAMETERS IN THE FITTED FUNCTION
    return u, I, x

def fit(I, x):
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
    guess_k = np.pi / 200.
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

def osc_func_k(u, x, A, offset, k):
    return A * (np.cos(k * x + u[0])) ** 2 + offset - A / 2

def osc_func_phase(u, x, A, offset, phase):
    return A * (np.cos(u[0] * x + phase)) ** 2 + offset - A / 2

def oscillate_func_1(u, x):
    return u[0] * (np.cos(u[1] * x + u[2])) ** 2 + u[-1] - u[0]/2

def oscillate_func_2(u, x):
    return (np.cos(u[0] * x + u[1])) ** 2

def plotdata(num, fig, row):
    """
    Plots the data.
    """
    num_up = int(num/2) + 1
    
    ax21 = fig.add_subplot(2, 3, 4)
    ax21.plot(voltages[:num_up], k_values[:num_up], 'gx', label='Increasing voltage')
    ax21.plot(voltages[:num_up], k_values[:num_up], 'g-')
    ax21.plot(voltages[num_up-1:], k_values[num_up-1:], 'kx', label='Decreasing voltage')
    ax21.plot(voltages[num_up-1:], k_values[num_up-1:], 'k-')
    k_plot0, = ax21.plot([], [], 'g-', linewidth=3)
    k_plot1, = ax21.plot([], [], 'k-', linewidth=3)
    ax21.set_title('Optimised wavenumbers with voltage')
    ax21.set_ylabel('$Wavenumber$ $(not$ $m^{-1})$', fontsize=18)
    ax21.set_xlabel('$Voltage$ $(V)$', fontsize=18)
    ax21.legend()
    
    ax22 = fig.add_subplot(2,3,5)
    ax22.yaxis.set_major_formatter(FuncFormatter(format_fn))
    ax22.yaxis.set_major_locator(MultipleLocator(base=np.pi))
    ax22.plot(voltages[:num_up], phase_values[:num_up], 'gx', label='Increasing voltage')
    ax22.plot(voltages[:num_up], phase_values[:num_up], 'g-')
    ax22.plot(voltages[num_up-1:], phase_values[num_up-1:], 'kx', label='Decreasing voltage')
    ax22.plot(voltages[num_up-1:], phase_values[num_up-1:], 'k-')
    phase_solved0, = ax22.plot([], [], 'g-', linewidth=3)
    phase_solved1, = ax22.plot([], [], 'k-', linewidth=3)
    ax22.set_title("Phases (solved)")
    ax22.set_ylabel('$Phase$', fontsize=18)
    ax22.set_xlabel('$Voltage$ $(V)$', fontsize=18)
    ax22.legend()

    ax23 = fig.add_subplot(2, 3, 6)
    ax23.plot(voltages[:num_up], original_phases[:num_up], 'gx', label='Increasing voltage')
    ax23.plot(voltages[:num_up], original_phases[:num_up], 'g-')
    ax23.plot(voltages[num_up-1:], original_phases[num_up-1:], 'kx', label='Decreasing voltage')
    ax23.plot(voltages[num_up - 1:], original_phases[num_up - 1:], 'k-')
    phase_orig0, = ax23.plot([], [], 'g-', linewidth=3)
    phase_orig1, = ax23.plot([], [], 'k-', linewidth=3)
    ax23.set_title('Original phases')
    ax23.set_ylabel('$Phase$', fontsize=18)
    ax23.set_xlabel('$Voltage$ $(V)$', fontsize=18)
    ax23.legend()

    ax = [ax21, ax22, ax23]
    k_plot = [k_plot0, k_plot1]
    phase_solved = [phase_solved0, phase_solved1]
    phase_orig = [phase_orig0, phase_orig1]
    return ax, k_plot, phase_solved, phase_orig


# INFO TO FIND IMAGES
folder = 'clean2'
base_name = 'new_image_'
file_ext = '.tif'
no_of_images = 19
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

voltages = [0.3, 10.6, 20.4, 29.8, 39.3, 49.3, 59.3, 70.3, 83.3,\
            89.7, 79.9, 66, 60.1, 49.7, 40.6, 30.8, 19.7, 8.4, 0.5]

# LISTS OF PARAMETERS FROM FITTING
amp_values = []
k_values = []
phase_values = []
offset_values = []
intensity_values = []
intensity_values_avg = []

# APPLY FITTING TO ALL IMAGES
for u in filenumbers:
    u = str(u)
    print(u)

    file_path = folder + '/' + base_name + u + file_ext

    if mode_or_mean == 1:
        wave_optimised_params, intensity, x = optimise_mode(file_path, vertical_fringes=False)
    else:
        wave_optimised_params, intensity, x = optimise_mean(file_path, vertical_fringes=False)

    amp_values.append(wave_optimised_params[0])
    k_values.append(wave_optimised_params[1])
    phase_values.append(wave_optimised_params[2])
    offset_values.append(wave_optimised_params[3])
    intensity_values.append(intensity)
    intensity_values_avg.append(oscillate_func_1(wave_optimised_params, x))

original_phases = np.array(phase_values)

for u in range(1, len(filenumbers)):
    diff = phase_values[u-1] - phase_values[u]
    diff_sign = np.sign(diff)
    while abs(diff) > np.pi/2:
        phase_values[u] = phase_values[u] + diff_sign*np.pi
        diff = abs(phase_values[u] - phase_values[u - 1])

# for i in range(len(phase_values)):
#     if phase_values[i] < -np.pi/4:
#         phase_values[i] += np.pi/4
#     elif phase_values[i] > np.pi/4:
#         phase_values[i] -= np.pi/4
#

# # ALGORITHM TO MAKE THE PHASE SHIFT FOLLOW A LINEAR PATH INSTEAD OF OSCILLATING
# differences = []
# for u in range(1, len(filenumbers)):
#     differences.append(abs(phase_values[u] - phase_values[u - 1]))
#
# phase_sign = np.sign(phase_values[0])
# for u in range(1, len(filenumbers)):
#     if u < no_of_points_forward:
#         phase_values[u] = phase_values[u - 1] + phase_sign * differences[u - 1]
#     else:
#         phase_values[u] = phase_values[u - 1] - phase_sign * differences[u - 1]

# #MAKE A LINEAR FITTING OF THE PHASE SHIFT
# guess_m = -1
# guess_c = 0
# linear_func = lambda u: u[0]*voltages + u[1]
# optimise_func = lambda u: linear_func(u) - phase_values
# linear_optimised_params = optimize.leastsq(optimise_func, [guess_m, guess_c])[0]

# #LINEAR FIT PHASE VALUES
# phase_fit = linear_func(linear_optimised_params)

#SAVE INFORMATION INTO A FILE
#makefile(picklefile)
datafile = open(picklefile, 'wb')
pickle.dump([voltages, k_values, phase_values, intensity_values, intensity_values_avg], datafile)
datafile.close()

# PLOT THE INFORMATION INTO GRAPHS
fig = plt.figure(num=1, figsize=(16, 9))

ax00 = fig.add_subplot(2, 1, 1)
plt.subplots_adjust(left=0.15, bottom=0.25)
pattern00, = ax00.plot(x, intensity_values[0], 'g-', linewidth=3, label='measurements')
fitted_pattern00, = ax00.plot(x, intensity_values_avg[0], 'b-', linewidth=3, label='fitted')
title00 = ax00.set_title(' ')
ax00.set_xlim(x[0], x[-1])
ax00.set_ylim(smallest_intensity(intensity_values), largest_intensity(intensity_values))
ax00.legend()

axcolor = 'lightgoldenrodyellow'
ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03], axisbg=axcolor)
shte = Slider(ax_slider, 'i', 0, no_of_images-1, valinit=0, valfmt='%0.0f')

ax_list, k_plot, phase_solved_plot, phase_orig_plot = plotdata(num=no_of_images, fig=fig, row=2)

def update(val):
    val = int(val)
    if val > no_of_points_forward:
        k_plot[0].set_data(voltages[:no_of_points_forward], k_values[:no_of_points_forward])
        k_plot[1].set_data(voltages[no_of_points_forward-1:val+1], k_values[no_of_points_forward-1:val+1])
        phase_solved_plot[0].set_data(voltages[:no_of_points_forward], phase_values[:no_of_points_forward])
        phase_solved_plot[1].set_data(voltages[no_of_points_forward-1:val+1], phase_values[no_of_points_forward-1:val+1])
        phase_orig_plot[0].set_data(voltages[:no_of_points_forward], original_phases[:no_of_points_forward])
        phase_orig_plot[1].set_data(voltages[no_of_points_forward-1:val+1], original_phases[no_of_points_forward-1:val+1])
    else:
        k_plot[0].set_data(voltages[:val+1], k_values[:val+1])
        k_plot[1].set_data([],[])
        phase_solved_plot[0].set_data(voltages[:val+1], phase_values[:val+1])
        phase_solved_plot[1].set_data([],[])
        phase_orig_plot[0].set_data(voltages[:val+1], original_phases[:val+1])
        phase_orig_plot[1].set_data([],[])
    pattern00.set_data(x, intensity_values[val])
    fitted_pattern00.set_data(x, intensity_values_avg[val])
    title00.set_text("i = %i, k = %0.5f, phase = %0.4f, orig_phase = %0.4f" % (val, k_values[val], phase_values[val], original_phases[val]))
    plt.draw()

shte.on_changed(update)

#anim = an.FuncAnimation(fig, animate, init_func=init, frames=no_of_images, interval=animation_speed)  # , blit=True)

plt.show()