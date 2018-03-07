import func_optimise
import func_clean
import func_crop

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.animation as an
from matplotlib.widgets import Slider
import pickle

def makefile(filename):
    # Create a file if it doesn't already exist
    with open(filename, 'w') as f:
        f.close()
    return

def largest_intensity(intensity_values):
    max_intensity = 0
    for intensity_array in intensity_values:
        for array in intensity_array:
            if max(array) > max_intensity:
                max_intensity = max(array)
    return max_intensity


def smallest_intensity(intensity_values):
    min_intensity = largest_intensity(intensity_values)
    for intensity_array in intensity_values:
        for array in intensity_array:
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

def ploterrorbar(x, y, yerr):
    plt
    
def plotdata(num, fig, voltages, k_values):
    """
    Plots the data.
    """
    num_up = int(num/2) + 1
    
    ax21 = fig.add_subplot(2, 3, 4)
    ax21.errorbar(voltages[:num_up], k_values[0][:num_up], yerr=k_values[0][:num_up], fmt='gx-',label='Increasing voltage')
    ax21.errorbar(voltages[num_up-1:], k_values[0][num_up-1:], yerr=k_values[0][num_up-1:], fmt='kx-', label='Decreasing voltage')
    k_plot0, = ax21.plot([], [], 'g-', linewidth=3)
    k_plot1, = ax21.plot([], [], 'k-', linewidth=3)
    ax21.set_title('Optimised wavenumbers with voltage')
    ax21.set_ylabel('$Wavenumber$ $(not$ $m^{-1})$', fontsize=18)
    ax21.set_xlabel('$Voltage$ $(V)$', fontsize=18)
    ax21.legend()
    
    ax22 = fig.add_subplot(2,3,5)
    ax22.yaxis.set_major_formatter(FuncFormatter(format_fn))
    ax22.yaxis.set_major_locator(MultipleLocator(base=np.pi))
    ax22.errobar(voltages[:num_up], phase_values[0][:num_up], yerr=phase_values[0][:num_up], fmt='gx-', label='Increasing voltage')
    ax22.plot(voltages[num_up-1:], phase_values[0][num_up-1:], yerr=phase_values[0][num_up-1:], fmt='kx-', label='Decreasing voltage')
    phase_solved0, = ax22.plot([], [], 'g-', linewidth=3)
    phase_solved1, = ax22.plot([], [], 'k-', linewidth=3)
    ax22.set_title("Phases (solved)")
    ax22.set_ylabel('$Phase$', fontsize=18)
    ax22.set_xlabel('$Voltage$ $(V)$', fontsize=18)
    ax22.legend()

    ax23 = fig.add_subplot(2, 3, 6)
    ax23.plot(voltages[:num_up], original_phases[0][:num_up], yerr=original_phases[0][:num_up], fmt='gx-', label='Increasing voltage')
    ax23.plot(voltages[num_up-1:], original_phases[0][num_up-1:], yerr=original_phases[0][num_up-1:], fmt='kx-', label='Decreasing voltage')
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

#DETAILS OF THE IMAGES
folder = 'original'
base_name = 'image_'
file_ext = '.tif'
no_of_images = 19
filenumbers = range(no_of_images)

no_of_repeats = 10
vertical_fringes = False

picklefile = 'data.p'

no_of_points_forward = (no_of_images + 1) // 2
no_of_points_backward = no_of_points_forward - 1

voltages = [0.3, 10.6, 20.4, 29.8, 39.3, 49.3, 59.3, 70.3, 83.3,\
            89.7, 79.9, 66, 60.1, 49.7, 40.6, 30.8, 19.7, 8.4, 0.5]

if len(voltages) != no_of_images:
    raise ValueError('Incorrect number of images or incorrect values of voltages')

if vertical_fringes == False:
    max_space = 1024
    x = np.arange(1280)
    xlen = 1280
else:
    max_space = 1280
    x = np.arange(1024)
    xlen = 1024

crop_intervals = max_space/(no_of_repeats+1)
crop_spacing = np.arange(0, max_space, crop_intervals)

# LISTS OF PARAMETERS FROM FITTING
amp_values = np.zeros((no_of_images, no_of_repeats))
k_values = np.zeros((no_of_images, no_of_repeats))
phase_values = np.zeros((no_of_images, no_of_repeats))
offset_values = np.zeros((no_of_images, no_of_repeats))

intensity_values = np.zeros((no_of_images, no_of_repeats, xlen))
intensity_values_fit = np.zeros((no_of_images, no_of_repeats, xlen))

#LOOP ON EVERY IMAGE
for file_number in filenumbers:

    file_number_str = str(file_number)
    print(file_number_str)
    
    #OPEN .tif FILE
    filename = folder + '/' + base_name + file_number_str + file_ext

    for repeat_number in range(no_of_repeats-1):
        new_image = func_crop.crop_algorithm(filename, vertical_fringes=vertical_fringes, min_crop=crop_spacing[repeat_number], max_crop=crop_spacing[repeat_number+1])
    
        new_image = np.asarray(new_image.convert('L'))
        new_image = func_clean.clean_algorithm(new_image)

        optimised_data = func_optimise.optimise_algorithm(new_image)

        amp_values[file_number, repeat_number] = optimised_data[0]
        k_values[file_number, repeat_number] = optimised_data[1]
        phase_values[file_number, repeat_number] = optimised_data[2]
        offset_values[file_number, repeat_number] = optimised_data[3]

        intensity_values[file_number, repeat_number] = optimised_data[6]
        intensity_values_fit[file_number, repeat_number] = optimised_data[5]

        for u in range(1, len(filenumbers)):
            diff = phase_values[u-1] - phase_values[u]
            diff_sign = np.sign(diff)
            while abs(diff) > np.pi/2:
                phase_values[u] = phase_values[u] + diff_sign*np.pi
                diff = abs(phase_values[u] - phase_values[u - 1])

datafile1 = open('mid_data.p', 'wb')
pickle.dump([voltages, k_values, phase_values, intensity_values, intensity_values_fit], datafile)
datafile1.close()

def mean_std(array):
    if array.ndim == 3:
        array = array.transpose(0, 2, 1)
        for array_index in filenumbers:
            array[i] = [list(map(np.mean, array[i]))] + [list(map(np.std, array[i]))]
        return array
    array = [list(map(np.mean, array))] + [list(map(np.std, array))]
    return array

amp_values = mean_std(amp_values)
k_values = mean_std(k_values)
phase_values = mean_std(phase_values)
offset_values = mean_std(offset_values)
intensity_values = mean_std(intensity_values)
intensity_values_fit = mean_std(intensity_values_fit)

original_phases = phase_values[:]
        
#SAVE INFORMATION INTO A FILE
#makefile(picklefile)
datafile = open(picklefile, 'wb')
pickle.dump([voltages, k_values, phase_values, intensity_values, intensity_values_fit], datafile)
datafile.close()

# PLOT THE INFORMATION INTO GRAPHS
fig = plt.figure(num=1, figsize=(16, 9))

ax00 = fig.add_subplot(2, 1, 1)
plt.subplots_adjust(left=0.15, bottom=0.25)
pattern00, = ax00.errorbar(x, intensity_values[0][0], yerr=intensity_values[0][1], fmt='g-', linewidth=3, label='measurements')
fitted_pattern00, = ax00.plot(x, intensity_values_fit[0][0], yerr=intensity_values_fit[0][1], fmt='b-', linewidth=3, label='fitted')
title00 = ax00.set_title(' ')
ax00.set_xlim(x[0], x[-1])
ax00.set_ylim(smallest_intensity(intensity_values), largest_intensity(intensity_values))
ax00.legend()

axcolor = 'lightgoldenrodyellow'
ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
shte = Slider(ax_slider, 'i', 0, no_of_images-1, valinit=0, valfmt='%i')

ax_list, k_plot, phase_solved_plot, phase_orig_plot = plotdata(num=no_of_images, fig=fig, voltages=voltages, k_values=k_values)

def plot_control(plot, val, past_half = True):
    if past_half == True:
        plot[0].set_data(voltages[:no_of_points_forward], plot[0][:no_of_points_forward])
        plot[1].set_data(voltages[no_of_points_forward - 1:val + 1], plot[0][no_of_points_forward - 1:val + 1])
    else:
        plot[0].set_data(voltages[:val+1], plot[0][:val+1])
        plot[1].set_data([],[])

def update(val):
    val = int(val)
    if val > no_of_points_forward:
        plot_control(k_values, val, past_half=True)
        plot_control(phase_solved_plot, val, past_half=True)
        plot_control(phase_orig_plot, val, past_half=True)
    else:
        plot_control(k_values, val, past_half=False)
        plot_control(phase_solved_plot, val, past_half=False)
        plot_control(phase_orig_plot, val, past_half=False)
    pattern00.set_data(x, intensity_values[val])
    fitted_pattern00.set_data(x, intensity_values_fit[val])
    title00.set_text("i = %i, k = %0.5f, phase = %0.4f, orig_phase = %0.4f" % (val, k_values[val], phase_values[val], original_phases[val]))
    plt.draw()

shte.on_changed(update)

plt.show()
    
    
    




















