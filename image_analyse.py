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
    
def plotdata(num, fig, voltages, k_values):
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

#DETAILS OF THE IMAGES
folder = 'original'
base_name = 'image_'
file_ext = '.tif'
no_of_images = 19
filenumbers = range(no_of_images)

picklefile = 'data.p'

no_of_points_forward = (no_of_images + 1) // 2
no_of_points_backward = no_of_points_forward - 1

voltages = [0.3, 10.6, 20.4, 29.8, 39.3, 49.3, 59.3, 70.3, 83.3,\
            89.7, 79.9, 66, 60.1, 49.7, 40.6, 30.8, 19.7, 8.4, 0.5]

if len(voltages) != no_of_images:
    raise ValueError('Incorrect number of images or incorrect values of voltages')
    
# LISTS OF PARAMETERS FROM FITTING
amp_values = np.zeros(no_of_images)
k_values = np.zeros(no_of_images)
phase_values = np.zeros(no_of_images)
offset_values = np.zeros(no_of_images)

intensity_values = []
intensity_values_fit = []
x = []

#LOOP ON EVERY IMAGE
for file_number in filenumbers:

    file_number_str = str(file_number)
    print(file_number_str)
    
    #OPEN .tif FILE
    filename = folder + '/' + base_name + file_number_str + file_ext
    new_image = func_crop.crop_algorithm(filename)
    
    new_image = np.asarray(new_image.convert('L'))
    new_image = func_clean.clean_algorithm(new_image)
    
    optimised_data = func_optimise.optimise_algorithm(new_image)

    amp_values[file_number] = optimised_data[0]
    k_values[file_number] = optimised_data[1]
    phase_values[file_number] = optimised_data[2]
    offset_values[file_number] = optimised_data[3]
    
    intensity_values.append(optimised_data[4])
    intensity_values_fit.append(optimised_data[5])

    x = optimised_data[6]
    
original_phases = phase_values

for u in range(1, len(filenumbers)):
    diff = phase_values[u-1] - phase_values[u]
    diff_sign = np.sign(diff)
    while abs(diff) > np.pi/2:
        phase_values[u] = phase_values[u] + diff_sign*np.pi
        diff = abs(phase_values[u] - phase_values[u - 1])
        
#SAVE INFORMATION INTO A FILE
#makefile(picklefile)
datafile = open(picklefile, 'wb')
pickle.dump([voltages, k_values, phase_values, intensity_values, intensity_values_fit], datafile)
datafile.close()

# PLOT THE INFORMATION INTO GRAPHS
fig = plt.figure(num=1, figsize=(16, 9))

ax00 = fig.add_subplot(2, 1, 1)
plt.subplots_adjust(left=0.15, bottom=0.25)
pattern00, = ax00.plot(x, intensity_values[0], 'g-', linewidth=3, label='measurements')
fitted_pattern00, = ax00.plot(x, intensity_values_fit[0], 'b-', linewidth=3, label='fitted')
title00 = ax00.set_title(' ')
ax00.set_xlim(x[0], x[-1])
ax00.set_ylim(smallest_intensity(intensity_values), largest_intensity(intensity_values))
ax00.legend()

axcolor = 'lightgoldenrodyellow'
ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
shte = Slider(ax_slider, 'i', 0, no_of_images-1, valinit=0, valfmt='%i')

ax_list, k_plot, phase_solved_plot, phase_orig_plot = plotdata(num=no_of_images, fig=fig, voltages=voltages, k_values=k_values)

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
    fitted_pattern00.set_data(x, intensity_values_fit[val])
    title00.set_text("i = %i, k = %0.5f, phase = %0.4f, orig_phase = %0.4f" % (val, k_values[val], phase_values[val], original_phases[val]))
    plt.draw()

shte.on_changed(update)

plt.show()
    
    
    




















