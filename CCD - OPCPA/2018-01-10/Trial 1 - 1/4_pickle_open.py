import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.widgets import Slider

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
        return "$%i\pi$"%(tick_val/np.pi)

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

def plotdata(num, fig):
    """
    Plots the data.
    """
    num_up = int(num / 2) + 1

    ax21 = plt.subplot2grid((10, 3), (0, 2), rowspan=4)
    ax21.plot(voltages[:num_up], k_values[:num_up], 'bx', label='Increasing voltage')
    ax21.plot(voltages[:num_up], k_values[:num_up], 'b-')
    ax21.plot(voltages[num_up - 1:], k_values[num_up - 1:], 'kx', label='Decreasing voltage')
    ax21.plot(voltages[num_up - 1:], k_values[num_up - 1:], 'k-')
    k_plot0, = ax21.plot([], [], 'g-', linewidth=3)
    k_plot1, = ax21.plot([], [], 'k-', linewidth=3)
    ax21.set_title('Optimised wavenumbers with voltage')
    ax21.set_ylabel('$Wavenumber$ $(not$ $m^{-1})$', fontsize=18)
    ax21.set_xlabel('$Voltage$ $(V)$', fontsize=18)
    ax21.legend(loc='best')

    ax22 = plt.subplot2grid((2, 10), (1, 0), colspan=6)
    # ax22.yaxis.set_major_formatter(FuncFormatter(format_fn))
    # ax22.yaxis.set_major_locator(MultipleLocator(base=np.pi))
    ax22.plot(voltages[:num_up], displacement[:num_up], 'bx', label='Increasing voltage')
    ax22.plot(voltages[:num_up], displacement[:num_up], 'b-')
    ax22.plot(voltages[num_up - 1:], displacement[num_up - 1:], 'kx', label='Decreasing voltage')
    ax22.plot(voltages[num_up - 1:], displacement[num_up - 1:], 'k-')
    phase_solved0, = ax22.plot([], [], 'g-', linewidth=3)
    phase_solved1, = ax22.plot([], [], 'k-', linewidth=3)
    #ax22.set_title("Displacement")
    ax22.set_ylabel('Displacement $(\mu m)$', fontsize=18)
    ax22.set_xlabel('Voltage $(V)$', fontsize=18)
    ax22.legend(loc=4)

    ax23 = plt.subplot2grid((10, 3), (6, 2), rowspan=4)
    ax23.plot(voltages[:num_up], original_phases[:num_up], 'bx', label='Increasing voltage')
    ax23.plot(voltages[:num_up], original_phases[:num_up], 'b-')
    ax23.plot(voltages[num_up - 1:], original_phases[num_up - 1:], 'kx', label='Decreasing voltage')
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

#filename = 'data.p'
#pickle_in = open(filename, "rb")
data = pickle.load(open("mean2.p", "rb"))

x, voltages, k_values, original_phases, intensity_values, intensity_values_fit, displacement = data[:]
no_of_images = len(voltages)

no_of_points_forward = (no_of_images + 1) / 2
no_of_points_backward = no_of_points_forward - 1

############ PLOT THE INFORMATION INTO GRAPHS #################################
fig = plt.figure(num=1, figsize=(16, 9))

ax00 = plt.subplot2grid((2, 10), (0, 0), colspan=6)
plt.subplots_adjust(left=0.15, bottom=0.25)
pattern00, = ax00.plot(x, intensity_values[0], 'g-', linewidth=3, label='measurements')
fitted_pattern00, = ax00.plot(x, intensity_values_fit[0], 'b-', linewidth=3, label='fitted')
title00 = ax00.set_title("i = %i, k = %0.5f, displacement = %0.4f, phase = %0.4f" % (0, k_values[0], displacement[0], original_phases[0]))
ax00.set_xlim(x[0], x[-1])
ax00.set_ylim(smallest_intensity(intensity_values), largest_intensity(intensity_values))
ax00.legend(loc='best')

axcolor = 'lightgoldenrodyellow'
ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03], axisbg=axcolor)
shte = Slider(ax_slider, 'i', 0, no_of_images-1, valinit=0, valfmt='%0.0f')

ax_list, k_plot, phase_solved_plot, phase_orig_plot = plotdata(num=no_of_images, fig=fig)

def update(val):
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
    title00.set_text("i = %i, k = %0.5f, displacement = %0.4f, phase = %0.4f" % (val, k_values[val], displacement[val], original_phases[val]))
    plt.draw()

shte.on_changed(update)

#anim = an.FuncAnimation(fig, animate, init_func=init, frames=no_of_images, interval=animation_speed)  # , blit=True)

plt.show()
#########################################################
