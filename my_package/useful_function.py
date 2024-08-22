"""
! All the pwd used in the following are in the context of my own analysis performed with the CCIN2P3 and cannot be accessed for unauthorized people.

Utility Functions and Information for JUNO Data Analysis

This script provides a set of utility functions for processing and analyzing JUNO data. 
It includes functions for creating ADC counts, removing pedestals from waveforms, plotting waveform data, 
creating animations, and performing various types of mathematical fits.

Created by: Lukas Péron
Last Update: 22/08/2024

Overview:
---------
The script includes the following functions:
1. `create_adc_count(WaveformQ)`: Converts waveform charge data into ADC counts.
2. `piedestal_remove(adc_count)`: Removes the pedestal (noise) from ADC counts.
3. `plot_colormesh_waveform(data, num_file, event_num)`: Creates a color mesh plot of waveform data.
4. `make_animation(num_file, event_num)`: Creates and saves a 3D event display animation for JUNO data.
5. Various fit functions (`linear_fit`, `quadratic_fit`, etc.) for data fitting.
6. Calibration functions to convert ADC counts to energy.

Error Handling:
---------------
- The `create_adc_count`, `piedestal_remove`, and `plot_colormesh_waveform` functions include basic checks to ensure valid input data.
- The `make_animation` and `update` functions check the validity of input data and handle potential issues.
- The plotting and fitting functions assume that the input data is valid; additional error handling may be added as needed.

Dependencies:
-------------
- ROOT
- numpy
- matplotlib

Instructions:
-------------
1. Ensure that ROOT, numpy, and matplotlib are installed and configured in your environment.
2. Use the provided functions to process and analyze JUNO data.
3. The output plots and animations will be saved in the specified directories.
"""

from ROOT import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors

# Detector PMT geometry
vertices = np.genfromtxt("/pbs/home/l/lperon/work_JUNO/JUNO_info/PMTPos_CD_LPMT.csv", usecols=(0, 1, 2, 3))

def create_adc_count(WaveformQ):
    """
    Converts waveform charge data into ADC counts.

    Parameters:
    -----------
    WaveformQ : list of lists or array-like
        The waveform charge data for each PMT.

    Returns:
    --------
    numpy.ndarray
        An array of ADC counts with pedestals removed.

    Raises:
    -------
    ValueError
        If the input WaveformQ is not in the expected format or is empty.
    """
    if not WaveformQ or not isinstance(WaveformQ, (list, np.ndarray)):
        raise ValueError("Input WaveformQ must be a non-empty list or array-like structure.")

    adc_count = np.zeros((len(WaveformQ), 1000))
    waveform_array = np.zeros((len(WaveformQ), 1000))  # Convert WaveformQ to a numpy array with the appropriate shape

    for i, waveform in enumerate(WaveformQ):
        waveform_array[i, :len(waveform)] = waveform

    adc_count[:, :1000] = waveform_array  # Assign waveform_array to adc_count
    adc_count = pedestal_remove(adc_count)
    return adc_count

def pedestal_remove(adc_count):
    """
    Removes the pedestal (noise) from ADC counts by analyzing the waveform data.

    Parameters:
    -----------
    adc_count : numpy.ndarray
        Array of ADC counts for each PMT.

    Returns:
    --------
    numpy.ndarray
        The ADC counts with pedestals removed.

    Raises:
    -------
    ValueError
        If adc_count is empty or not in the expected format.
    """
    if adc_count.size == 0 or not isinstance(adc_count, np.ndarray):
        raise ValueError("Input adc_count must be a non-empty numpy array.")

    for pmt in range(len(adc_count)):
        max_loc = np.where(adc_count[pmt] == np.max(adc_count[pmt]))[0][0]
        max_window_min = max_loc - 20
        max_window_max = max_loc + 51
        noise = np.concatenate((adc_count[pmt][:max_window_min], adc_count[pmt][max_window_max:]))
        mean = np.mean(noise)
        std = np.std(noise)
        adc_count[pmt][adc_count[pmt] <= mean + 3 * std] = 0
        adc_count[pmt][adc_count[pmt] > mean + 3 * std] -= mean
    return adc_count

def update(frame, ax, vertices, lst_pmt_on, data, event_num, num_file):
    """
    Update function for `matplotlib.animation.FuncAnimation` to generate frames of the 3D event display animation.

    Parameters:
    -----------
    frame : int
        The current frame number of the animation.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D subplot where the PMT positions and data will be plotted.
    vertices : numpy.ndarray
        Array containing the PMT positions.
    lst_pmt_on : list of int
        List of indices of PMTs that are active (on) during the event.
    data : numpy.ndarray
        Array containing the data for each PMT at each time frame.
    event_num : int
        The event number being displayed.
    num_file : int
        The file number associated with the data.

    Raises:
    -------
    ValueError
        If `vertices`, `lst_pmt_on`, or `data` is not in the expected format.
    """
    if not isinstance(vertices, np.ndarray) or vertices.shape[1] != 4:
        raise ValueError("Vertices must be a numpy array with shape (N, 4).")

    if not isinstance(lst_pmt_on, list) or not all(isinstance(i, int) for i in lst_pmt_on):
        raise ValueError("lst_pmt_on must be a list of integers.")

    if not isinstance(data, np.ndarray) or data.shape[0] != len(lst_pmt_on):
        raise ValueError("Data must be a numpy array with a shape compatible with lst_pmt_on.")

    ax.clear()
    for j, id_on in enumerate(lst_pmt_on):
        pos = vertices[id_on, :]
        q = data[j][frame]
        alpha = 1 / (1 + np.exp(-0.2 * (q - 4110)))  # Sigmoid function for transparency
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.set_zlabel("z", fontsize=14)
        ax.set_title(f"Event display for event {event_num} of file {num_file}\n t = {frame+1} ns", fontsize=14)
        ax.plot(pos[0], pos[1], pos[2], 'o', markersize=8, alpha=alpha)

def make_animation(num_file, event_num, vertices, lst_pmt_on, data):
    """
    TODO: Change the way we do the 3D plot for a better readability and pedestal removing taken into account

    Creates and saves a 3D event display animation for JUNO data.

    Parameters:
    -----------
    num_file : int
        The file number associated with the data.
    event_num : int
        The event number within the file.
    vertices : numpy.ndarray
        Array containing the PMT positions.
    lst_pmt_on : list of int
        List of indices of PMTs that are active (on) during the event.
    data : numpy.ndarray
        Array containing the data for each PMT at each time frame.

    Returns:
    --------
    None
        The function saves the animation as an MP4 file.

    Raises:
    -------
    ValueError
        If `vertices`, `lst_pmt_on`, or `data` is not in the expected format.
    """
    if not isinstance(vertices, np.ndarray) or vertices.shape[1] != 4:
        raise ValueError("Vertices must be a numpy array with shape (N, 4).")

    if not isinstance(lst_pmt_on, list) or not all(isinstance(i, int) for i in lst_pmt_on):
        raise ValueError("lst_pmt_on must be a list of integers.")

    if not isinstance(data, np.ndarray) or data.shape[0] != len(lst_pmt_on):
        raise ValueError("Data must be a numpy array with a shape compatible with lst_pmt_on.")

    fig = plt.figure(figsize=(20, 17))
    ax = fig.add_subplot(111, projection='3d')
    ani = FuncAnimation(fig, update, frames=1000, interval=10, fargs=(ax, vertices, lst_pmt_on, data, event_num, num_file))
    ani.save(f'/pbs/home/l/lperon/work_JUNO/figures/mp4/TEST_event_display_file{num_file}_event{event_num}_pileup.mp4', writer='imagemagick')

def plot_colormesh_waveform(data, num_file, event_num):
    """
    Creates a color mesh plot of the waveform data.

    Parameters:
    -----------
    data : numpy.ndarray
        The waveform data to be plotted.
    num_file : int
        The file number associated with the data.
    event_num : int
        The event number within the file.

    Returns:
    --------
    int
        Returns 0 upon successful completion.

    Raises:
    -------
    ValueError
        If the data input is empty or not a numpy array.
    """
    if data.size == 0 or not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a non-empty numpy array.")

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(data, cmap='pink', rasterized=True)
    cbar = plt.colorbar(pcm)
    cbar.set_label("ADC counts", fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel("Time elapsed since trigger [ns]", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title(f'Large PMTs Q waveform for sim {num_file} entry {event_num}', fontsize=14)
    plt.savefig(f"./../figures/svg/PMT_WF_Q_pileup_file{num_file}_event{event_num}.svg")
    plt.savefig(f"./../figures/png/PMT_WF_Q_pileup_file{num_file}_event{event_num}.png")
    return 0

# Mathematical fitting functions
def linear_fit(x, a):
    """
    Linear fit function.

    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable.
    a : float
        Slope of the linear function.

    Returns:
    --------
    numpy.ndarray
        Linear fit result.
    """
    return a * x

def quadratic_fit(x, a, x0, c):
    """
    Quadratic fit function.

    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable.
    a : float
        Coefficient for the quadratic term.
    x0 : float
        Offset of the quadratic term.
    c : float
        Constant term.

    Returns:
    --------
    numpy.ndarray
        Quadratic fit result.
    """
    return a * (x - x0) ** 2 + c

def affine_fit(x, a, b):
    """
    Affine fit function (linear function with offset).

    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable.
    a : float
        Slope of the line.
    b : float
        Intercept of the line.

    Returns:
    --------
    numpy.ndarray
        Affine fit result.
    """
    return a * x + b

def sqrt_fit(x, a, b):
    """
    Square root fit function.

    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable.
    a : float
        Coefficient for the square root function.
    b : float
        Constant term.

    Returns:
    --------
    numpy.ndarray
        Square root fit result.
    """
    return a * np.sqrt(x) + b

def power_law_fit(x, a, b):
    """
    Power law fit function.

    Parameters:
    -----------
    x : numpy.ndarray
        Independent variable.
    a : float
        Coefficient for the power law.
    b : float
        Exponent of the power law.

    Returns:
    --------
    numpy.ndarray
        Power law fit result.
    """
    return a * x ** b

# Calibration functions
def calib_ADC_to_Energy_peak(adc):
    """
    Calibration function to convert ADC counts to energy (MeV) based on the peak method.
    ! The constants used for convertion come from previous calibration. If needed, run your own calibration.

    Parameters:
    -----------
    adc : float or numpy.ndarray
        ADC counts.

    Returns:
    --------
    float or numpy.ndarray
        Calibrated energy in MeV.
    """
    return (adc / 559) ** (4 / 3) / 1852

def calib_ADC_to_Energy_integrated(adc):
    """
    Calibration function to convert ADC counts to energy (MeV) based on the integrated method.
    ! The constants used for convertion come from previous calibration. If needed, run your own calibration.

    Parameters:
    -----------
    adc : float or numpy.ndarray
        ADC counts.

    Returns:
    --------
    float or numpy.ndarray
        Calibrated energy in MeV.
    """
    return adc / (1852 * 808)