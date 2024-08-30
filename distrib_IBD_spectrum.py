"""
Create Charge/Energy Spectrum for Each Occurrence of an IBD Event

This script processes ELECSIM data to create charge/energy spectra for each occurrence of an Inverse Beta Decay (IBD) event. It loads the relevant event data, identifies IBD events, calculates the energy deposited in the detector for each event, and generates histograms to visualize the energy spectra for the triggered PMTs.

Created by: Lukas Peron
Last Update: 01/09/2024

Overview:
---------
The script performs the following steps:
1. Load event classification data to identify IBD events.
2. Load ELECSIM data files and extract relevant branches and attributes.
3. Identify IBD events without pileup and calculate the energy deposition for each triggered PMT.
4. Aggregate the energy data and create histograms for the first four hits to visualize the energy spectra.

Error Handling:
---------------
- This script assumes that the input files exist and are formatted correctly. If files are missing or corrupted, the numpy file loading functions (`np.loadtxt`) will raise an `IOError`.
- The calculations assume valid data with the correct shapes; otherwise, numpy functions may raise `ValueError` or `IndexError` if the shapes do not align for matrix operations.

Dependencies:
-------------
- numpy
- matplotlib
- ROOT
- my_package (contains `elec_branches` and `useful_function`)

Instructions:
-------------
1. Ensure that numpy, matplotlib, ROOT, and my_package are installed and configured in your environment.
2. Run the script to process the ELECSIM data, generate the energy spectra, and display the histograms.
"""

import my_package.elec_branches as elec
from my_package.useful_function import create_adc_count, calib_ADC_to_Energy_integrated
import numpy as np
import matplotlib.pyplot as plt

# Load event classification data
event_classification = np.loadtxt('/pbs/home/l/lperon/work_JUNO/code/txt/detsim/event_classification.txt', skiprows=1).transpose()

# Set the file number (for demonstration purposes; in practice, this could be input by the user)
num_file_elec = 0

# Initialize lists to store data
all_info_elec = []
energy_tot = []
true_evt = []
k = 0

# Load ELECSIM data and extract relevant branches
file = elec.load_file(num_file_elec)
tree = file.Get("evt")
NGenEvts, TrueEvtID, WaveformQ = elec.load_branches(tree, "NGenEvts", "TrueEvtID", "WaveformQ")

# Filter the events to process only the concerned files
ls_file_evnt = np.transpose(event_classification)[np.where(event_classification[0] == num_file_elec)[0]]

# Process each entry in the tree
for entry in tree:
    energy = []
    NGenEvts, TrueEvtID, WaveformQ = elec.load_attributes(entry, "NGenEvts", "TrueEvtID", "WaveformQ")
    
    # Check for events without pileup and that are IBD events
    if NGenEvts == 1:  # No pileup situation
        if ls_file_evnt[TrueEvtID[0]][2] == 1:  # Select only IBD events
            true_evt.append(TrueEvtID[0])
            adc_count = create_adc_count(WaveformQ)
            
            # Calculate energy for each PMT and accumulate the total energy
            for pmt in range(len(adc_count)):
                charge = np.sum(adc_count[pmt])  # If working with ADC peak, use np.max(adc_count[pmt])
                pmt_energy = calib_ADC_to_Energy_integrated(charge)
                energy.append(pmt_energy)
            
            energy_tot.append(energy)
            print(k)
            k += 1

true_evt = np.array(true_evt)
energy_tot = np.array(energy_tot)

# Prepare lists to store the hit energy from events that generated up to four deposits
first_dep = []
second_dep = []
third_dep = []
fourth_dep = []
all_dep = [first_dep, second_dep, third_dep, fourth_dep]

# Sum elecsim events that come from the same TrueEvtID
n_repeted_events = []
for evt_id in set(true_evt):
    ls_index = np.where(true_evt[:] == evt_id)[0]
    for i in range(len(ls_index)):
        all_dep[i].extend(energy_tot[ls_index[i]])

# Create histograms for energy distributions
fig, ax = plt.subplots(2, 2)
ax[0][0].hist(first_dep, bins=50)
ax[0][0].set_xlabel("Energy [MeV] 1st hit")
ax[1][0].hist(second_dep, bins=50)
ax[1][0].set_xlabel("Energy [MeV] 2nd hit")
ax[0][1].hist(third_dep, bins=50)
ax[0][1].set_xlabel("Energy [MeV] 3rd hit")
ax[1][1].hist(fourth_dep, bins=50)
ax[1][1].set_xlabel("Energy [MeV] 4th hit")

# Set limits and display the plot
for i in range(2):
    for j in range(2):
        ax[i][j].set_xlim(0, 0.15)
fig.suptitle("Energy spectrum for individual LPTMs of elecsim file 0")
plt.show()
