"""
Classify Events from DETSIM Files

This script classifies events from DETSIM files based on the types of initial particles detected in each event.
It reads data from ROOT files, determines the event type using specific criteria, and saves the classification results to a text file.

Created by: Lukas Peron
Last Update: 01/09/2024

Overview:
---------
The script performs the following steps:
1. Load DETSIM data files and extract relevant branches and attributes.
2. Classify each event based on the initial particles and their properties.
3. Store the classification results and energy deposition in a list.
4. Save the classification results to a text file for further analysis.

Error Handling:
---------------
- The file loading process includes error handling to ensure that the ROOT files are successfully opened. 
  If a file cannot be opened or is corrupted, an error is raised with a descriptive message.
- The branch and attribute loading functions include basic checks to ensure that the requested branches or attributes are available.
- The event classification function (`event_type`) assumes valid input data; additional error handling may be added as needed.

Dependencies:
-------------
- numpy
- ROOT
- my_package (contains `detsim_branches`)

Instructions:
-------------
1. Ensure that numpy, ROOT, and my_package are installed and configured in your environment.
2. Run the script to classify events and save the results to the specified directory.
"""

import my_package.detsim_branches as detsim
import numpy as np

def event_type(nInitParticles, InitPDGID):
    """
    Classify the type of event based on the number and types of initial particles.

    Parameters:
    -----------
    nInitParticles : int
        Number of initial particles in the event.
    InitPDGID : list of int
        List of Particle Data Group (PDG) IDs for the initial particles.

    Returns:
    --------
    int
        The type of event: (from V. Lebrin thesis, 'Towards the Detection of Core-Collapse Supernovae Burst Neutrinos with the 3-inch PMT System of the JUNO Detector', Table 2.3)
        0 - pES (proton Elastic Scattering)
        1 - IBD (Inverse Beta Decay)
        2 - vbarC (antineutrino Charged Current interaction with a Carbon atom)
        3 - vCstar (neutrino Neutral Current interaction with a Carbon atom realising a photon)
        4 - eES_or_vC (electron Elastic Scattering or neutrino Charged or Neutral Current interaction with a Carbon atom)
        -1 - Undefined event type
    """
    # Mapping of (nInitParticles, first PDGID, optional second PDGID) to event type
    event_type_map = {
        (1, 2212): 0,        # pES
        (2, -11, 2112): 1,   # IBD
        (1, -11): 2,         # vbarC
        (1, 22): 3,          # vCstar
        (1, 11): 4           # eES_or_vC
    }

    # Tuple key for looking up in event_type_map
    event_key = (nInitParticles, InitPDGID[0])
    if nInitParticles == 2:
        event_key += (InitPDGID[1],)

    return event_type_map.get(event_key, -1)

# Initialize list to store event classification results
event_type_ls = [["num_file", "num_entry", "type", "edep[MeV]"]]

# Loop through each file and classify events
for num_file in range(120):
    try:
        # Load the DETSIM file and extract relevant branches
        file = detsim.load_file(num_file)
        tree = file.Get("geninfo")
        tree_evt = file.Get("evt")
        nInitParticles, InitPDGID, evtID, edep = detsim.load_branches(tree, "nInitParticles", "InitPDGID", "evtID", "edep")
    except Exception as e:
        print(f"Error loading file {num_file}: {e}")
        continue

    num_entry = 0
    for entry in tree:
        try:
            # Get event ID and energy deposition for each entry
            evtID, nInitParticles, InitPDGID = detsim.load_attributes(entry, "evtID", "nInitParticles", "InitPDGID")
            tree_evt.GetEntry(num_entry)
            edep = detsim.load_attributes(tree_evt, "edep")
            # Classify the event and append results to the list
            event_type_ls.append([num_file, evtID, event_type(nInitParticles, InitPDGID), edep])
        except Exception as e:
            print(f"Error processing entry {num_entry} in file {num_file}: {e}")
            continue

        num_entry += 1

# Convert results list to numpy array and save to file
event_type_ls = np.array(event_type_ls)
np.savetxt("/pbs/home/l/lperon/work_JUNO/code/txt/detsim/event_classification.txt", event_type_ls, fmt='%s')
