from sksurgerynditracker.nditracker import NDITracker
import numpy as np
import os
from pathlib import Path


def record_data_pointer(save_folder,rom_files_list=["../data/8700339.rom"], num_points=10):
    """
    function that records tracking info from tracked pointer
    """
    
    # CREATING FOLDERS WHERE TO SAVE TRACKING INFO
    if not os.path.isdir(f'{save_folder}'):
        os.makedirs(f'{save_folder}')

    # initialising tracker
    print('init params')
    # init params
    SETTINGS = {
    "tracker type": "polaris",
    "romfiles" : rom_files_list
        }
    TRACKER = NDITracker(SETTINGS)
    TRACKER.start_tracking()
    print('finished init')
    
    # obtaining a bunch of tracking points
    vecs_all = []
    for i in range(num_points):
        # recordinng frames and Rt vectors
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        # SAVE 4X4 MATRICES
        vecs_all.append(tracking[0])
    
    # SAVE tracking
    np.save(f'{save_folder}/tracking', np.array(vecs_all))

    TRACKER.stop_tracking()
    TRACKER.close()


def main():
    project_path = Path(__file__).parent.resolve()

    # folder will be structured as follows:
    # assets/type/folder/images
    type = 'EM'  # random / phantom / EM_tracker_calib / tests
    folder = 'glove_box_clamps'

    save_folder = f'{project_path}/assets/data/{type}/{folder}'

    NUM_POINTS = 30 # number of tracking points recorded
    ROM_FILES_LIST = [""]

    record_data_pointer(save_folder,rom_files_list=ROM_FILES_LIST, num_points=NUM_POINTS)

if __name__ == "__main__":
    main()