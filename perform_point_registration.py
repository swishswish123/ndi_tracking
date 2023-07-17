import os

from sksurgerynditracker.nditracker import NDITracker
import numpy as np


def record_fiducial_points(save_folder, points_dictionary, rom_path="../data/8700339.rom"):

    # create folder where to save fiducial data
    if not os.path.isdir(f'{save_folder}'):
        os.makedirs(f'{save_folder}')

    # initialising tracker
    print('init params')
    SETTINGS = {
        "tracker type": "vega",
        "ip address": "169.254.59.34",
        "port": 8765,
        "romfiles": [
            rom_path
        ]
    }
    TRACKER = NDITracker(SETTINGS)
    TRACKER.start_tracking()
    print('finished init')

    vecs_all = []  # tracking in marker coord system
    
    # for each fiducial point,
    for key in points_dictionary:
        input(f"place pointer on fiducial marker {key} which is on the patient's {points_dictionary[key]}. Once done press enter")
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        # TODO check if need to convert to pointer tip coordinates
        print(tracking)
        vecs_all.append(tracking[0])

    # SAVE tracking
    np.save(f'{save_folder}/tracking', np.array(vecs_all))

    TRACKER.stop_tracking()
    TRACKER.close()


def main():

    # defining fiducials and what they represent
    points_dictionary = {
        0:'bottom right',
        1:'top right',
        2:'top middle',
        3:'top left',
        4:'bottom left',
        5:'tumour'
    }
    save_folder = 'data/'
    record_fiducial_points(save_folder, points_dictionary, rom_path="config/rom_files/8700340.rom")


if __name__ == '__main__':
    main()