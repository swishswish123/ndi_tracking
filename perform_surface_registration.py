import os

from sksurgerynditracker.nditracker import NDITracker
import numpy as np


def record_surface_points(save_folder, rom_path="../data/8700339.rom"):

    # create folder where to save data
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
    
    input(f"press enter when ready to start surface recording")

    # for each fiducial point,
    for i in range(1000):
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        # TODO check if need to convert to pointer tip coordinates
        print(tracking)
        vecs_all.append(tracking[0])

    # SAVE tracking
    np.save(f'{save_folder}/surface_registration', np.array(vecs_all))

    TRACKER.stop_tracking()
    TRACKER.close()


def main():

    save_folder = 'data/'
    record_surface_points(save_folder, rom_path="config/rom_files/8700340.rom")


if __name__ == '__main__':
    main()