from sksurgerynditracker.nditracker import NDITracker
import numpy as np
import os
from pathlib import Path
from sksurgerycore.algorithms.procrustes import orthogonal_procrustes
from sksurgerycalibration.video.video_calibration_utils import extrinsic_vecs_to_matrix


def perform_point_registration(fixed, moving): # TODO- when would I be able to obtain fixed and moving points? Implement with rest of code
    '''
    CT points --> fixed
    pointer tip points --> moving

    returns 4x4 matrix to convert pointer tip coords to CT coords
    '''

    # converting 4x4 tracking matrices to points
    R, t, FRE = orthogonal_procrustes(fixed, moving) #fixed, moving
    T = extrinsic_vecs_to_matrix(R, t)
    return T
    

def perform_pointer_calibration():
    '''
    obtain transform between pointer's marker and pointer tip
    ''' 
    T = np.eye(4) # TODO convert this to either pivot calibration or template calib
    return T


def record_data_pointer(save_folder, CT_T_marker , rom_files_list=["../data/8700339.rom"], num_points=10):
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
    vecs_all = [] # tracking in marker coord system
    vecs_CT_all = [] # tracking in CT coord system
    for i in range(num_points):
        # recordinng frames and Rt vectors
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        #tracking_in_CT = tracking[0] @ CT_T_marker
        # SAVE 4X4 MATRICES
        vecs_all.append(tracking[0])
        #vecs_CT_all.append(tracking_in_CT)
        
    
    # SAVE tracking
    np.save(f'{save_folder}/tumour_points_marker_coords.npy', np.array(vecs_all))
    #np.save(f'{save_folder}/tracking_CT_coords', np.array(vecs_CT_all))

    TRACKER.stop_tracking()
    TRACKER.close()


def main():
    project_path = Path(__file__).parent.resolve()

    save_folder = f'{project_path}/assets/data/'

    NUM_POINTS = 30 # number of tracking points recorded
    ROM_FILES_LIST = [""]

    # obtain marker to tip using pivot calibration
    tip_T_marker = np.eye(4)
    # obtain tip to CT using point registration
    fixed = np.loadtxt('data/CT_fiducial_points.txt')
    moving = np.load('data/registration.npy')
    CT_T_tip = perform_point_registration(fixed, moving)

    # combine transform
    CT_T_marker = CT_T_tip @ tip_T_marker

    input('press enter when you have placed the pointer on the tumour location')
    record_data_pointer(save_folder, CT_T_marker ,rom_files_list=ROM_FILES_LIST, num_points=NUM_POINTS)


if __name__ == "__main__":
    main()