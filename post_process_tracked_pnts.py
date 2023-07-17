from sksurgerynditracker.nditracker import NDITracker
import numpy as np
import os
from pathlib import Path
from sksurgerycore.algorithms.procrustes import orthogonal_procrustes
from sksurgerycalibration.video.video_calibration_utils import extrinsic_vecs_to_matrix
import cv2
import matplotlib.pyplot as plt


def add_scatter(ax, pnts_3d, color='b'):
    for i in range(len(pnts_3d)): #plot each point + it's index as text above
        ax.scatter(pnts_3d[i,0],pnts_3d[i,1],pnts_3d[i,2],color=color) 
        ax.text(pnts_3d[i,0],pnts_3d[i,1],pnts_3d[i,2],  '%s' % (str(i)), size=20, zorder=1,  
        color='k') 

def perform_point_registration(fixed, moving, tip_T_marker): # TODO- when would I be able to obtain fixed and moving points? Implement with rest of code
    '''
    CT points --> fixed
    pointer tip points --> moving

    returns 4x4 matrix to convert pointer tip coords to CT coords
    '''

    # converting 4x4 tracking matrices to points
    fixed_pnts = fixed[:-1,1:]
    T_total =tip_T_marker @ moving[:,:,:] # 
    pnts_hom = T_total  @ np.array([0,0,0,1])
    moving_pnts = cv2.convertPointsFromHomogeneous(pnts_hom).squeeze()
    R, t, FRE = orthogonal_procrustes(fixed_pnts, moving_pnts) #fixed, moving
    T = extrinsic_vecs_to_matrix(cv2.Rodrigues(R)[0], t)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(moving_pnts[:,0], moving_pnts[:,1], moving_pnts[:,2])
    ax.scatter(fixed_pnts[:,0], fixed_pnts[:,1], fixed_pnts[:,2])
    plt.show()

    converted_moving = np.zeros(pnts_hom.shape)
    for IDX, points in enumerate(pnts_hom):
        converted_moving[IDX]=T@points

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    add_scatter(ax, converted_moving, color='b')
    add_scatter(ax, fixed_pnts, color='r')


    #ax.scatter(converted_moving[:,0], converted_moving[:,1], converted_moving[:,2])
    #ax.scatter(fixed_pnts[:,0], fixed_pnts[:,1], fixed_pnts[:,2])
    plt.show()
    return T


def calculate_euclid_dist(pointer_tip_in_mri_space,tumour_in_mri_space ):
    """
    calculate_euclid_dist(pointer_tip_in_mri_space,tumour_in_mri_space )
    calculates euclidean distance between two 3D points (euclid_dist is x^2+y^2+x^2)

    Args:
        pointer_tip_in_mri_space: 4x1 np array of where we think the tumor is
        tumour_in_mri_space: 4x1 np array of where the tumor actually is

    Returns:
        euclidean distance between the two points
    """
    voxel_dims = np.array([0.7754, 0.7754, 0.8330])

    dist =  (pointer_tip_in_mri_space[0] - tumour_in_mri_space[0]) \
            * (pointer_tip_in_mri_space[0] - tumour_in_mri_space[0]) \
            * voxel_dims[0] \
            + (pointer_tip_in_mri_space[1] - tumour_in_mri_space[1]) \
            * (pointer_tip_in_mri_space[1] - tumour_in_mri_space[1]) \
            * voxel_dims[1] \
            + (pointer_tip_in_mri_space[2] - tumour_in_mri_space[2]) \
            * (pointer_tip_in_mri_space[2] - tumour_in_mri_space[2]) \
            * voxel_dims[2] 
    
    return np.sqrt(dist)

    

def main(): 

    # marker to tip (obtained from Joao)
    tip_T_marker = extrinsic_vecs_to_matrix(np.array([0,0,0],dtype=np.float32),np.array([-18.5, 0.6, -157.6], dtype=np.float64))
    
    # obtain tip to CT using point registration
    fixed = np.loadtxt('data/CT_fiducial_points.txt')
    moving = np.load('registration.npy')
    

    CT_T_tip = perform_point_registration(fixed, moving, tip_T_marker)

    # combine transform
    CT_T_marker = CT_T_tip @ tip_T_marker

    # loading all points to mark tumour
    pnts_pointer_tip = np.load('/Users/aure/PycharmProjects/scikit_surgery_ndi/data/tumour_points_marker_coords.npy')
    T_total = CT_T_marker @ pnts_pointer_tip
    pnts_tumour_CT = T_total@np.array([0,0,0,1])   
    pnt_tumour_CT = pnts_tumour_CT.mean(axis=0)[:3]
    pnt_tumour_CT_gt = fixed[-1,1:]

    accuracy = calculate_euclid_dist(pnt_tumour_CT_gt,pnt_tumour_CT)

    print('tumour original CT')
    print(pnt_tumour_CT)
    print('tumour predicted CT')
    print(pnt_tumour_CT_gt)
    print('accuracy')
    print(accuracy)

    print('accuracy in mm')


    return 


if __name__=='__main__': 
    main() 