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

import open3d as o3d
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def perform_surface_registration(fixed, moving,tip_T_marker):
    '''
    fixed: CT scan 3D point cloud
    moving: tracked 4x4 matrices
    tip_T_marker: transform from marker to tip
    '''
    # converting 4x4 tracking matrices to points
    #fixed_pnts = fixed[:-1,1:]
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    fixed_o3d = o3d.geometry.PointCloud()
    fixed_o3d.points = o3d.utility.Vector3dVector(fixed)

    # converting to pointer tip coords
    T_total = tip_T_marker @ moving[:,:,:] 
    pnts_hom = T_total  @ np.array([0,0,0,1])
    moving_pnts = cv2.convertPointsFromHomogeneous(pnts_hom).squeeze()

    moving_o3d = o3d.geometry.PointCloud()
    moving_o3d.points = o3d.utility.Vector3dVector(moving_pnts)

    draw_registration_result(moving_o3d, fixed_o3d, np.eye(4))

    # PERFORM regisrtation
    moving_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.geometry.PointCloud.estimate_covariances(moving_o3d)
    fixed_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.geometry.PointCloud.estimate_covariances(fixed_o3d)

    # generalised ICP
    target_T_source_generalised = o3d.pipelines.registration.registration_icp(
        moving_o3d, fixed_o3d, max_correspondence_distance=10000,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
        )
    
    # Define the parameters for the ICP algorithm
    threshold = 1000 # Distance threshold for corresponding points
    trans_init = np.eye(4)  # Initial transformation
    target_T_source_ICP = o3d.pipelines.registration.registration_icp(
        moving_o3d, fixed_o3d, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    T = target_T_source_generalised.transformation

    draw_registration_result(moving_o3d, fixed_o3d, T)
    
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

    moving = np.load('registration.npy')
    method = 'surface'

    # marker to tip (obtained from Joao)
    R_tip = np.array([0,0,0],dtype=np.float32)
    T_tip = np.array([-18.5, 0.6, -157.6], dtype=np.float64)
    tip_T_marker = extrinsic_vecs_to_matrix(R_tip,T_tip)
    

    if method == 'point':
        # obtain tip to CT using point registration
        fiducial_points = np.loadtxt('data/CT_fiducial_points.txt')
        
        CT_T_tip = perform_point_registration(fiducial_points, moving, tip_T_marker)

    elif method == 'surface':
        # CT scan
        CT_file='data/phantom_surface_CT.ply'
        pcd_CT = o3d.io.read_point_cloud(CT_file)
        CT_points_np = np.array(pcd_CT.points)
        
        CT_T_tip = perform_surface_registration(CT_points_np, moving, tip_T_marker)

    # combine transform
    CT_T_marker = CT_T_tip @ tip_T_marker

    # loading all points to mark tumour
    pnts_pointer_tip = np.load('/Users/aure/PycharmProjects/scikit_surgery_ndi/data/tumour_points_marker_coords.npy')
    T_total = CT_T_marker @ pnts_pointer_tip
    pnts_tumour_CT = T_total@np.array([0,0,0,1])   
    pnt_tumour_CT = pnts_tumour_CT.mean(axis=0)[:3]
    fixed = np.loadtxt('data/CT_fiducial_points.txt')
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