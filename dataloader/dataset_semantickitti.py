# -*- coding:utf-8 -*-
# author: Xinge

"""
SemKITTI dataloader
"""
import numpy as np
import torch
import numba as nb
from torch.utils import data
import time
import random

from utils.util import Bresenham3D
REGISTERED_DATASET_CLASSES = {}

def from_voxel_to_voxel(max_volume_space,min_volume_space,intervals,pose0,pose):
    x_bias = (max_volume_space[0] - min_volume_space[0])/2
    max_bound = np.asarray(max_volume_space)
    min_bound = np.asarray(min_volume_space)
    min_bound[0] -= x_bias
    max_bound[0] -= x_bias
    max_bound2 = np.asarray(max_volume_space)
    min_bound2 = np.asarray(min_volume_space)

    voxel_grid = np.indices((256, 256, 32)).transpose(1, 2, 3, 0)
    voxel_grid = voxel_grid.reshape(-1,3)
    full_voxel_center=(voxel_grid.astype(np.float32) + 0.5) * intervals + min_bound
    full_voxel_center[:,0]+= x_bias
    current_vox_center=frame_transform_scan(full_voxel_center,pose0,pose)
    current_vox_center=np.concatenate([current_vox_center,voxel_grid],1)
    vox_xyz0 = current_vox_center
    for ci in range(3):
        vox_xyz0[current_vox_center[:, ci] < min_bound2[ci], :] = 1000
        vox_xyz0[current_vox_center[:, ci] > max_bound2[ci], :] = 1000
    vox_valid_inds = vox_xyz0[:, 0] != 1000
    current_vox_center = current_vox_center[vox_valid_inds, :]
    current_vox_center[:, 0]-= x_bias    # current_vox_center is 
    vox_grid_from=current_vox_center[:,-3:]
    vox_grid_to=current_vox_center[:,:3]
    vox_grid_to = (np.floor((np.clip(vox_grid_to, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)
    return vox_grid_from,vox_grid_to

def frame_transform_scan(points, pose0, pose):

    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
    new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
    new_points = new_points[:, :3]
    new_coords = new_points - pose0[:3, 3]
    new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
    new_coords = np.hstack((new_coords, points[:, 3:]))

    return new_coords
    
def register_dataset(cls, name=None):
    global REGISTERED_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_DATASET_CLASSES, f"exist class: {REGISTERED_DATASET_CLASSES}"
    REGISTERED_DATASET_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_DATASET_CLASSES
    assert name in REGISTERED_DATASET_CLASSES, f"available class: {REGISTERED_DATASET_CLASSES}"
    return REGISTERED_DATASET_CLASSES[name]


@register_dataset
class voxel_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, 50, 1.5], min_volume_space=[-50, -50, -3]):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.flip_aug = flip_aug
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data,prev_data,pose_list,stride_list = self.point_cloud_dataset[index]
        
        if len(data) == 4:                                                   
            xyz, labels, sig, origin_len = data
            prev_xyz, prev_labels, prev_sig,_ = prev_data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        origin_len = len(xyz)
        max_bound = np.asarray(self.max_volume_space)
        min_bound = np.asarray(self.min_volume_space)
        
        ### Cut point cloud and segmentation label for valid range                             
        xyz0 = xyz
        for ci in range(3):
            xyz0[xyz[:, ci] < min_bound[ci], :] = 1000
            xyz0[xyz[:, ci] > max_bound[ci], :] = 1000
        valid_inds = xyz0[:, 0] != 1000
        xyz = xyz[valid_inds, :]
        sig = sig[valid_inds]

        ### post_scan_preprocess  ###
        
        prev_raws=[]
        prev_sigs=[]
        prev_vox=[]
        prev_velodyne=[]
        prev_trans_sigs=[]
        
        if len(pose_list)>1:
            prev_frames_num=len(prev_xyz)
            for f_idx in range(prev_frames_num):
                
                ### cut point clound
                prev_xyz0 = prev_xyz[f_idx]
                prev_xyz_sin=prev_xyz[f_idx]
                prev_sig_sin=prev_sig[f_idx]
                for ci in range(3):
                    prev_xyz0[prev_xyz_sin[:, ci] < min_bound[ci], :] = 1000
                    prev_xyz0[prev_xyz_sin[:, ci] > max_bound[ci], :] = 1000
                valid_inds = prev_xyz0[:, 0] != 1000
                prev_xyz_single = prev_xyz_sin[valid_inds, :]
                prev_sig_single = prev_sig_sin[valid_inds]
                prev_raws.append(prev_xyz_single)
                prev_sigs.append(prev_sig_single)
                
                ### Transform prev point cloud
                transformed_prev_xyz = frame_transform_scan(prev_xyz[f_idx], pose_list[0], pose_list[f_idx+1])
                transformed_prev_velodyne=frame_transform_scan(np.zeros([1,4]), pose_list[0], pose_list[f_idx+1])
                
                ### Cut prev point cloud
                prev_xyz0 = transformed_prev_xyz
                for ci in range(3):
                    prev_xyz0[transformed_prev_xyz[:, ci] < min_bound[ci], :] = 1000
                    prev_xyz0[transformed_prev_xyz[:, ci] > max_bound[ci], :] = 1000
                prev_valid_inds = prev_xyz0[:, 0] != 1000
                transformed_prev_xyz = transformed_prev_xyz[prev_valid_inds, :]
                transformed_prev_sig_single = prev_sig_sin[prev_valid_inds]
                prev_vox.append(transformed_prev_xyz)
                prev_trans_sigs.append(transformed_prev_sig_single)
                prev_velodyne.append(transformed_prev_velodyne)
                
        # transpose centre coord for x axis                                                       
        x_bias = (self.max_volume_space[0] - self.min_volume_space[0])/2
        min_bound[0] -= x_bias
        max_bound[0] -= x_bias
        xyz[:, 0] -= x_bias
        if len(pose_list)>1:
            for f_idx in range(prev_frames_num):
                prev_raws_sim=prev_raws[f_idx]
                prev_vox_sin=prev_vox[f_idx]
                prev_raws_sim[:, 0]-= x_bias
                prev_vox_sin[:, 0]-= x_bias
                prev_velodyne_sin=prev_velodyne[f_idx]
                prev_velodyne_sin=prev_velodyne_sin[:,:3]
                prev_velodyne_sin[:,0]-= x_bias
                prev_vox[f_idx]=prev_vox_sin
                prev_velodyne[f_idx]=prev_velodyne_sin
                prev_raws[f_idx]=prev_raws_sim

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)
        if (intervals == 0).any(): print("Zero interval!")
        
        grid_ind = (np.floor((np.clip(xyz, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz), axis=1)
        return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)  # 7:xyz_bias + xyz + intensity
        
        
        prev_grid_list=[]
        prev_fea_list=[]
        prev_label_list=[]
        prev_transformed_fea_list=[]
        prev_transformed_grid_list=[]
        if len(pose_list)>1:
            for f_idx in range(prev_frames_num):
                single_prev_xyz=prev_raws[f_idx]
                single_prev_sig=prev_sigs[f_idx]
                single_trans_prev_sig=prev_trans_sigs[f_idx]
                single_prev_vox=prev_vox[f_idx]

                prev_grid_ind = (np.floor((np.clip(single_prev_xyz, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)
                prev_voxel_centers = (prev_grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
                return_prev_xyz = single_prev_xyz - prev_voxel_centers
                return_prev_xyz = np.concatenate((return_prev_xyz, single_prev_xyz), axis=1)
                return_prev_fea = np.concatenate((return_prev_xyz, single_prev_sig[..., np.newaxis]), axis=1)  # 7:xyz_bias + xyz + intensity
                
                prev_transformed_grid_ind = (np.floor((np.clip(single_prev_vox, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)
                prev_transformed_voxel_centers = (prev_transformed_grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
                return_transformed_prev_xyz = single_prev_vox - prev_transformed_voxel_centers
                return_transformed_prev_xyz = np.concatenate((return_transformed_prev_xyz, single_prev_vox), axis=1)
                return_transformed_prev_fea = np.concatenate((return_transformed_prev_xyz, single_trans_prev_sig[..., np.newaxis]), axis=1)  # 7:xyz_bias + xyz + intensity
                
                prev_fea_list.append(return_prev_fea)
                prev_transformed_fea_list.append(return_transformed_prev_fea)
                prev_transformed_grid_list.append(prev_transformed_grid_ind)
                prev_grid_list.append(prev_grid_ind) 
                prev_label_list.append(prev_labels[f_idx]) 
   
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        processed_label = labels  # voxel labels

        data_tuple = (voxel_position, processed_label)
        
        vox_from_list=[]
        vox_to_list=[]
        if len(pose_list)>1:
            for f_idx in range(prev_frames_num):
                vox_grid_from,vox_grid_to= from_voxel_to_voxel(self.max_volume_space,self.min_volume_space,intervals,pose_list[0],pose_list[f_idx+1])
                vox_from_list.append(vox_grid_from)
                vox_to_list.append(vox_grid_to)
                
        else:
            vox_grid_to=np.array([[0,0,0]])
            vox_grid_from=np.array([[0,0,0]])
            vox_from_list.append(vox_grid_from)
            vox_to_list.append(vox_grid_to)

        
        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)

        data_tuple += (origin_len,prev_velodyne,vox_to_list,vox_from_list,prev_grid_list,prev_fea_list,prev_transformed_grid_list,prev_transformed_fea_list,prev_label_list,min_bound,max_bound,intervals,stride_list)

        return data_tuple


# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)

@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label



def collate_fn_BEV_ms_tta(data):
    label2stack = np.stack([d[1] for d in data]).astype(np.int)      
    grid_ind_stack = [d[2] for d in data]         
    xyz = [d[4] for d in data]                    
    index = [d[5] for d in data]                  
    prev_velodyne=[d[7] for d in data]
    vox_grid_to=[d[8] for d in data]
    vox_grid_from=[d[9] for d in data]
    prev_grid_ind=[d[10] for d in data]
    prev_feat=[d[11] for d in data]
    prev_trans_ind=[d[12] for d in data]
    prev_trans_feat=[d[13] for d in data]
    min_bound=[d[15] for d in data]
    max_bound=[d[16] for d in data]
    interval=[d[17] for d in data]
    strides=[d[18] for d in data]
    current_frame={'grid_ind':grid_ind_stack, 'pt_feat': xyz, 'index': index, 'gt':torch.from_numpy(label2stack),'min_bound':min_bound,'max_bound':max_bound,'interval':interval,'stride':strides[0]}
    prev_frame={'vox_grid_to':vox_grid_to[0], 'vox_grid_from': vox_grid_from[0], 'grid_ind': prev_grid_ind[0], 'pt_feat':prev_feat[0],'trans_grid_ind':prev_trans_ind[0],'trans_pt_feat':prev_trans_feat[0],'lidar_pose':prev_velodyne[0]}
    
    return current_frame,prev_frame


def collate_fn_BEV(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz

def collate_fn_BEV_tta(data):    
    voxel_label = []
    for da1 in data:
        for da2 in da1:
            voxel_label.append(da2[1])
    grid_ind_stack = []
    for da1 in data:
        for da2 in da1:
            grid_ind_stack.append(da2[2])
    point_label = []
    for da1 in data:
        for da2 in da1:
            point_label.append(da2[3])
    xyz = []
    for da1 in data:
        for da2 in da1:
            xyz.append(da2[4])
    index = []
    for da1 in data:
        for da2 in da1:
            index.append(da2[5])
    return xyz, xyz, grid_ind_stack, point_label, xyz, index

def collate_fn_BEV_ms(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    index = [d[5] for d in data]
    origin_len = [d[6] for d in data]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz, index, origin_len

