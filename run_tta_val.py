# -*- coding:utf-8 -*-
import os
import time
import argparse
import sys
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataloader.pc_dataset import get_SemKITTI_label_name, get_eval_mask, unpack
from config.config import load_config_data
from builder import loss_builder
from builder import model_builder_unlock as model_builder
from builder import data_builder as data_builder
from utils.load_save_util import load_checkpoint
import warnings
warnings.filterwarnings("ignore")
import yaml
from utils.util import Bresenham3D
import random

import ast
import pdb
import copy
import importlib.util as ilutil
import glob
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from utils.np_ioueval import iouEval
from utils.softmax_entropy import softmax_entropy

import pandas as pd
import seaborn as sn

CATS = ["empty", 
        "car", "bicycle", "motorcycle", "truck", "other-vehicle", 
        "person", "bicyclist", "motorcyclist", "road", "parking", 
        "sidewalk", "other-ground", "building", "fence", "vegetation", 
        "trunk", "terrain", "pole", "traffic-sign"]

mapping_forward = {0:0, 
                   1:0, 10:1, 11:2, 13:5, 15:3,
                   16:5, 18:4, 20:5, 30:6, 31:7,
                   32:8, 40:9, 44:10, 48:11, 49:12,
                   50:13, 51:14, 52:0, 60:9, 70:15,
                   71:16, 72:17, 80:18, 81:19, 99:0,
                   252:1, 253:7, 254:6, 255:8, 256:5,
                   257:5, 258:4, 259:5}

max_key = max(mapping_forward.keys())

MAP_ARRAY = np.zeros(max_key + 1, dtype=int)

for key, value in mapping_forward.items():
    MAP_ARRAY[key] = value

PALLETE = np.asarray([[0, 0, 0],[245, 150, 100],[245, 230, 100],[150, 60, 30],[180, 30, 80],[255, 0, 0],[30, 30, 255],[200, 40, 255],[90, 30, 150],[255, 0, 255],
                     [255, 150, 255],[75, 0, 75],[75, 0, 175],[0, 200, 255],[50, 120, 255],[0, 175, 0],[0, 60, 135],[80, 240, 150],[150, 240, 255],[0, 0, 255], [255,255,255]]).astype(np.uint8)
PALLETE[:,[0,2]]=PALLETE[:,[2,0]]
PALLETE_BINARY = np.asarray([[255,0,0], [0,255,0], [0,0,255], [0,0,0]]).astype(np.uint8)

def train2SemKITTI(input_label):
    # delete 0 label (uses uint8 trick : 0 - 1 = 255 )
    return input_label + 1

def get_remap_first(semkittiyaml):
    # make lookup table for mapping
    learning_map_inv = semkittiyaml["learning_map_inv"]
    maxkey = max(learning_map_inv.keys())
    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(learning_map_inv.keys())] = list(learning_map_inv.values())
    return remap_lut,learning_map_inv

def get_remap_second(semkittiyaml):
    class_remap = semkittiyaml["learning_map"]
    maxkey2 = max(class_remap.keys())
    remap_lut = np.zeros((maxkey2 + 100), dtype=np.int32)
    remap_lut[list(class_remap.keys())] = list(class_remap.values())
    remap_lut[remap_lut == 0] = 255   # map 0 to 'invalid'
    remap_lut[0] = 0 
    return remap_lut


def remapping(pred, remap=None):
    ### save prediction after remapping
    upper_half = pred >> 16  # get upper half for instances
    lower_half = pred & 0xFFFF  # get lower half for semantics
    lower_half = remap[lower_half]  # do the remapping of semantics
    pred = (upper_half << 16) + lower_half  # reconstruct full label
    pred = pred.astype(np.uint32)
    pred = pred.astype(np.uint16)
    return pred

def save_pred(final_preds, save_dir, output_path):
    _, dir2 = save_dir.split('/sequences/',1)
    new_save_dir = output_path + '/sequences/' + dir2.replace('velodyne', 'predictions')[:-3]+'label'
    if not os.path.exists(os.path.dirname(new_save_dir)):
        try:
            os.makedirs(os.path.dirname(new_save_dir))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    final_preds.tofile(new_save_dir)

def extract_bev_for_vis(arr, ignore_idx=0):
    for i in arr[::-1]:
        if i != ignore_idx:
            return i
    return ignore_idx

def extract_mev_for_vis(arr, ignore_idx=0):
    for i in arr:
        if i != ignore_idx:
            return i
    return ignore_idx

def extract_bev_for_vis_dual(arr, ignore_idx=(0,255)):
    for i in arr[::-1]:
        if i not in ignore_idx:
            return i
    return 255


def main(args):
    pytorch_device = torch.device('cuda:0')
    epsilon = np.finfo(np.float32).eps

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']
    val_batch_size = val_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']
    loss_fn_ce, loss_fn_lovasz = loss_builder.build(wce=True, lovasz=True, num_class=num_class, ignore_label=ignore_label)
    loss_fn_ce_binary, loss_fn_lovasz_binary = loss_builder.build(wce=True, lovasz=True, num_class=2, ignore_label=ignore_label)

    # Define dataset/loader
    with open("config/label_mapping/semantic-kitti.yaml", 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    remap_first,class_inv_remap = get_remap_first(semkittiyaml)
    remap_second = get_remap_second(semkittiyaml)
    class_strings = semkittiyaml["labels"]
    strides=[int(num) for num in ast.literal_eval(args.stride)]
        
    _, test_dataset_loader, test_pt_dataset = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size,
                                                                  use_tta=True,
                                                                  use_multiscan=True,
                                                                  stride=args.stride)

    # Define experiment path
    exp_name = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + '_' + __file__.replace('run_tta_','').replace('.py','') + args.name
    exp_path = args.talos_root+'experiments/' + exp_name
    print("Experiment path is "+exp_path)
    os.makedirs(exp_path, exist_ok=True)
    
    # Save run_code snapshots
    os.system('scp -r '+args.talos_root+__file__+' '+exp_path+'/'+__file__)
    
    
    writer = SummaryWriter(exp_path)
    config_file = exp_path + '/config.txt'
    with open(config_file, 'w') as log:
        log.write(str(args))

    loss_cont_names = ['loss_cont', 'loss_cont_occ_ce', 'loss_cont_occ_lovasz', 'loss_cont_pgt_ce','loss_cont_pgt_lovasz']
    loss_adapt_names = ['loss_adapt', 'loss_adapt_occ_ce', 'loss_adapt_occ_lovasz', 'loss_adapt_pgt_ce','loss_adapt_pgt_lovasz']
    evaluator_all = iouEval(num_class, [])
    baseline_performance = open(args.baseline_perf_txt, 'r')
    baseline_prediction_paths = sorted(glob.glob(args.baseline_preds+'/*.label'))

    model_load_path = train_hypers['model_load_path']
    model_load_path += 'iou37.5557_epoch3.pth'
    
    model_baseline = model_builder.build(model_config)
    print('Load model from: %s' % model_load_path)
    model_baseline = load_checkpoint(model_load_path, model_baseline)
    model_baseline.to(pytorch_device)

    # For freeze
    module_names_mlp = ['cylinder_3d_generator']
    module_names_comp = ['a_conv1', 'a_conv2', 'a_conv3', 'a_conv4', 'a_conv5', 'a_conv6', 'a_conv7', 'ch_conv1','res_1','res_2','res_3']
    module_names_seg = ['downCntx', 'resBlock2', 'resBlock3', 'resBlock4', 'resBlock5', 'upBlock0', 'upBlock1', 'upBlock2', 'upBlock3', 'ReconNet']
    module_names_logit = ['logits']

    assert (args.do_adapt or args.do_cont)

    if args.do_cont:
        print("continual")
        model_cont = copy.deepcopy(model_baseline)
        param_to_update_cont = []
        
        print("Update segmentation module for continual tta.")
        for tn in module_names_seg:
            param_to_update_cont += list(getattr(model_cont.cylinder_3d_spconv_seg, tn).parameters())
        
        optimizer_cont = optim.Adam(param_to_update_cont, lr=args.cont_lr)

    if args.do_adapt:
        print("scan-wise adaptation")
    
    # current_frame={'grid_ind','pt_feat','index','gt'}
    # prev_frame={'vox_grid_to','vox_grid_from','grid_ind','pt_feat','trans_grid_ind','trans_pt_feat','lidar_pose'}
    for idx_test, (frame_curr, frame_aux) in enumerate(tqdm(test_dataset_loader)):

        print('')

        #########################################################################################################################################
        ############################################################# Data process ##############################################################
        #########################################################################################################################################

        exist_stride = [stride in frame_curr['stride'] for stride in strides]
        
        flag_aux_exist= bool(len(frame_aux['trans_grid_ind'])!=0)
        flag_adapt_aux_exist= bool(exist_stride[0])
        flag_cont_aux_exist= bool(exist_stride[1])
        adapt_aux_idx= 0
        cont_aux_idx = 1 if flag_adapt_aux_exist else 0

        feat_curr = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in frame_curr['pt_feat']]
        grid_curr = [torch.from_numpy(i).to(pytorch_device) for i in frame_curr['grid_ind']]
        gt_curr = frame_curr['gt'].to(pytorch_device)

        if flag_aux_exist:
            feat_aux = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in frame_aux['pt_feat']]
            grid_aux = [torch.from_numpy(i).to(pytorch_device) for i in frame_aux['grid_ind']]
        else:
            print("The pointer is at the edge of the sequence: " + str(idx_test))
        
        #########################################################################################################################################
        ##################################################### Pseudo GT generation ##############################################################
        #########################################################################################################################################       

        if args.use_los:
            print("Generate occupancy pgt by checking the line of sight")
            voxel_los_adapt = 255*torch.ones([1,256,256,32],dtype=torch.long).to(pytorch_device) # 0:empty, 1:occupied, 2:LoS, 255:ignore
            voxel_los_cont = 255*torch.ones([1,256,256,32],dtype=torch.long).to(pytorch_device) # 0:empty, 1:occupied, 2:LoS, 255:ignore
            if flag_adapt_aux_exist:
                idx_curr_occupied = frame_curr['grid_ind'][0]
                idx_curr_occupied = np.unique(idx_curr_occupied, axis =0)
                idx_aux_occupied = frame_aux['trans_grid_ind'][adapt_aux_idx]
                idx_aux_occupied = np.unique(idx_aux_occupied, axis=0)
                idx_cat_occupied = np.concatenate([idx_curr_occupied, idx_aux_occupied],0)

                voxel_aux = np.zeros([256,256,32])
                voxel_aux[idx_aux_occupied[:,0],idx_aux_occupied[:,1],idx_aux_occupied[:,2]] = 1 # Empty:0, Occupy:1         
                voxel_curr = np.zeros([256,256,32])
                voxel_curr[idx_curr_occupied[:,0],idx_curr_occupied[:,1],idx_curr_occupied[:,2]] = 1 # Empty:0, Occupy:1
        
                voxel_aux_only = np.where(np.logical_and(voxel_curr!=1,voxel_aux==1), True, False)
                los_start = (np.floor((np.clip(frame_aux['lidar_pose'][0] , frame_curr['min_bound'][0], frame_curr['max_bound'][0]) - frame_curr['min_bound'][0]) / frame_curr['interval'][0])).astype(np.int)
                los_end = np.argwhere(voxel_aux_only==True)
                los_end = los_end[random.sample(range(los_end.shape[0]),los_end.shape[0]//8)]
                idx_los_empty = Bresenham3D(los_start,los_end,idx_cat_occupied)
                idx_los_empty = np.unique(np.array(idx_los_empty), axis=0)
                idx_los_occupied = idx_aux_occupied

                voxel_los_adapt[0, idx_los_empty[:,0],idx_los_empty[:,1],idx_los_empty[:,2]] = 0
                voxel_los_adapt[0, idx_los_occupied[:,0],idx_los_occupied[:,1],idx_los_occupied[:,2]] = 1

            if flag_cont_aux_exist:
                idx_curr_occupied = frame_curr['grid_ind'][0]
                idx_curr_occupied = np.unique(idx_curr_occupied, axis =0)
                idx_aux_occupied = frame_aux['trans_grid_ind'][cont_aux_idx]
                idx_aux_occupied = np.unique(idx_aux_occupied, axis=0)
                idx_cat_occupied = np.concatenate([idx_curr_occupied, idx_aux_occupied],0)

                voxel_aux = np.zeros([256,256,32])
                voxel_aux[idx_aux_occupied[:,0],idx_aux_occupied[:,1],idx_aux_occupied[:,2]] = 1 # Empty:0, Occupy:1         
                voxel_curr = np.zeros([256,256,32])
                voxel_curr[idx_curr_occupied[:,0],idx_curr_occupied[:,1],idx_curr_occupied[:,2]] = 1 # Empty:0, Occupy:1
        
                voxel_aux_only = np.where(np.logical_and(voxel_curr!=1,voxel_aux==1), True, False)
                los_start = (np.floor((np.clip(frame_aux['lidar_pose'][cont_aux_idx] , frame_curr['min_bound'][0], frame_curr['max_bound'][0]) - frame_curr['min_bound'][0]) / frame_curr['interval'][0])).astype(np.int)
                los_end = np.argwhere(voxel_aux_only==True)
                los_end = los_end[random.sample(range(los_end.shape[0]),los_end.shape[0]//30)]
                idx_los_empty = Bresenham3D(los_start,los_end,idx_cat_occupied)
                idx_los_empty = np.unique(np.array(idx_los_empty), axis=0)
                idx_los_occupied = idx_aux_occupied

                voxel_los_cont[0, idx_los_empty[:,0],idx_los_empty[:,1],idx_los_empty[:,2]] = 0
                voxel_los_cont[0, idx_los_occupied[:,0],idx_los_occupied[:,1],idx_los_occupied[:,2]] = 1
        else:
            print("Skip to generate los-based pgt.")

        dim_chn=20
        conv_ones = nn.Conv3d(dim_chn, dim_chn, kernel_size=(5,5,3), stride=1, padding=(2,2,1),  bias=False)
        conv_ones.weight = torch.nn.Parameter(torch.ones((dim_chn,dim_chn,5,5,3)))
        conv_ones.weight.requires_grad=False
        conv_ones.cuda()
        proposure_idx=None

        if args.use_pgt:
            print("Generate class pgt according to entropy-based confidence.")        
            model_baseline.eval()
            voxel_pgt_aux_cont = 255*torch.ones([1,256,256,32]).type(torch.LongTensor).to(pytorch_device)
            voxel_pgt_aux_adapt = 255*torch.ones([1,256,256,32]).type(torch.LongTensor).to(pytorch_device)
            voxel_pgt_curr = 255*torch.ones([1,256,256,32]).type(torch.LongTensor).to(pytorch_device)
            voxel_pgt_all_adapt = 255*torch.ones([1,256,256,32]).type(torch.LongTensor).to(pytorch_device)
            voxel_pgt_all_cont = 255*torch.ones([1,256,256,32]).type(torch.LongTensor).to(pytorch_device)
            
            with torch.no_grad():

                ### Current PGT
                print("- Generate class pgt using current scan.")  
                pred_logit_curr = model_baseline(feat_curr, grid_curr, val_batch_size, frame_curr['grid_ind'], use_tta=False)

                pred_cls_curr = torch.argmax(pred_logit_curr, 1).type(torch.LongTensor).to(pytorch_device)
                mask_pred_curr_zeroforced = (torch.sum(pred_logit_curr, dim=1)==0) # Forced to be zero during dense-to-sparse (refer to spconv)
                
                conf_curr = softmax_entropy(pred_logit_curr)
                conf_curr[mask_pred_curr_zeroforced] = 0
                conf_curr = 1 - conf_curr/torch.max(conf_curr)
                conf_curr[mask_pred_curr_zeroforced] = -1
                
                cur_middle_rfield=conv_ones(pred_logit_curr)
                cur_middle_field = (torch.sum(cur_middle_rfield, dim=1)!=0)
                cur_empty_field=(torch.sum(pred_logit_curr, dim=1)==0)
                proposure_idx=torch.where(cur_middle_field*cur_empty_field,1,0)
                proposure_idx=proposure_idx.squeeze().nonzero()
                
                mask_reliable_curr_occupied = torch.logical_and(conf_curr>args.th_pgt_occupied, pred_cls_curr!=0)
                mask_reliable_curr_empty = torch.logical_and(conf_curr>args.th_pgt_empty, pred_cls_curr==0)
                mask_reliable_curr = torch.logical_or(mask_reliable_curr_empty, mask_reliable_curr_occupied)

                voxel_pgt_curr[mask_reliable_curr] = pred_cls_curr[mask_reliable_curr]

                vis_voxel_pgt_curr = voxel_pgt_curr[0].cpu().detach().numpy()
                vis_voxel_pgt_curr = np.apply_along_axis(extract_bev_for_vis_dual, 2, vis_voxel_pgt_curr, ignore_idx=(0,255))
                vis_voxel_pgt_curr[vis_voxel_pgt_curr==255] = 20
                vis_voxel_pgt_curr = PALLETE[vis_voxel_pgt_curr]

                ### Aux PGT
                if flag_adapt_aux_exist:
                    print("- Generate class pgt for adapt using auxiliary scan.")  

                    vox_grid_to = frame_aux['vox_grid_to'][adapt_aux_idx].astype(np.int)
                    vox_grid_from = frame_aux['vox_grid_from'][adapt_aux_idx].astype(np.int)
                    
                    pred_logit_aux = model_baseline([feat_aux[adapt_aux_idx]], [grid_aux[adapt_aux_idx]], val_batch_size, [frame_aux['grid_ind'][adapt_aux_idx]], use_tta=False)
                    pred_cls_aux = torch.argmax(pred_logit_aux, 1).type(torch.LongTensor).to(pytorch_device)
                    mask_pred_aux_zeroforced = (torch.sum(pred_logit_aux, dim=1)==0) # Forced to be zero during dense-to-sparse (refer to spconv)
                    
                    conf_aux = softmax_entropy(pred_logit_aux)
                    conf_aux[mask_pred_aux_zeroforced] = 0
                    conf_aux = 1 - conf_aux/torch.max(conf_aux)
                    conf_aux[mask_pred_aux_zeroforced] = -1

                    mask_reliable_aux_occupied = torch.logical_and(conf_aux>args.th_pgt_occupied, pred_cls_aux!=0)
                    mask_reliable_aux_empty = torch.logical_and(conf_aux>args.th_pgt_empty, pred_cls_aux==0)
                    mask_reliable_aux = torch.logical_or(mask_reliable_aux_occupied, mask_reliable_aux_empty)
                    
                    pred_cls_aux[~mask_reliable_aux] = 255
                    pred_cls_aux[mask_pred_aux_zeroforced] = 255
                    voxel_pgt_aux_adapt[0, vox_grid_to[:, 0], vox_grid_to[:, 1], vox_grid_to[:, 2]] = pred_cls_aux[0, vox_grid_from[:, 0], vox_grid_from[:, 1], vox_grid_from[:, 2]]
                
                else:
                    vis_voxel_pgt_aux_adapt = None
                    print("- Skip to make aux pgt for adapt.")

                vis_voxel_pgt_aux_adapt = voxel_pgt_aux_adapt[0].cpu().detach().numpy()
                vis_voxel_pgt_aux_adapt = np.apply_along_axis(extract_bev_for_vis_dual, 2, vis_voxel_pgt_aux_adapt, ignore_idx=(0,255))
                vis_voxel_pgt_aux_adapt[vis_voxel_pgt_aux_adapt==255] = 20
                vis_voxel_pgt_aux_adapt = PALLETE[vis_voxel_pgt_aux_adapt]

                if flag_cont_aux_exist:
                    
                    print("- Generate class pgt for cont using auxiliary scan.")  

                    vox_grid_to = frame_aux['vox_grid_to'][cont_aux_idx].astype(np.int)
                    vox_grid_from = frame_aux['vox_grid_from'][cont_aux_idx].astype(np.int)
                    
                    pred_logit_aux = model_baseline([feat_aux[cont_aux_idx]], [grid_aux[cont_aux_idx]], val_batch_size, [frame_aux['grid_ind'][cont_aux_idx]], use_tta=False)
                    pred_cls_aux_cont = torch.argmax(pred_logit_aux, 1).type(torch.LongTensor).to(pytorch_device)
                    mask_pred_aux_zeroforced = (torch.sum(pred_logit_aux, dim=1)==0) # Forced to be zero during dense-to-sparse (refer to spconv)
                    
                    conf_aux = softmax_entropy(pred_logit_aux)
                    conf_aux[mask_pred_aux_zeroforced] = 0
                    conf_aux = 1 - conf_aux/torch.max(conf_aux)
                    conf_aux[mask_pred_aux_zeroforced] = -1

                    mask_reliable_aux_occupied = torch.logical_and(conf_aux>args.th_pgt_occupied, pred_cls_aux_cont!=0)
                    mask_reliable_aux_empty = torch.logical_and(conf_aux>args.th_pgt_empty, pred_cls_aux_cont==0)
                    mask_reliable_aux = torch.logical_or(mask_reliable_aux_occupied, mask_reliable_aux_empty)
                    
                    pred_cls_aux_cont[~mask_reliable_aux] = 255
                    pred_cls_aux_cont[mask_pred_aux_zeroforced] = 255
                    voxel_pgt_aux_cont[0, vox_grid_to[:, 0], vox_grid_to[:, 1], vox_grid_to[:, 2]] = pred_cls_aux_cont[0, vox_grid_from[:, 0], vox_grid_from[:, 1], vox_grid_from[:, 2]]
                
                else:
                    vis_voxel_pgt_aux_cont = None
                    print("- Skip to make aux pgt for cont.")

                vis_voxel_pgt_aux_cont = voxel_pgt_aux_cont[0].cpu().detach().numpy()
                vis_voxel_pgt_aux_cont = np.apply_along_axis(extract_bev_for_vis_dual, 2, vis_voxel_pgt_aux_cont, ignore_idx=(0,255))
                vis_voxel_pgt_aux_cont[vis_voxel_pgt_aux_cont==255] = 20
                vis_voxel_pgt_aux_cont = PALLETE[vis_voxel_pgt_aux_cont]


                ### Aggregation of current PGT and aux PGT for adapt
                voxel_pgt_all_adapt = voxel_pgt_curr.clone()
                mask_temp = (voxel_pgt_all_adapt==255)*(voxel_pgt_aux_adapt!=255)
                voxel_pgt_all_adapt[mask_temp] = voxel_pgt_aux_adapt[mask_temp]
                mask_temp = (voxel_pgt_curr!=255)*(voxel_pgt_aux_adapt!=255)*(voxel_pgt_curr!=voxel_pgt_aux_adapt)
                voxel_pgt_all_adapt[mask_temp] = 255
                ### Aggregation of current PGT and aux PGT for cont
                voxel_pgt_all_cont = voxel_pgt_curr.clone()
                mask_temp = (voxel_pgt_all_cont==255)*(voxel_pgt_aux_cont!=255)
                voxel_pgt_all_cont[mask_temp] = voxel_pgt_aux_cont[mask_temp]
                mask_temp = (voxel_pgt_curr!=255)*(voxel_pgt_aux_cont!=255)*(voxel_pgt_curr!=voxel_pgt_aux_cont)
                voxel_pgt_all_cont[mask_temp] = 255  
                
                
                vis_voxel_pgt = voxel_pgt_all_adapt[0].cpu().detach().numpy()
                vis_voxel_pgt = np.apply_along_axis(extract_bev_for_vis_dual, 2, vis_voxel_pgt, ignore_idx=(0,255))
                vis_voxel_pgt[vis_voxel_pgt==255] = 20
                vis_voxel_pgt = PALLETE[vis_voxel_pgt]

        else:
            print("Skip to generate class pgt.")

        #########################################################################################################################################
        ########################################################## Scan-wise Adaptation #########################################################
        #########################################################################################################################################
        
        if args.do_adapt:

            print("Do scan-wise adaptation.")
            print("- From baseline model")
            model_adapt = copy.deepcopy(model_baseline)

            param_to_update_adapt = []
            
            print("- Scan-wise adapt segmentation module")
            for tn in module_names_seg:
                param_to_update_adapt += list(getattr(model_adapt.cylinder_3d_spconv_seg, tn).parameters())
                
            print("- Scan-wise adapt final logit layer")
            for tn in module_names_logit:
                param_to_update_adapt += list(getattr(model_adapt.cylinder_3d_spconv_seg, tn).parameters())
                
            optimizer_adapt = optim.Adam(param_to_update_adapt, lr=args.adapt_lr)
            
            model_adapt.train()

            # Partial freeze
            for name, param in model_adapt.named_parameters():
                #freeze_mlp
                if name.split('.')[0] in module_names_mlp:
                    param.requires_grad = False
                #freeze_comp:
                if any(mona in name.split('.')[1] for mona in module_names_comp):
                    param.requires_grad = False
                
            for idx_adapt in range(args.adapt_iter):
                
                logit = model_adapt(feat_curr, grid_curr, val_batch_size, frame_curr['grid_ind'], use_tta=False) # (B,C,x,y,z)
                loss_adapt_occ_ce = 0
                loss_adapt_occ_lovasz = 0
                loss_adapt_pgt_ce = 0
                loss_adapt_pgt_lovasz = 0

                if args.use_los and flag_adapt_aux_exist:
                    logit_empty = logit[:,:1,:,:,:]
                    logit_occupied = logit[:,1:,:,:,:].max(dim=1, keepdim=True)[0]
                    logit_comp = torch.cat((logit_empty, logit_occupied), dim=1) # (B,2,x,y,z)
                    loss_adapt_occ_ce += loss_fn_ce_binary(logit_comp, voxel_los_adapt)
                    loss_adapt_occ_lovasz += loss_fn_lovasz_binary(F.softmax(logit_comp), voxel_los_adapt, ignore=255)
                    
                if args.use_pgt:
                    loss_adapt_pgt_ce += loss_fn_ce(logit, voxel_pgt_all_adapt)
                    loss_adapt_pgt_lovasz += loss_fn_lovasz(F.softmax(logit), voxel_pgt_all_adapt, ignore=255)

                optimizer_adapt.zero_grad()
                loss_adapt = args.weight_adapt_occ_ce*loss_adapt_occ_ce + args.weight_adapt_occ_lovasz*loss_adapt_occ_lovasz \
                            + args.weight_adapt_pgt_ce*loss_adapt_pgt_ce + args.weight_adapt_pgt_lovasz*loss_adapt_pgt_lovasz
                if loss_adapt!=0 and not torch.isnan(loss_adapt):
                    loss_adapt.backward()
                    optimizer_adapt.step()
                else:
                    print('Loss is zero or NaN! Skip adapt optimization.')

                # plot adapt losses
                for loss_name in loss_adapt_names:
                    loss_now = locals()[loss_name]
                    writer.add_scalar("loss_adapt/"+loss_name, loss_now, global_step=args.adapt_iter*idx_test+idx_adapt)

        
        else:
            print("Skip scan-wise adaptation.")

        #########################################################################################################################################
        ############################################################## Eval phase ###############################################################
        #########################################################################################################################################

        model_cont.eval()
        model_adapt.eval()
        with torch.no_grad():
            pred_logit = model_cont(feat_curr, grid_curr, val_batch_size, frame_curr['grid_ind'], use_tta=False, extraction=proposure_idx)

            pred = torch.argmax(pred_logit, dim=1)

            pred_logit_bs = model_adapt(feat_curr, grid_curr, val_batch_size, frame_curr['grid_ind'], use_tta=False)
            pred_bs = torch.argmax(pred_logit_bs, dim=1)

            mask_pred_=(pred==9)+(pred==10)+(pred==11)+(pred==12)+(pred==13)+(pred==15)+(pred==16)+(pred==17)
            pred[~mask_pred_]=pred_bs[~mask_pred_]
                
            pred = pred.cpu().detach().numpy()
            pred = np.squeeze(pred)
            pred = pred.astype(np.uint32)
            pred = pred.reshape((-1))

            ### save prediction after remapping
            pred_remapped = remapping(pred, remap=remap_first)
            name_velodyne = test_pt_dataset.im_idx[frame_curr['index'][0]]
            save_pred(pred_remapped, name_velodyne, exp_path)

            ### Evaluation for current step
            name_invalid = name_velodyne.replace('velodyne', 'voxels')[:-3]+'invalid'
            name_label = name_invalid.replace('invalid','label')
            invalid_voxels = unpack(np.fromfile(name_invalid, dtype=np.uint8)) # Binary, (256x256x32)
            gt_label = np.fromfile(name_label, dtype=np.uint16)

            pred_remapped = remap_second[pred_remapped]
            gt_label = remap_second[gt_label]            
            mask = get_eval_mask(gt_label, invalid_voxels)
            pred_remapped = pred_remapped[mask]
            gt_label = gt_label[mask]

            evaluator_current_step = iouEval(num_class, [])
            evaluator_current_step.addBatch(pred_remapped, gt_label)
            evaluator_all.addBatch(pred_remapped, gt_label)
            
            ### Plot            
            miou_baseline = float(baseline_performance.readline())
            _, class_ious = evaluator_current_step.getIoU()
            miou = class_ious[1:].mean()
            writer.add_scalar('0_metrics/miou', miou, global_step=idx_test)
            writer.add_scalar('0_metrics_diff/miou', round(miou,6)-miou_baseline, global_step=idx_test)
            for cidx in range(20):
                class_name_ = CATS[cidx]
                iou_baseline_ = float(baseline_performance.readline())
                iou_ = class_ious[cidx]
                writer.add_scalar('ious/'+str(cidx).zfill(2)+'_'+class_name_, iou_, global_step=idx_test)
                writer.add_scalar('ious_diff/'+str(cidx).zfill(2)+'_'+class_name_, round(iou_,6)-iou_baseline_, global_step=idx_test)
            
            conf = evaluator_current_step.get_confusion()
            precision = np.sum(conf[1:,1:]) / (np.sum(conf[1:,:]) + epsilon)
            recall = np.sum(conf[1:,1:]) / (np.sum(conf[:,1:]) + epsilon)
            iou_comp = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0,0])

            conf_pd = pd.DataFrame(conf / np.sum(conf, axis=1)[:, None], index=[i for i in CATS], columns=[i for i in CATS])
            plt.figure(figsize=(18, 7), dpi=100)
            vis_conf = sn.heatmap(conf_pd, annot=True).get_figure()
            writer.add_figure("z_conf_all", vis_conf, idx_test)
            
            conf_binary = np.array([[conf[0,0],np.sum(conf[0,1:])],[np.sum(conf[1:,0]),np.sum(conf[1:,1:])]])
            conf_binary_pd = pd.DataFrame(conf_binary / np.sum(conf_binary, axis=1)[:, None], index=[i for i in ['empty','occupied']], columns=[i for i in ['empty','occupied']])
            plt.figure(figsize=(5, 3), dpi=100)
            vis_conf_binary = sn.heatmap(conf_binary_pd, annot=True).get_figure()
            writer.add_figure("z_conf_binary", vis_conf_binary, idx_test)

            writer.add_scalar('0_metrics/prec', precision, global_step=idx_test)
            writer.add_scalar('0_metrics/recall', recall, global_step=idx_test)
            writer.add_scalar('0_metrics/iou_comp', iou_comp, global_step=idx_test)
            
            precision_baseline = float(baseline_performance.readline())
            recall_baseline = float(baseline_performance.readline())
            iou_comp_baseline = float(baseline_performance.readline())

            writer.add_scalar('0_metrics_diff/prec', round(precision,6)-precision_baseline, global_step=idx_test)
            writer.add_scalar('0_metrics_diff/recall', round(recall,6)-recall_baseline, global_step=idx_test)
            writer.add_scalar('0_metrics_diff/iou_comp', round(iou_comp,6)-iou_comp_baseline, global_step=idx_test)

            ### Visualize
            invalid_voxels = invalid_voxels.reshape(256,256,32)

            pred = pred.reshape(256,256,32)
            pred[invalid_voxels==1] = 255
            pred_bs = np.fromfile(baseline_prediction_paths[idx_test], dtype=np.uint16)
            pred_bs = MAP_ARRAY[pred_bs].reshape(256,256,32)
            pred_bs[invalid_voxels==1] = 255
            gt = gt_curr[0].cpu().detach().numpy()
            gt[invalid_voxels==1] = 255

            vis_pred = np.apply_along_axis(extract_bev_for_vis, 2, pred,  ignore_idx=0)
            vis_pred_bs = np.apply_along_axis(extract_bev_for_vis, 2, pred_bs,  ignore_idx=0)
            vis_gt = np.apply_along_axis(extract_bev_for_vis, 2, gt, ignore_idx=0)

            vis_pred[vis_pred==255] = 20
            vis_pred_bs[vis_pred_bs==255] = 20
            vis_gt[vis_gt==255] = 20

            vis_pred = PALLETE[vis_pred]
            vis_pred_bs = PALLETE[vis_pred_bs]
            vis_gt = PALLETE[vis_gt]

            sep_vert = np.ones_like(vis_gt[:,:2,:])*125       
            vis_all = np.concatenate((vis_pred_bs, sep_vert, vis_pred, sep_vert, vis_gt), axis=1)

            if args.use_los:
                current_idx_occupied = torch.unique(grid_curr[0], dim=0)
                future_voxel_ = voxel_los_adapt.clone()
                future_voxel_[0, current_idx_occupied[:,0],current_idx_occupied[:,1],current_idx_occupied[:,2]] = 2 # Green
                vis_occupied = np.apply_along_axis(extract_bev_for_vis, 2, future_voxel_[0].cpu().detach().numpy(), ignore_idx=255)
                vis_occupied[vis_occupied==255] = 3
                vis_occupied = PALLETE_BINARY[vis_occupied]
                vis_all = np.concatenate((vis_occupied, sep_vert, vis_all), axis=1)
            
            if args.use_pgt:
                if args.use_los:
                    vis_down = np.concatenate((vis_voxel_pgt_aux_adapt, sep_vert, vis_voxel_pgt_curr, sep_vert, vis_voxel_pgt, sep_vert, vis_gt), axis=1)
                    sep_hori = np.ones_like(vis_all[:2,:,:])*125
                    vis_all = np.concatenate((vis_all, sep_hori, vis_down), axis=0)
                else:
                    vis_down = np.concatenate((vis_voxel_pgt_aux_adapt, sep_vert, vis_voxel_pgt_curr, sep_vert, vis_voxel_pgt), axis=1)
                    sep_hori = np.ones_like(vis_all[:2,:,:])*125
                    vis_all = np.concatenate((vis_all, sep_hori, vis_down), axis=0)


            writer.add_image('vis', np.transpose(vis_all,(2,0,1)), global_step=idx_test)

            ### Cumulative eval
            _, class_ious = evaluator_all.getIoU()
            miou = class_ious[1:].mean()
            conf = evaluator_all.get_confusion()
            precision = np.sum(conf[1:,1:]) / (np.sum(conf[1:,:]) + epsilon)
            recall = np.sum(conf[1:,1:]) / (np.sum(conf[:,1:]) + epsilon)
            iou_comp = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0,0])

            writer.add_scalar('0_final_metrics/miou', miou, global_step=idx_test)
            writer.add_scalar('0_final_metrics/prec', precision, global_step=idx_test)
            writer.add_scalar('0_final_metrics/recall', recall, global_step=idx_test)
            writer.add_scalar('0_final_metrics/iou_comp', iou_comp, global_step=idx_test)

            for cidx in range(20):
                class_name_ = CATS[cidx]
                iou_ = class_ious[cidx]
                writer.add_scalar('final_ious/'+str(cidx).zfill(2)+'_'+class_name_, iou_, global_step=idx_test)
                
        #########################################################################################################################################
        ######################################################## Continual TTA phase ############################################################
        #########################################################################################################################################
        
        if args.do_cont:

            print("Do continual adaptation.")

            model_cont.train()
            # Partial freeze
            for name, param in model_cont.named_parameters():
                #freeze_mlp
                if name.split('.')[0] in module_names_mlp:
                    param.requires_grad = False
                #freeze_comp
                if any(mona in name.split('.')[1] for mona in module_names_comp):
                    param.requires_grad = False
                #freeze_logit
                if any(mona in name.split('.')[1] for mona in module_names_logit):
                    param.requires_grad = False
                        
            # Continual TTA loop
            for idx_cont in range(args.cont_iter):

                # Random masking
                mask_size=4
                mask_ratio=0.1
                upsample = nn.Upsample(scale_factor=mask_size, mode='nearest')
                mask_voxel=torch.zeros(int(256/mask_size),int(256/mask_size),int(32/mask_size))
                mask_voxel=mask_voxel.reshape(-1)
                rand_idx=torch.randperm(mask_voxel.shape[0])
                mask_number=int(rand_idx.shape[0]*mask_ratio)
                mask_patch=rand_idx[:mask_number]
                mask_voxel[mask_patch[:]]=1
                mask_voxel=mask_voxel.reshape(int(256/mask_size),int(256/mask_size),int(32/mask_size))
                mask_voxel=upsample(mask_voxel.unsqueeze(0).unsqueeze(0)).squeeze()
                voxel_curr_grid = torch.zeros([256,256,32])
                voxel_curr_grid[grid_curr[0][:,0],grid_curr[0][:,1],grid_curr[0][:,2]] = 1
                masked_voxel=voxel_curr_grid*mask_voxel
                masked_coords=masked_voxel.nonzero()
                
                matches = torch.nonzero((grid_curr[0][:, None] == masked_coords.cuda()).all(-1), as_tuple=True)
                masked_idx  = matches[0]
                all_indices = torch.arange(grid_curr[0].shape[0])
                retain_idx = all_indices[~torch.isin(all_indices.cuda(), masked_idx)]
                feat_curr_retain = [feat_curr[0][retain_idx]]
                grid_curr_retain = [grid_curr[0][retain_idx]]
                frame_curr_grid_ind_retain = [frame_curr['grid_ind'][0][retain_idx.cpu().detach().numpy()]]
                grid_curr_masked = grid_curr[0][masked_idx]
                grid_curr_masked=torch.cat([grid_curr_masked,proposure_idx],0)
                logit = model_cont(feat_curr_retain, grid_curr_retain, val_batch_size, frame_curr_grid_ind_retain, use_tta=False, extraction=grid_curr_masked) # (B,C,x,y,z)
                pred = torch.argmax(logit, dim=1) # (B,x,y,z)
                
                loss_cont_occ_ce = 0
                loss_cont_occ_lovasz = 0
                loss_cont_pgt_ce = 0
                loss_cont_pgt_lovasz = 0

                if args.use_los and flag_cont_aux_exist:
                    logit_empty = logit[:,:1,:,:,:]
                    logit_occupied = logit[:,1:,:,:,:].max(dim=1, keepdim=True)[0]
                    logit_comp = torch.cat((logit_empty, logit_occupied), dim=1) # (B,2,x,y,z)
                    loss_cont_occ_ce += loss_fn_ce_binary(logit_comp, voxel_los_cont)
                    loss_cont_occ_lovasz += loss_fn_lovasz_binary(F.softmax(logit_comp), voxel_los_cont, ignore=255)

                if args.use_pgt:
                    loss_cont_pgt_ce += loss_fn_ce(logit, voxel_pgt_all_cont)
                    loss_cont_pgt_lovasz += loss_fn_lovasz(F.softmax(logit), voxel_pgt_all_cont, ignore=255)                    

                optimizer_cont.zero_grad()
                loss_cont = args.weight_cont_occ_ce*loss_cont_occ_ce + args.weight_cont_occ_lovasz*loss_cont_occ_lovasz \
                            + args.weight_cont_pgt_ce*loss_cont_pgt_ce + args.weight_cont_pgt_lovasz*loss_cont_pgt_lovasz
                if loss_cont!=0 and not torch.isnan(loss_cont):
                    loss_cont.backward()
                    optimizer_cont.step()
                else:
                    print('Loss is zero or NaN! Skip cont optimization.')

                # plot cont tta losses
                for loss_name in loss_cont_names:
                    loss_now = locals()[loss_name]
                    writer.add_scalar("loss_cont/"+loss_name, loss_now, global_step=args.cont_iter*idx_test+idx_cont)
        else:
            print("Skip continual adaptation.")

    baseline_performance.close()
    _, class_ious = evaluator_all.getIoU()
    miou = class_ious[1:].mean()

    conf = evaluator_all.get_confusion()
    precision = np.sum(conf[1:,1:]) / (np.sum(conf[1:,:]) + epsilon)
    recall = np.sum(conf[1:,1:]) / (np.sum(conf[:,1:]) + epsilon)
    iou_comp = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0,0])

    print(class_ious)
    print(iou_comp)
    print(miou)
    print(precision)
    print(recall)
    
    eval_file = exp_path + '/eval.txt'
    with open(eval_file, 'w') as log:
        log.write(str(class_ious)+'\n')
        log.write(str(iou_comp)+'\n')
        log.write(str(miou)+'\n')
        log.write(str(precision)+'\n')
        log.write(str(recall)+'\n')


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')

    # Sources
    parser.add_argument('--talos_root', default='/home/user/talos-official/', type=str)
    parser.add_argument('--config_path', default='config/semantickitti-tta-val.yaml')
    parser.add_argument('--baseline_perf_txt', default='baseline_performance.txt', type=str)
    parser.add_argument('--baseline_preds', default='experiments/baseline/sequences/08/predictions', type=str)
    
    # Experiment
    parser.add_argument('--name', default='debug')
    parser.add_argument('--ang', action='store_true', help='Auto Name Generator')
    parser.add_argument('--loader', default='data_builder', type=str)
    parser.add_argument('--stride', default='[-5,5]', type=str)
    parser.add_argument('--do_cont', action='store_true')
    parser.add_argument('--do_adapt', action='store_true')
    
    # Attributes
    parser.add_argument('--use_los', action='store_true')
    parser.add_argument('--use_pgt', action='store_true')
    parser.add_argument('--th_pgt_occupied', default=0.75, type=float)
    parser.add_argument('--th_pgt_empty', default=0.999, type=float)
    
    # Optimization (cont)
    parser.add_argument('--cont_lr', default=3e-05, type=float)
    parser.add_argument('--cont_iter', default=1, type=int)
    parser.add_argument('--weight_cont_occ_ce', default=1, type=float)
    parser.add_argument('--weight_cont_occ_lovasz', default=1, type=float)
    parser.add_argument('--weight_cont_pgt_ce', default=1, type=float)
    parser.add_argument('--weight_cont_pgt_lovasz', default=1, type=float)

    # Optimization (adapt)
    parser.add_argument('--adapt_lr', default=0.0003, type=float)
    parser.add_argument('--adapt_iter', default=3, type=int)
    parser.add_argument('--weight_adapt_occ_ce', default=1, type=float)
    parser.add_argument('--weight_adapt_occ_lovasz', default=1, type=float)
    parser.add_argument('--weight_adapt_pgt_ce', default=1, type=float)
    parser.add_argument('--weight_adapt_pgt_lovasz', default=1, type=float)

    args = parser.parse_args()

    args.baseline_perf_txt = args.talos_root+args.baseline_perf_txt
    args.baseline_preds = args.talos_root+args.baseline_preds

    print(' '.join(sys.argv))
    print(args)
    print('#####')
    print('Stride: '+str(args.stride))
    print('#####')

    if args.ang:

        args.name = ''
        args.name += '_stride'+str(args.stride)

        if args.use_los:
            args.name += '_los'
        if args.use_pgt:
            args.name += '_pgt'
            if args.th_pgt_occupied != 0.75:
                args.name += '_thocc'+str(args.th_pgt_occupied)
            if args.th_pgt_empty != 0.999:
                args.name += '_themp'+str(args.th_pgt_empty)

        if args.do_cont:
            args.name += '_cont'
            args.name += '_clr'+str(args.cont_lr)
            args.name += '_cit'+str(args.cont_iter)
            
        if args.do_adapt:
            args.name += '_adapt'
            args.name += '_alr'+str(args.adapt_lr)
            args.name += '_ait'+str(args.adapt_iter)
            


        print(args.name)
    
    main(args)
