import copy
import numpy as np
import torch
import torch.nn as nn
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import loss_utils
from typing import List


class CenterHeadSingle(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        self.qat_flag = kwargs.get('quantize', None)
        
        self.out_normlize = self.model_cfg.get('OUT_NORMLIZE', False)
        self.debug_mode = False
        self.shared_channel = self.model_cfg.SHARED_CONV_CHANNEL

        self.hm_loss_type = self.model_cfg.get('HM_LOSS', 'FOCAL_LOSS')
        self.reg_loss_type = self.model_cfg.get('REG_LOSS', "SMOOTH_L1_LOSS")
        self.iou_loss_type = self.model_cfg.get('IOU_LOSS', "SMOOTH_L1_LOSS")

        self.iou_aware_flag = self.model_cfg.get('WITH_IOU_AWARE_LOSS', True)
        self.iou_aware_weight = self.model_cfg.get('IOU_AWARE_WEIGHT', [0.5])
        self.hm_head_c = self.model_cfg.get('HM_HEAD_CHANNEL', 2)
        self.reg_head_c = self.model_cfg.get('REG_HEAD_CHANNEL', 9)
        self.export_onnx = False

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes in [len(self.class_names), len(self.class_names) - 1], f'class_names_each_head={self.class_names_each_head}'

        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_channels, self.shared_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.shared_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.shared_channel, self.shared_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.shared_channel),
            nn.ReLU(inplace=True),
        )

        self.hm_head = nn.Conv2d(input_channels, self.hm_head_c, kernel_size=3, stride=1, padding=1, bias=True)
        self.reg_head = nn.Conv2d(input_channels, self.reg_head_c, kernel_size=3, stride=1, padding=1, bias=True)
        if self.qat_flag:
            self.dequant = torch.quantization.DeQuantStub()

        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.data_dict_info = {}
        self.hm_loss = loss_utils.FocalLossCenterNet()
        self.loc_loss = loss_utils.RegLossCenterNet()
        self.iou_loss = loss_utils.RegLossCenterNet()
        self.init_weight()

    def init_weight(self):
        init_bias = -2.19
        nn.init.constant_(self.hm_head.bias, init_bias)
        nn.init.normal_(self.reg_head.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_head.bias, 0)

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            num_classes: 2
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]
            feature_map_stride: 4
            num_max_objs: M (100)
        Returns:
            heatmap: (B, 2, H, W)
            heatmap_mask: (B, 2, H, W)
            ret_boxes: (B, 5*M, 8) [cx, cy, cz, dx, dy, dz, cos, sin]
            inds: (B, 5*M)
            mask: (B, 5*M)
        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        heatmap_mask = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])

        ret_boxes = gt_boxes.new_zeros((5*num_max_objs, gt_boxes.shape[-1]))
        inds = gt_boxes.new_zeros(5*num_max_objs).long()
        mask = gt_boxes.new_zeros(5*num_max_objs).long()       
        

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()

        dx, dy, dz, ry, idx = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5], gt_boxes[:, 6], gt_boxes[:, 7].long()
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        gt_box_int = torch.cat([coord_x[:, None], coord_y[:, None], dx[:, None], dy[:, None], ry[:, None], idx[:, None]], dim=1)
        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue
            
            cur_class_id = (gt_boxes[k, -1] - 1).long()
            if cur_class_id >= 0:
                centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())
                heatmap_mask[cur_class_id][heatmap[cur_class_id]>0] = 1
                mask[k*5:k*5+5] = 1
            else:
                mask[k*5:k*5+5] = -1

            inds[k*5] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            ret_boxes[k*5:k*5+5, 0:2] = center[k] - center_int[k].float()
            ret_boxes[k*5:k*5+5, 2] = z[k]
            ret_boxes[k*5:k*5+5, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k*5:k*5+5, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k*5:k*5+5, 7] = torch.sin(gt_boxes[k, 6])

            temp_l = gt_boxes[k,3]
            temp_w = gt_boxes[k,4]
            
            if (self.voxel_size[0] * feature_map_stride > temp_w/4):
                mask[k*5+1:k*5+3] = 0
            else:
                if (center_int[k, 0] -1 >=0):
                    inds[k*5+1] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0] -1
                    ret_boxes[k*5+1, 0] += 1
                else:
                    mask[k*5+1] = 0

                if (center_int[k, 0] + 1 < feature_map_size[0]):
                    inds[k*5+2] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0] + 1
                    ret_boxes[k*5+2, 0] -= 1
                else:
                    mask[k*5+2] = 0

            if (self.voxel_size[1] * feature_map_stride > temp_l/4):
                mask[k*5+3:k*5+5] = 0
            else:
                if (center_int[k, 1] -1 >=0):
                    inds[k*5+3] = (center_int[k, 1] - 1) * feature_map_size[0] + center_int[k, 0]
                    ret_boxes[k*5+3, 1] += 1
                else:
                    mask[k*5+3] = 0

                if (center_int[k, 1] +1 < feature_map_size[1]):
                    inds[k*5+4] = (center_int[k, 1] +1 ) * feature_map_size[0] + center_int[k, 0]
                    ret_boxes[k*5+4, 1] -= 1
                else:
                    mask[k*5+4] = 0

            if gt_boxes.shape[1] > 8:
                ret_boxes[k*5:k*5+5, 8:] = gt_boxes[k, 7:-1]

        gt_box_int_ignore = gt_box_int[gt_box_int[:, -1] == 0]

        # torch.cuda.synchronize()
        # import time
        # start = time.time()
        if gt_box_int_ignore.size(0) > 0:
            for i in range(num_classes):
                for k in range(gt_box_int_ignore.size(0)):
                    temp_gt_box = gt_box_int_ignore[k]
                    cur_class_id = (gt_box_int_ignore[k, -1] - 1).long()
                    if cur_class_id == -1:
                        heatmap_mask[i] = centernet_utils.draw_ignore_to_heatmap_mask(heatmap_mask[i], temp_gt_box, self.debug_mode)
                        heatmap_mask[i][heatmap_mask[i]>200] = -1
        # torch.cuda.synchronize()
        # end = time.time()
        # print(f"time is {(end-start)*1000}ms.")
        return heatmap, ret_boxes, inds, mask, heatmap_mask

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:
            ret_dict:
                heatmaps: (B, C, H, W)
                target_boxes: (B, M, 8)
                inds: (B, M, 8)
                masks: (B, M, 8)
                heatmap_masks: (B, C, H, W)
        """
        feature_map_size = feature_map_size[::-1]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, heatmap_masks_list = [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]

                if self.debug_mode:
                    print("cls id:", cur_gt_boxes[:, -1].long())
                    print("all_names: ", all_names)
                    print("cur_class_names: ", cur_class_names)

                gt_boxes_single_head = cur_gt_boxes
                
                heatmap, ret_boxes, inds, mask, heatmap_mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
            

                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))
                heatmap_masks_list.append(heatmap_mask.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['heatmap_masks'].append(torch.stack(heatmap_masks_list, dim=0))

        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_bst_loss(self, pred_dicts, target_dicts):

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            
            pred_hm = self.sigmoid(pred_dict['hm'])
            pred_boxes = pred_dict['reg']

            target_boxes = target_dicts['target_boxes'][idx]
            target_boxes_clone = target_boxes.clone()

            if self.iou_aware_flag:
                pred_iou = pred_boxes[:, 8:, :, :]
                pred_boxes = pred_boxes[:, :8, :, :]


            # hm loss
            hm_loss = self.hm_loss(
                pred_hm, target_dicts['heatmaps'][idx], target_dicts['heatmap_masks'][idx], mode=self.hm_loss_type.lower()
            )
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
            loss += hm_loss

            if self.out_normlize:
                target_boxes[:, :, 0:1] -= 0.5
                target_boxes[:, :, 1:2] -= 0.5
                target_boxes[:, :, 2:3] -= 1.0
                target_boxes[:, :, 3:4] -= 1.25
                target_boxes[:, :, 4:5] -= 0.6
                target_boxes[:, :, 5:6] -= 0.8
            else:
                target_boxes = target_boxes_clone
            
            # reg loss
            reg_loss_weight = pred_boxes.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])
            reg_loss = self.loc_loss(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes, weight=reg_loss_weight, mode=self.reg_loss_type.lower()
            )
            loc_loss = reg_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            loss += loc_loss
            
            if self.iou_aware_flag:
                
                batch_size = target_boxes.shape[0]
                
                if self.out_normlize:
                    pred_x = pred_boxes[:, 0:1, :, :] + 0.5
                    pred_y = pred_boxes[:, 1:2, :, :] + 0.5
                    pred_z = pred_boxes[:, 2:3, :, :] + 1.0
                    pred_l = pred_boxes[:, 3:4, :, :] + 1.25
                    pred_w = pred_boxes[:, 4:5, :, :] + 0.6
                    pred_h = pred_boxes[:, 5:6, :, :] + 0.8

                    center = torch.cat([pred_x, pred_y], dim=1)
                    center_z = pred_z
                    dim = torch.cat([pred_l, pred_w, pred_h], dim=1)

                    batch_center = center
                    batch_center_z = center_z
                    batch_dim = dim.exp()
                else:
                    batch_center = pred_boxes[:, 0:2, :, :]
                    batch_center_z = pred_boxes[:, 2:3, :, :]
                    batch_dim = pred_boxes[:, 3:6, :, :].exp()
                
                batch_rot_cos = pred_boxes[:, 6:7, :, :]
                batch_rot_sin = pred_boxes[:, 7:8, :, :]

                center_ = loss_utils._transpose_and_gather_feat(batch_center,target_dicts['inds'][idx])*self.feature_map_stride*self.voxel_size[0]
                center_z_ = loss_utils._transpose_and_gather_feat(batch_center_z,target_dicts['inds'][idx])
                dim_ = loss_utils._transpose_and_gather_feat(batch_dim,target_dicts['inds'][idx])
                cos_ = loss_utils._transpose_and_gather_feat(batch_rot_cos,target_dicts['inds'][idx])
                sin_ = loss_utils._transpose_and_gather_feat(batch_rot_sin,target_dicts['inds'][idx])
                angle_ = torch.atan2(sin_, cos_)
                final_pred_dicts = torch.cat([center_,center_z_,dim_,angle_], dim=-1)
                final_pred_dicts = final_pred_dicts.view(-1,7)
                final_pred_dicts = final_pred_dicts.detach()

                final_target = target_boxes_clone
                final_target[:,:,:2] *= self.feature_map_stride*self.voxel_size[0]
                final_target[:,:,3:6] = final_target[:,:,3:6].exp()
                final_target[:,:,6:7] = torch.atan2(final_target[:,:,7:8], final_target[:,:,6:7])
                final_target = final_target[:,:,:7].view(-1,7)

                iou_target = iou3d_nms_utils.boxes_iou3d_gpu(final_pred_dicts, final_target)
                iou_target = iou_target[range(iou_target.shape[0]),range(iou_target.shape[0])].view(batch_size,-1,1)
                iou_target = 2 * iou_target - 1
                iou_target = iou_target.detach()

                # iou aware loss
                iou_aware_loss = self.iou_loss(
                    pred_iou, target_dicts['masks'][idx], target_dicts['inds'][idx], iou_target, mode=self.iou_loss_type.lower()
                )
                iou_aware_loss = iou_aware_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight']
                loss += iou_aware_loss


            if self.debug_mode:
                print("hm_loss: ", hm_loss)
                print("loc_loss: ", loc_loss)
                print("iou_aware_loss: ", iou_aware_loss)
                print("loss sum: ", loss)

            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            
            pred_hm = self.sigmoid(pred_dict['hm'])
            pred_boxes = pred_dict['reg']

            target_boxes = target_dicts['target_boxes'][idx]
            target_boxes_clone = target_boxes.clone()
            if self.iou_aware_flag:
                pred_iou = pred_boxes[:, 8:, :, :]
                pred_boxes = pred_boxes[:, :8, :, :]

            hm_loss = self.hm_loss(
                pred_hm, target_dicts['heatmaps'][idx], target_dicts['heatmap_masks'][idx], mode=self.hm_loss_type.lower()
            )
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
            loss += hm_loss

            if self.out_normlize:
                target_boxes[:, :, 0:1] -= 0.5
                target_boxes[:, :, 1:2] -= 0.5
                target_boxes[:, :, 2:3] -= 1.0
                target_boxes[:, :, 3:4] -= 1.25
                target_boxes[:, :, 4:5] -= 0.6
                target_boxes[:, :, 5:6] -= 0.8
            else:
                target_boxes = target_boxes_clone
            
            # for i in range(8):
            #     print(i, "---", target_boxes[:, :, i].max(), target_boxes[:, :, i].min())
            # print("\n\n")

            # reg loss
            reg_loss_weight = pred_boxes.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])

            reg_loss = self.loc_loss(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes, weight=reg_loss_weight,
                mode=self.reg_loss_type.lower()
            )

            loc_loss = reg_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            loss += loc_loss
            
            if self.iou_aware_flag:
                
                batch_size = target_boxes.shape[0]
                
                if self.out_normlize:
                    pred_x = pred_boxes[:, 0:1, :, :] + 0.5
                    pred_y = pred_boxes[:, 1:2, :, :] + 0.5
                    pred_z = pred_boxes[:, 2:3, :, :] + 1.0
                    pred_l = pred_boxes[:, 3:4, :, :] + 1.25
                    pred_w = pred_boxes[:, 4:5, :, :] + 0.6
                    pred_h = pred_boxes[:, 5:6, :, :] + 0.8

                    center = torch.cat([pred_x, pred_y], dim=1)
                    center_z = pred_z
                    dim = torch.cat([pred_l, pred_w, pred_h], dim=1)

                    batch_center = center
                    batch_center_z = center_z
                    batch_dim = dim.exp()
                else:
                    batch_center = pred_boxes[:, 0:2, :, :]
                    batch_center_z = pred_boxes[:, 2:3, :, :]
                    batch_dim = pred_boxes[:, 3:6, :, :].exp()
                
                batch_rot_cos = pred_boxes[:, 6:7, :, :]
                batch_rot_sin = pred_boxes[:, 7:8, :, :]

                center_ = loss_utils._transpose_and_gather_feat(batch_center,target_dicts['inds'][idx])*self.feature_map_stride*self.voxel_size[0]
                center_z_ = loss_utils._transpose_and_gather_feat(batch_center_z,target_dicts['inds'][idx])
                dim_ = loss_utils._transpose_and_gather_feat(batch_dim,target_dicts['inds'][idx])
                cos_ = loss_utils._transpose_and_gather_feat(batch_rot_cos,target_dicts['inds'][idx])
                sin_ = loss_utils._transpose_and_gather_feat(batch_rot_sin,target_dicts['inds'][idx])
                angle_ = torch.atan2(sin_, cos_)
                final_pred_dicts = torch.cat([center_,center_z_,dim_,angle_], dim=-1)
                final_pred_dicts = final_pred_dicts.view(-1,7)
                final_pred_dicts = final_pred_dicts.detach()

                final_target = target_boxes_clone
                final_target[:,:,:2] *= self.feature_map_stride*self.voxel_size[0]
                final_target[:,:,3:6] = final_target[:,:,3:6].exp()
                final_target[:,:,6:7] = torch.atan2(final_target[:,:,7:8], final_target[:,:,6:7])
                final_target = final_target[:,:,:7].view(-1,7)

                iou_target = iou3d_nms_utils.boxes_iou3d_gpu(final_pred_dicts, final_target)
                iou_target = iou_target[range(iou_target.shape[0]),range(iou_target.shape[0])].view(batch_size,-1,1)
                iou_target = 2 * iou_target - 1
                iou_target = iou_target.detach()

                # iou aware loss
                iou_aware_loss = self.iou_loss(
                    pred_iou, target_dicts['masks'][idx], target_dicts['inds'][idx], iou_target, mode=self.iou_loss_type.lower()
                )
                iou_aware_loss = iou_aware_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['iou_weight']
                loss += iou_aware_loss


            if self.debug_mode:
                print("hm_loss: ", hm_loss)
                print("loc_loss: ", loc_loss)
                print("iou_aware_loss: ", iou_aware_loss)
                print("loss sum: ", loss)

            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()


        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]

        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_boxes = pred_dict['reg']


            if self.iou_aware_flag:
                iou_pre = (batch_boxes[:, 7:8, :, :] + 1) * 0.5
                iou_pre = torch.clamp(iou_pre, min=0, max=1.0)
                iou_aware_weight = self.iou_aware_weight
                for i, iou_w in enumerate(iou_aware_weight):
                    batch_hm[:, i:i+1, :, :] = torch.pow(batch_hm[:, i:i+1, :, :], iou_w) * \
                            torch.pow(iou_pre, 1-iou_w)
                
            if self.out_normlize:
                pred_x = batch_boxes[:, 0:1, :, :] + 0.5
                pred_y = batch_boxes[:, 1:2, :, :] + 0.5
                pred_z = batch_boxes[:, 2:3, :, :] + 1.0
                pred_l = batch_boxes[:, 3:4, :, :] + 1.25
                pred_w = batch_boxes[:, 4:5, :, :] + 0.6
                pred_h = batch_boxes[:, 5:6, :, :] + 0.8

                center = torch.cat([pred_x, pred_y], dim=1)
                center_z = pred_z
                dim = torch.cat([pred_l, pred_w, pred_h], dim=1)

                batch_center = center
                batch_center_z = center_z
                batch_dim = dim.exp()
            else:
                batch_center = batch_boxes[:, 0:2, :, :]
                batch_center_z = batch_boxes[:, 2:3, :, :]
                batch_dim = batch_boxes[:, 3:6, :, :].exp()

            batch_rot_cos = batch_boxes[:, 6:7, :, :]
            batch_rot_sin = batch_boxes[:, 7:8, :, :]

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=None,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict


    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)

        pred_dicts = []
        if self.qat_flag:
            pre_cls = self.dequant(self.hm_head(x))
            pre_box = self.dequant(self.reg_head(x))
        else:
            pre_cls = self.hm_head(x)
            pre_box = self.reg_head(x)

        pred_dict = dict()
        pred_dict["hm"] = pre_cls
        pred_dict["reg"] = pre_box
        pred_dicts.append(pred_dict)

        if self.qat_flag or self.export_onnx:
            return pred_dicts

        if self.training:

            # import time
            # start_ = time.time()
            self.forward_ret_dict['target_dicts'] = self.batch_target_dicts(batch_dict= data_dict)
            # end_ = time.time()
            # print(f"\n----------------The forward time is  {(end_-start_)*1000}ms.-------------------")


        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )
        data_dict['final_box_dicts'] = pred_dicts
        return data_dict

    def batch_target_dicts(self, batch_dict):
        """

        :param batch_dict:
        :return: batch_dict list
        """

        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds':[],
            'masks': [],
            'heatmap_masks': []
        }

        ret_dict['heatmaps'].append(batch_dict['heatmaps'][0])
        ret_dict['target_boxes'].append(batch_dict['target_boxes'][0])
        ret_dict['inds'].append(batch_dict['inds'][0])
        ret_dict['masks'].append(batch_dict['mask'][0])
        ret_dict['heatmap_masks'].append(batch_dict['heatmap_mask'][0])

        return ret_dict