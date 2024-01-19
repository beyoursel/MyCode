from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
from ...ops.roiaware_pool3d import roiaware_pool3d_utils

from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_utils
from ...utils import box_utils
from ...utils import common_utils, loss_utils


class PointSegAux(nn.Module):

    def __init__(self, model_cfg, num_class, input_channels, point_cloud_range, voxel_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.point_fc = nn.Linear(320, 64, bias=False)
        self.point_cls = nn.Linear(64, 1, bias=False)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        # self.point_reg = nn.Linear(64, 3, bias=False)

    def nearest_neighbot_interpolate(self, unknown, known, known_feats):
        dist, idx = pointnet2_utils.three_nn(unknown, known)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

        return interpolated_feats


    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coord']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        point_coords = point_coords.reshape(-1, 4)
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False
        )

        return targets_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.training:
            voxel_features = batch_dict['voxel_features']
            voxel_coords = batch_dict['voxel_coords']
            batch_size = batch_dict['batch_size']
            points_coord = torch.zeros_like(voxel_features)

            points_mean = points_coord[:, 1:]
            points_coord[:, 0] = voxel_coords[:, 0]
            points_coord[:, 1:] = voxel_features[:, :3]
            points_mean[:, 0:] = voxel_features[:, :3]
            batch_dict['point_coord'] = points_coord
            multi_scale_3d_features = batch_dict['multi_scale_3d_features']
            x_conv2 = multi_scale_3d_features['x_conv2']
            x_conv3 = multi_scale_3d_features['x_conv3']
            x_conv4 = multi_scale_3d_features['x_conv4']

            cur_coords2 = x_conv2.indices
            vx_feat2 = x_conv2.features.contiguous()
            # vx_feat2 = vx_feat2.reshape(batch_size, -1, vx_feat2.shape[-1]).permute(0, 2, 1).contiguous()
            xyz2 = common_utils.get_voxel_centers(
                cur_coords2[:, 1:4], downsample_times=2,
                voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range
            )
            # xyz2 = xyz2.reshape(batch_size, -1, 3).contiguous()

            cur_coords3 = x_conv3.indices
            vx_feat3 = x_conv3.features.contiguous()
            # vx_feat3 = vx_feat3.reshape(batch_size, -1, vx_feat3.shape[-1]).permute(0, 2, 1).contiguous()
            xyz3 = common_utils.get_voxel_centers(
                cur_coords3[:, 1:4], downsample_times=4,
                voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range
            )

            # xyz3 = xyz3.reshape(batch_size, -1, 3).contiguous()
            cur_coords4 = x_conv4.indices
            vx_feat4 = x_conv4.features.contiguous()
            # vx_feat4 = vx_feat4.reshape(batch_size, -1, vx_feat4.shape[-1]).permute(0, 2, 1).contiguous()
            xyz4 = common_utils.get_voxel_centers(
                cur_coords4[:, 1:4], downsample_times=8,
                voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range
            )
            # xyz4 = xyz4.reshape(batch_size, -1, 3).contiguous()


            point_bn_id = points_coord[:, 0]
            coor2_id = cur_coords2[:, 0]
            coor3_id = cur_coords3[:, 0]
            coor4_id = cur_coords4[:, 0]
            # points_mean = points_mean.reshape(batch_size, -1, 3).contiguous()
            point_feature_list = []
            for i in range(batch_size):
                mask1 = point_bn_id == i
                mask_2 = coor2_id == i
                mask_3 = coor3_id == i
                mask_4 = coor4_id == i
                point_per_batch = points_mean[mask1].reshape(1, -1, 3)

                vx_feat2_bn = vx_feat2[mask_2]
                vx_feat2_bn = vx_feat2_bn.reshape(1, -1, vx_feat2_bn.shape[-1]).permute(0, 2, 1).contiguous()
                xyz2_bn = xyz2[mask_2].reshape(1, -1, 3).contiguous()
                p1 = self.nearest_neighbot_interpolate(point_per_batch, xyz2_bn, vx_feat2_bn)

                vx_feat3_bn = vx_feat3[mask_3]
                vx_feat3_bn = vx_feat3_bn.reshape(1, -1, vx_feat3_bn.shape[-1]).permute(0, 2, 1).contiguous()
                xyz3_bn = xyz3[mask_3].reshape(1, -1, 3).contiguous()
                p2 = self.nearest_neighbot_interpolate(point_per_batch, xyz3_bn, vx_feat3_bn)

                vx_feat4_bn = vx_feat4[mask_4]
                vx_feat4_bn = vx_feat4_bn.reshape(1, -1, vx_feat4_bn.shape[-1]).permute(0, 2, 1).contiguous()
                xyz4_bn = xyz4[mask_4].reshape(1, -1, 3).contiguous()
                p3 = self.nearest_neighbot_interpolate(point_per_batch, xyz4_bn, vx_feat4_bn)
                # pointwise = self.point_fc(torch.cat([p1, p2, p3], dim=1).permute(0, 2, 1).contiguous())
                point_feature = torch.cat([p1, p2, p3], dim=1).squeeze(dim=0).permute(1, 0).contiguous()
                point_feature_list.append(point_feature)
            point_feature = torch.cat(point_feature_list, dim=0)
            pointwise = self.point_fc(point_feature)
            point_cls_preds = self.point_cls(pointwise)
            # point_reg = self.point_reg(pointwise)
            ret_dict = {
                'point_cls_preds': point_cls_preds,
            }
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            self.forward_ret_dict = ret_dict

        return batch_dict

    def assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None,
                             ret_box_labels=False, ret_part_labels=False,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = common_utils.rotate_points_along_z(
                    transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]
                ).view(-1, 3)
                offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                point_part_labels_single[fg_flag] = (transformed_points / gt_box_of_fg_points[:, 3:6]) + offset
                point_part_labels[bs_mask] = point_part_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'point_part_labels': point_part_labels
        }
        return targets_dict

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_type = losses_cfg.get('LOSS_REG', None)
        if reg_loss_type == 'smooth-l1':
            self.reg_loss_func = F.smooth_l1_loss
        elif reg_loss_type == 'l1':
            self.reg_loss_func = F.l1_loss
        elif reg_loss_type == 'WeightedSmoothL1Loss':
            self.reg_loss_func = loss_utils.WeightedSmoothL1Loss(
                code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
            )
        else:
            self.reg_loss_func = F.smooth_l1_loss

    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class)

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_loss_cls': point_loss_cls.item(),
            'point_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()

        point_loss = point_loss_cls
        tb_dict.update(tb_dict_1)
        return point_loss, tb_dict
