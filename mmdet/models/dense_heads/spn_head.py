import cv2
import numpy as np
import pyclipper
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init

from mmdet.core import force_fp32, multi_apply
from ..builder import HEADS, build_loss


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=has_bias
    )


def conv3x3_bn_relu(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


def vatti_clipping(contours, ratio):
    clipped_contours = []
    for contour in contours:
        contour = np.array(contour).reshape(-1, 2)
        length = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        d = area * ratio / max(length, 1)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(contour, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        clipped = pco.Execute(-d)
        for s in clipped:
            s = np.array(s).reshape(-1, 2)
            clipped_contours.append(s)
    return clipped_contours


def clip_contours(contours):
    return vatti_clipping(contours, (1 - np.power(0.4, 2)))


def unclip_contours(contours):
    return vatti_clipping(contours, -3.0)


@HEADS.register_module()
class SPNHead(nn.Module):

    def __init__(self, in_channels, feat_channels, train_cfg, test_cfg, loss_mask):
        super(SPNHead, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_mask = build_loss(loss_mask)

        self.prob = nn.Sequential(
            conv3x3_bn_relu(self.in_channels, self.feat_channels, 1),
            nn.ConvTranspose2d(self.feat_channels, self.feat_channels, 2, 2),
            nn.BatchNorm2d(self.feat_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.feat_channels, 1, 2, 2),
            #nn.Sigmoid(),
        )

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, )):
                xavier_init(m, distribution='uniform')

    def forward_single(self, feats):
        return tuple([self.prob(feats)])

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        gt_masks = kwargs['gt_masks']
        outs = self(x)
        loss_inputs = outs + (gt_masks, img_metas)
        losses = self.loss(*loss_inputs)
        assert proposal_cfg is not None

        proposal_list = self.get_bboxes(*outs, cfg=proposal_cfg, loss=losses['loss_rpn_mask'].cpu().detach().numpy())
        return losses, proposal_list

    def simple_test_rpn(self, x, img_metas):
        rpn_outs = self(x)
        proposal_list = self.get_bboxes(*rpn_outs, cfg=None)
        return proposal_list

    def get_targets(self, mask_pred, gt_masks):
        united_masks = []
        for masks in gt_masks:
            assert len(masks[0]), '????'
            final_mask = np.zeros_like(masks[0])
            clipped_contours = []
            for mask in masks:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = clip_contours(contours)
                clipped_contours.extend(contours)

            final_mask = np.squeeze(final_mask, axis=0)
            cv2.drawContours(final_mask, clipped_contours, -1, 1, -1)
            united_masks.append(final_mask)

        mask_targets_per_level = []
        for level_idx, level_pred in enumerate(mask_pred):
            mask_targets = [cv2.resize(
                mask, level_pred.shape[-2:][::-1], cv2.INTER_NEAREST) for mask in united_masks]
            mask_targets = [np.expand_dims(mask, axis=(0, 1))
                            for mask in mask_targets]
            mask_targets = [torch.tensor(mask, device=level_pred.device, dtype=level_pred.dtype)
                            for mask in mask_targets]
            mask_targets_per_level.append(torch.cat(mask_targets))

        return mask_targets_per_level

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, gt_masks, img_metas):
        mask_targets = self.get_targets(mask_pred, gt_masks)

        if 0:
            cpu_targets = [np.squeeze(mask.cpu().numpy().astype(np.uint8))
                        for mask in mask_targets]
            cpu_preds = [c.detach().cpu().numpy() for c in mask_pred]
            for m, cc in zip(cpu_targets, cpu_preds):
                for mb, c in zip(m, cc):
                    cv2.imshow('target', np.squeeze(mb) * 255)
                    cv2.imshow('preds', (np.squeeze(c) * 255).astype(np.uint8))
                    cv2.imshow('thresdolded', ((np.squeeze(c) > 0.5) * 255).astype(np.uint8))
                    cv2.waitKey(1)

        mask_loss = sum(self.loss_mask(pred, target)
                        for pred, target in zip(mask_pred, mask_targets))
        assert not np.isnan(mask_loss.cpu().detach().numpy())
        loss = {'loss_rpn_mask': mask_loss}
        return loss

    def get_bboxes(self, mask_preds, cfg, rescale=False, loss=None):
        proposals_list = []

        thr = 0.5

        find_proposals = True
        # if loss is not None:
        #     val = min(max(loss - 0.1, 0), 1.0)
        #     if np.random.uniform() < val:
        #         find_proposals = False

        for level_idx, level_pred in enumerate(mask_preds):
            for mask_idx_in_batch, mask_in_batch in enumerate(level_pred):

                boxes = []
                labels = []

                if find_proposals:
                    pred = torch.nn.functional.sigmoid(mask_in_batch)
                    pred_cpu = np.squeeze(pred.detach().cpu().numpy())
                    mask_cpu = (pred_cpu > thr).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    contours = [cv2.approxPolyDP(c, 0.001*cv2.arcLength(c, True), True)
                                for c in contours]

                    contours = unclip_contours(contours)

                    mask_cpu = np.zeros_like(mask_cpu)
                    cv2.drawContours(mask_cpu, contours, -1, 1, -1)

                    for c in contours:
                        x, y, w, h = cv2.boundingRect(c)
                        xmin, ymin, xmax, ymax = x, y, x + w, y + h
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)

                        xmax = min(xmax, mask_cpu.shape[1] - 1)
                        ymax = min(ymax, mask_cpu.shape[0] - 1)

                        min_side = 2
                        if xmax - xmin > min_side and ymax - ymin > min_side:
                            boxes.append(torch.tensor(np.array([[xmin, ymin, xmax, ymax, 1.0]]),
                                         device=mask_preds[0].device, dtype=torch.float))
                            labels.append(torch.tensor(np.array([[1]])))
                            cv2.rectangle(mask_cpu, (xmin, ymin), (xmax, ymax), 2)

                    # cv2.imshow("res", mask_cpu * 120)
                    # cv2.waitKey(1000)

                if boxes:
                    boxes = torch.cat(boxes)
                    labels = torch.cat(labels)
                else:
                    boxes = torch.zeros(
                        (0, 5), device=mask_preds[0].device, dtype=torch.float)
                    labels = torch.zeros(
                        (0, 1), device=mask_preds[0].device, dtype=torch.int)

                proposals_list.append(boxes)


        return proposals_list