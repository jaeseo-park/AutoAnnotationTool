import os
import random
from collections import OrderedDict
from collections import defaultdict

import cv2
import json_tricks as json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision import transforms
from tqdm import tqdm

from HumanPoseEstimation import HumanPoseEstimationDataset as Dataset

class MOBISDataset(Dataset):
    """
    MOBISDataset class.
    """

    def __init__(self,
                 root_path="", 
                 data_version="train2017", 
                 is_train=True, 
                 use_gt_bboxes=True, 
                 bbox_path="",
                 image_width=288, 
                 image_height=384, 
                 color_rgb=True,
                 scale=True, 
                 scale_factor=0.35, 
                 flip_prob=0.5, 
                 rotate_prob=0.5, 
                 rotation_factor=45., 
                 half_body_prob=0.3,
                 use_different_joints_weight=False, 
                 heatmap_sigma=3, 
                 soft_nms=False,
        ):
        super(MOBISDataset, self).__init__()
        self.root_path = root_path
        self.data_version = data_version
        self.is_train = is_train
        self.use_gt_bboxes = use_gt_bboxes
        self.bbox_path = bbox_path
        self.image_width = image_width
        self.image_height = image_height
        self.color_rgb = color_rgb
        self.scale = scale  # ToDo Check
        self.scale_factor = scale_factor
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotation_factor = rotation_factor
        self.half_body_prob = half_body_prob
        self.use_different_joints_weight = use_different_joints_weight  # ToDo Check
        self.heatmap_sigma = heatmap_sigma
        self.soft_nms = soft_nms
        
        self.data_path = os.path.join(self.root_path, self.data_version)
        self.annotation_path = os.path.join('/home/jayce1/MOBIS/ETC', 'keypoint_annotation2.json')
        
        self.image_size = (self.image_width, self.image_height)
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.heatmap_size = (int(self.image_width / 4), int(self.image_height / 4))
        self.heatmap_type = 'gaussian'
        self.pixel_std = 200 # Don't know what it is
        
        # Keypoints section
        self.nof_joints = 18
        self.nof_joints_half_body = 8
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.upper_body_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.lower_body_ids = [11, 12, 13, 14, 15, 16]
        self.joints_weight = np.asarray(
            [1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5, 1.,],
            dtype=np.float32
        ).reshape((self.nof_joints, 1))
        
        # Transformation for images
        self.transform = transforms.Compose([
            transforms.ToTensor,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load MOBIS dataset - Create MOBIS object then load images and annotations
        self.coco = COCO(self.annotation_path)
        self.imgIds = self.coco.getImgIds()
        
        # Create a list of annotations and the corresponding image
        # --> each image can contain more than one detection
        # @@@@ To our case, set self.use_gt_bboxes always true
        
        
        self.data = []
        
        # Load annotations for each image of MOBIS
        for imgId in tqdm(self.imgIds):
            ann_ids = self.coco.getAnnIds(imgIds=imgId, iscrowd=False)
            img = self.coco.loadImgs(imgId)[0]
            
            if self.use_gt_bboxes: # Set always true
                objs = self.coco.loadAnns(ann_ids)
                
                # Sanitize bboxes
                valid_objs = []
                for obj in objs:
                    # Skip non-person objects (it should never happen)
                    if obj['category_id'] != 1:
                        continue
                        
                    # ignore objs without keypoints annotation
                    if max(obj['keypoints']) == 0:
                        continue
                        
                    x, y, w, h = obj['bbox']
                    x1 = np.max((0, x))
                    y1 = np.max((0, y))
                    x2 = np.min((img['width'] - 1, x1 + np.max((0, w - 1))))
                    y2 = np.min((img['height'] - 1, y1 + np.max((0, h - 1))))
                    
                    # Use only valid bounding boxes
                    if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                        obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                        valid_objs.append(obj)
                    
                objs = valid_objs
            
            else:
                pass
            
            # For each annotation of this image, add the formatted annotation to self.data
            for obj in objs:
                bbox = obj['bbox']
                joints = np.zeros((self.nof_joints, 2), dtype=np.float)
                joints_visibility = np.ones((self.nof_joints, 2), dtype=np.float)
                
                if self.use_gt_bboxes: # Set always true
                    for pt in range(self.nof_joints):
                        joints[pt, 0] = obj['keypoints'][pt * 3 + 0]
                        joints[pt, 1] = obj['keypoints'][pt * 3 + 1]
                        t_vis = int(np.clip(obj['keypoints'][pt * 3 + 2], 0, 1)) # ToDo check correctness
                        # COCO:
                        # if visibility == 0 -> keypoint is not in the image
                        # if visibility == 1 -> keypoint is in the image BUT not visible (e.g. behind an object)
                        # if visibility == 2 -> keypoint looks clearly (i.e. it is not hidden)
                        joints_visibility[pt, 0] = t_vis
                        joints_visibility[pt, 1] = t_vis
                
                center, scale = self._box2cs(obj['clean_bbox'][:4])
                
                self.data.append({
                    'imgId': imgId,
                    'annId': obj['id'],
                    'imgPath': os.path.join(''), # TODO: when dataset is stored in a single folder
                    'center': center,
                    'scale': scale,
                    'bbox': bbox,
                    'joints': joints,
                    'joints_visibility': joints_visibility,
                })
                
        # Done check if we need prepare_data -> we should not
        print('\nCOCO dataset loaded!')
        
        # Default values
        self.bbox_thre = 1.0
        self.image_thre = 0.0
        self.in_vis_thre = 0.2
        self.nms_thre = 1.0
        self.oks_thre = 0.9
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        joints_data = self.data[index].copy()
        
        # Read the image from disk -- NOT YET
        #image = cv2.imread(joints_data['imgPath'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image = '.'
        
        '''
        if self.color_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        if image is None:
            raise ValueError('Fail to read %s' % image)
        '''
        
        joints = joints_data['joints']
        joints_vis = joints_data['joints_visibility']
        
        c = joints_data['center']
        s = joints_data['scale']
        score = joints_data['score'] if 'score' in joints_data else 1
        r = 0
        
        

        target, target_weight = self._generate_target(joints, joints_vis)
        
        # Update metadata
        joints_data['joints'] = joints
        joints_data['joints_visibility'] = joints_vis
        joints_data['center'] = c
        joints_data['scale'] = s
        joints_data['rotation'] = r
        joints_data['score'] = score

        return image, target.astype(np.float32), target_weight.astype(np.float32), joints_data
    
    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)
    
    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2,), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale
    
    def _generate_target(self, joints, joints_vis):
        """
        :param joints:  [nof_joints, 3]
        :param joints_vis: [nof_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        """
        target_weight = np.ones((self.nof_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        if self.heatmap_type == 'gaussian':
            target = np.zeros((self.nof_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.heatmap_sigma * 3

            for joint_id in range(self.nof_joints):
                feat_stride = np.asarray(self.image_size) / np.asarray(self.heatmap_size)
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.heatmap_sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        else:
            raise NotImplementedError

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

if __name__ == "__main__":
    ds_train = MOBISDataset(
        root_path='', data_version="train2017", is_train=True, use_gt_bboxes=True, bbox_path="",
        image_width=640, image_height=480, color_rgb=True,
    )
    '''
    for step, (image, target, target_weight, joints_data) in enumerate(tqdm(ds_train, desc='Training')):
    
        c = joints_data['center']
        s = joints_data['scale']
        score = joints_data['score']
    
        print("----bbox_data----")
        print(joints_data['bbox'])
    '''
    for step, (image, target, target_weight, joints_data) in enumerate(ds_train):
    
        c = joints_data['center']
        s = joints_data['scale']
        score = joints_data['score']
    
        print("----bbox_data----")
        print(joints_data['bbox'])
        print("----joint_data----")
        print(joints_data['joints'])
