import os.path as osp
import json
import os
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from .base import BaseDataset


class PoseTrackReid(BaseDataset):

    def __init__(self, root, transforms, split, annotated_only=True, keypoints_only=True):
        self.root = root
        self.transforms = transforms
        self.split = split
        assert self.split in ("train", "val")
        self.annotated_only = annotated_only
        self.name = "PoseTrackReid"
        self.img_prefix = osp.join(root, "images")
        self.keypoints_only = keypoints_only
        self.annotations = self._load_annotations()

    def _load_anno_data(self):
        """
        Load the image names for the specific split.
        """
        assert self.split in ("train", "val")
        anno_folder = osp.join(self.root, 'annotations/', self.split)
        seq_files = os.listdir(anno_folder)

        seq_data = dict()

        total_images = 0
        for file in seq_files:
            with open(osp.join(anno_folder, file), 'r') as f:
                data = json.load(f)

            total_images += len(data['images'])
            seq_data[file] = data

        return seq_data

    def _generate_ignore_region(self, anno, img):
        ignore_region = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
        ignore_region = Image.fromarray(ignore_region)
        if 'ignore_regions_x' in anno.keys():
            num_regions = len(anno['ignore_regions_x'])
            if num_regions > 0:
                for r_idx in range(num_regions):
                    contour = []
                    for x, y in zip(anno['ignore_regions_x'][r_idx], anno['ignore_regions_y'][r_idx]):
                        contour.append((x, y))

                    if len(contour) > 2:
                        ImageDraw.Draw(ignore_region).polygon(contour, fill=255, outline=255)

        return ignore_region

    def __getitem__(self, index):
        anno = self.annotations[index]
        img = Image.open(anno["img_path"]).convert("RGB")
        boxes = torch.as_tensor(anno["boxes"], dtype=torch.float32)
        box_centers = torch.as_tensor(anno['box_centers'], dtype=torch.float32)
        keypoints = torch.as_tensor(anno['keypoints'], dtype=torch.float32)
        labels = torch.as_tensor(anno["pids"], dtype=torch.int64)
        target = {"img_name": anno["img_name"],
                  "boxes": boxes,
                  'box_centers': box_centers,
                  "labels": labels,
                  'keypoints': keypoints,
                  'image_id': anno['image_id'],
                  'vid_id': anno['cam_id'],
                  'seq_name': anno['seq_name'],
                  'dataset_root': self.root}

        if 'ignore_regions_x' in anno.keys():
            target['ignore_regions_x'] = anno['ignore_regions_x']
            target['ignore_regions_y'] = anno['ignore_regions_y']
        else:
            target['ignore_regions_x'] = []
            target['ignore_regions_y'] = []

        ignore_region = self._generate_ignore_region(anno, img)
        target['ignore_region'] = ignore_region

        # ToDO: Handle ignore regions!
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def _load_annotations(self):

        annotations = []
        seq_data = self._load_anno_data()

        for seq_name, seq_info in seq_data.items():
            images = {img['id']: img for img in seq_info['images']}

            # sort annotations by im_id
            im_annos = {}
            for anno in seq_info['annotations']:
                im_id = anno['image_id']

                if im_id not in im_annos:
                    im_annos[im_id] = []

                im_annos[im_id].append(anno)

            for im_id, im_info in images.items():
                im_anno = im_annos[im_id] if im_id in im_annos else []

                # get bboxes
                rois = []
                ids = []
                keypoints = []
                box_centers = []

                # we only extract annotations that contain keypoints!
                for ann in im_anno:
                    if self.keypoints_only:
                        if 'keypoints' not in ann:
                            continue
                        if len(ann['keypoints']) == 0:
                            continue

                    bbox = ann['bbox']
                    box_center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
                    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                    person_id = ann['person_id']

                    rois.append(bbox)
                    ids.append(person_id)
                    kpts = np.array(ann['keypoints']).reshape([-1, 3])
                    keypoints.append(kpts)
                    box_centers.append(box_center)

                assert len(rois) == len(ids)

                if (self.annotated_only and len(im_anno) > 0) or not self.annotated_only:
                    annotations.append(
                        dict(
                            img_path=osp.join(self.root, im_info['file_name']),
                            img_name=im_info['file_name'],
                            ignore_regions_x=im_info['ignore_regions_x'],
                            ignore_regions_y=im_info['ignore_regions_y'],
                            boxes=np.array(rois).astype(np.float32),
                            box_centers=np.array(box_centers).astype(np.float32),
                            pids=np.array(ids).astype(np.int32),
                            cam_id=im_info['vid_id'],
                            keypoints=np.array(keypoints).astype(np.float32),
                            image_id=im_info['id'],
                            seq_name=seq_name
                        )
                )

        return annotations
