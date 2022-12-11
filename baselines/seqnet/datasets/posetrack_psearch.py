import os.path as osp
import json
import numpy as np
import torch
import random
from PIL import Image, ImageDraw
from .base import BaseDataset
import cv2


def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + \
        (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union

class PoseTrackPSearch(BaseDataset):
    def __init__(self, root, transforms, split, **kwargs):
        self.kwargs = kwargs
        self.name = "PoseTrackPSearch"
        self.img_prefix = osp.join(root, "images")
        super(PoseTrackPSearch, self).__init__(root, transforms, split)

    def _get_cam_id(self, img_name):
        raise NotImplemented("Not implemented")

    def _refine_annotations(self, selected_annotations, images):
        # remove annotations that only occur in a single image, they are likely to be a probe
        refined_annotations = []
        selected_images = {}
        for anno in selected_annotations:
            refined_annotations.append(anno)

            if not anno['image_id'] in selected_images:
                selected_images[anno['image_id']] = 1
            else:
                selected_images[anno['image_id']] += 1

        refined_images = []
        for im_id, count in selected_images.items():
            # ToDO: Optional: remove images that do not contain gt annotations!
            if count == 0:
                continue
            refined_images.append(images[im_id])

        pt_data = {'images': refined_images,
                   'annotations': refined_annotations}
        return pt_data

    def _load_queries(self):
        query_info = osp.join(self.root, 'annotations/query.json')

        with open(query_info, 'r') as f:
            query_data = json.load(f)

        image_info = {im['id']: im for im in query_data['images']}
        query_annotations = query_data['annotations']

        queries = []
        # assumption: for every image, we have a query
        for query in query_annotations:
            p_id = query['person_id']
            im_id = query['image_id']
            x, y, w, h = query['bbox']

            box_center = [x + w / 2, y + h / 2]
            box_center = np.array(box_center).astype(np.int32)
            bbox = np.array([x, y, x + w, y + h]).astype(np.int32)

            if len(query['keypoints']) == 0:
                keypoints = np.zeros([15, 3])
                has_kpts = False
            else:
                kpts = np.array(query['keypoints']).reshape([-1, 3]).astype(np.float)
                # remove place-holder indices
                kpts = np.delete(kpts, [3, 4], axis=0)
                keypoints = kpts
                has_kpts = True

            queries.append({
                'img_path': osp.join(self.root, image_info[im_id]['file_name']),
                'img_name': image_info[im_id]['file_name'],
                'boxes': bbox[np.newaxis, :],
                 'box_centers': box_center[np.newaxis, :],
                'pids': np.array([p_id]),
                'flipped': False,
                'cam_id': image_info[im_id]['vid_id'],
                'keypoints': keypoints,
                'has_keypoints': np.array(has_kpts),
                'image_id': im_id
            })

        return queries


    def _load_split_img_names(self):
        """
        Load the image names for the specific split.
        """
        assert self.split in ("train", "gallery")
        if self.split == "train":
            anno_path = osp.join(self.root, 'annotations/train.json')
        else:
            anno_path = osp.join(self.root, 'annotations/val.json')

        total_images_val = 0
        with open(anno_path, 'r') as f:
            pt_data = json.load(f)

        total_images_val += len(pt_data['images'])
        print(total_images_val)
        return pt_data

    def __getitem__(self, index):
        anno = self.annotations[index]
        img = Image.open(anno["img_path"]).convert("RGB")

        boxes = torch.as_tensor(anno["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(anno["pids"], dtype=torch.int64)
        keypoints = torch.as_tensor(anno['keypoints'], dtype=torch.float32)
        has_keypoints = torch.as_tensor(anno['has_keypoints'], dtype=torch.bool)

        target = {
            "img_name": anno["img_name"],
            "image_id": anno['image_id'],
            "boxes": boxes,
            "labels": labels,
            "keypoints": keypoints,
            "has_keypoints": has_keypoints,
            # "box_density": box_density
        }

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

        target['ignore_region'] = ignore_region

        # ToDO: Handle ignore regions!
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def _load_annotations(self):
        if self.split == "query":
            return self._load_queries()

        annotations = []
        pt_data = self._load_split_img_names()
        images = {img['id']: img for img in pt_data['images']}

        self.seq_images = dict()
        for im_id, img in images.items():
            vid_id = img['vid_id']
            if vid_id not in self.seq_images:
                self.seq_images[vid_id] = []

            self.seq_images[vid_id].append(im_id)

        # sort annotations by im_id
        im_annos = {}

        # assign unique pid from 1 to N during training!
        pid_mapping = {}
        pid_ctr = 1
        for anno in pt_data['annotations']:
            im_id = anno['image_id']
            if im_id not in im_annos:
                im_annos[im_id] = []

            if self.split == 'train':
                pid = anno['person_id']
                if pid not in pid_mapping:
                    pid_mapping[pid] = pid_ctr
                    pid_ctr += 1

                anno['person_id'] = pid_mapping[pid]

            im_annos[im_id].append(anno)

        self.sample_idx_im_id_mapping = {}

        for im_id, im_anno in im_annos.items():
            im_info = images[im_id]

            # get bboxes
            rois = []
            ids = []
            box_centers = []
            keypoints = []
            has_kpts = []
            for ann in im_anno:
                bbox = ann['bbox']
                box_center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                person_id = ann['person_id']
                rois.append(bbox)
                ids.append(person_id)
                box_centers.append(box_center)
                if len(ann['keypoints']) == 0:
                    keypoints.append(np.zeros([15, 3]))
                    has_kpts.append(False)
                else:
                    kpts = np.array(ann['keypoints']).reshape([-1, 3]).astype(np.float)
                    # remove place-holder indices
                    kpts = np.delete(kpts, [3, 4], axis=0)
                    keypoints.append(kpts)
                    has_kpts.append(True)

            assert len(rois) == len(ids)

            annotations.append(
                {
                    'img_path': osp.join(self.root, im_info['file_name']),
                    'img_name': im_info['file_name'],
                    'ignore_regions_x': im_info['ignore_regions_x'],
                    'ignore_regions_y': im_info['ignore_regions_y'],
                    'boxes': np.array(rois).astype(np.int32),
                    'box_centers': np.array(box_centers).astype(np.int32),
                    'keypoints': keypoints,
                    'has_keypoints': np.array(has_kpts),
                    'pids': np.array(ids).astype(np.int32),
                    'cam_id': im_info['vid_id'],
                    'image_id': im_id
                }
            )

            self.sample_idx_im_id_mapping[im_id] = len(annotations) - 1

        return annotations
