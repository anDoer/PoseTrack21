import os.path as osp

from numpy.lib.arraysetops import isin

import cv2
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import average_precision_score

from utils.km import run_kuhn_munkres
from utils.utils import write_json
from numba import jit
from tqdm import tqdm
from shapely.geometry import box, Polygon, MultiPolygon
from sklearn.metrics import average_precision_score
from scipy.optimize import linear_sum_assignment

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


def vis_detections(gallery_dataset, gallery_dets, save_path, det_thresh=0.5):
    """
    gallery_det (list of ndarray): n_det x [x1, y1, x2, y2, score] per image
    det_thresh (float): filter out gallery detections whose scores below this
    iou_thresh (float): treat as true positive if IoU is above this threshold
    labeled_only (bool): filter out unlabeled background people
    """
    assert len(gallery_dataset) == len(gallery_dets)
    annos = gallery_dataset.annotations

    for anno, det in tqdm(zip(annos, gallery_dets)):
        gt_boxes = anno["boxes"]
        num_gt = gt_boxes.shape[0]

        if det != []:
            det = np.asarray(det)
            inds = np.where(det[:, 4].ravel() >= det_thresh)[0]
            det = det[inds]
            num_det = det.shape[0]
        else:
            num_det = 0

        im_path = anno['img_path']
        img = cv2.imread(im_path)
        output_file_name = f'{osp.basename(osp.dirname(anno["img_name"]))}-{osp.basename(anno["img_name"])}'
        output_file_name = osp.join(save_path, output_file_name)

        for gt_idx in range(num_gt):
            gt_box = gt_boxes[gt_idx]
            x1, y1, x2, y2 = gt_box
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), [0, 255, 0], 3)

        for det_idx in range(num_det):
            det_box = det[det_idx]
            x1, y1, x2, y2 = det_box[:4]
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), [0, 0, 255], 3)

        cv2.imwrite(output_file_name, img)


def eval_detection(
    gallery_dataset, gallery_dets, det_thresh=0.5, iou_thresh=0.5, labeled_only=False
):
    """
    gallery_det (list of ndarray): n_det x [x1, y1, x2, y2, score] per image
    det_thresh (float): filter out gallery detections whose scores below this
    iou_thresh (float): treat as true positive if IoU is above this threshold
    labeled_only (bool): filter out unlabeled background people
    """
    assert len(gallery_dataset) == len(gallery_dets)
    annos = gallery_dataset.annotations

    y_true, y_score = [], []
    count_gt, count_tp = 0, 0
    detections_to_ignore = []

    count_gt_seq, count_tp_seq = dict(), dict()
    for gallery_idx, (anno, det) in enumerate(zip(annos, gallery_dets)):
        detections_to_ignore.append([])
        gt_boxes = anno["boxes"]
        ignore_x = anno['ignore_regions_x']
        ignore_y = anno['ignore_regions_y']

        # ToDO: get ignore region

        if anno['cam_id'] not in count_gt_seq:
            count_gt_seq[anno['cam_id']] = 0
        if anno['cam_id'] not in count_tp_seq:
            count_tp_seq[anno['cam_id']] = 0

        if labeled_only:
            # exclude the unlabeled people (pid == 5555)
            assert False
            inds = np.where(anno["pids"].ravel() != 5555)[0]
            if len(inds) == 0:
                continue
            gt_boxes = gt_boxes[inds]
        num_gt = gt_boxes.shape[0]

        if det != []:
            det = np.asarray(det)
            inds = np.where(det[:, 4].ravel() >= det_thresh)[0]
            det = det[inds]
            num_det = det.shape[0]
        else:
            num_det = 0
        if num_det == 0:
            count_gt += num_gt
            count_gt_seq[anno['cam_id']] += num_gt
            continue

        # remove boxes in ignore regions 
        ignore_box_candidates = set()
        ignore_iou_thres = 0.1 

        if len(ignore_x) > 0:
            if not isinstance(ignore_x[0], list):
                ignore_x = [ignore_x]
        if len(ignore_y) > 0:
            if not isinstance(ignore_y[0], list):
                ignore_y = [ignore_y]
        assert len(ignore_y) == len(ignore_x)
        # build ignore regions 
        ignore_regions = []

        for r_idx in range(len(ignore_x)):
            region = []

            for x, y in zip(ignore_x[r_idx], ignore_y[r_idx]):
                region.append([x, y])

            ignore_region = Polygon(region)
            ignore_regions.append(ignore_region)

        region_ious = np.zeros((len(ignore_regions), num_det), dtype=np.float32)
        det_boxes = []
        for j in range(num_det):
            x1 = det[j, 0]
            y1 = det[j, 1]
            x2 = det[j, 2]
            y2 = det[j, 3]

            box_poly = Polygon([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
                [x1, y1],
            ])

            det_boxes.append(box_poly)

        for i in range(len(ignore_regions)):
            for j in range(num_det):
                if ignore_regions[i].is_valid:
                    poly_intersection = ignore_regions[i].intersection(det_boxes[j]).area
                    poly_union = ignore_regions[i].union(det_boxes[j]).area
                else:
                    multi_poly = ignore_regions[i].buffer(0)
                    poly_intersection = 0
                    poly_union = 0

                    if isinstance(multi_poly, Polygon):
                        poly_intersection = multi_poly.intersection(det_boxes[j]).area
                        poly_union = multi_poly.union(det_boxes[j]).area
                    else:
                        for poly in multi_poly:
                            poly_intersection += poly.intersection(det_boxes[j]).area
                            poly_union += poly.union(det_boxes[j]).area

                region_ious[i, j] = poly_intersection / poly_union

            candidates = np.argwhere(region_ious[i] > ignore_iou_thres)
            if len(candidates) > 0:
                candidates = candidates[:, 0].tolist()
                ignore_box_candidates.update(candidates)

        ious = np.zeros((num_gt, num_det), dtype=np.float32)
        for i in range(num_gt):
            for j in range(num_det):
                ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
        tfmat = ious >= iou_thresh
        # for each det, keep only the largest iou of all the gt
        for j in range(num_det):
            largest_ind = np.argmax(ious[:, j])
            for i in range(num_gt):
                if i != largest_ind:
                    tfmat[i, j] = False
        # for each gt, keep only the largest iou of all the det
        for i in range(num_gt):
            largest_ind = np.argmax(ious[i, :])
            for j in range(num_det):
                if j != largest_ind:
                    tfmat[i, j] = False

        for j in range(num_det):
            if j in ignore_box_candidates and not tfmat[:, j].any():
                # we have a detection in ignore region
                # a=1
                # box = det[j, :4]
                # image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [0, 0, 255])
                # plt.imshow(image[:, :, ::-1])
                # plt.show()
                detections_to_ignore[gallery_idx].append(j)
                continue

            y_score.append(det[j, -1])
            if tfmat[:, j].any():
                y_true.append(True)
            else:
                y_true.append(False)
            # y_true.append(tfmat[:, j].any())
        count_tp += tfmat.sum()
        count_gt += num_gt

        count_tp_seq[anno['cam_id']] += tfmat.sum()
        count_gt_seq[anno['cam_id']] += num_gt

    det_rate = count_tp * 1.0 / count_gt
    if len(y_true) > 0:
        ap = average_precision_score(y_true, y_score) * det_rate
        if ap != ap:
            ap = 0
    else:
        ap = 0

    print("{} detection:".format("labeled only" if labeled_only else "all"))
    print("  recall = {:.2%}".format(det_rate))
    if not labeled_only:
        print("  ap = {:.2%}".format(ap))

    for cam_id in count_tp_seq.keys():
        det_rate_s = count_tp_seq[cam_id] * 1.0 / count_gt_seq[cam_id]
        # print(f"{cam_id}: recall {det_rate_s}")

    return det_rate, ap, detections_to_ignore


# @jit(forceobj=True)
def eval_search_pts(
    gallery_dataset,
    query_dataset,
    gallery_dets,
    gallery_feats,
    query_box_feats,
    query_dets,
    query_feats,
    k1=30,
    k2=4,
    det_thresh=0.5,
    cbgm=False,
    ignore_cam_id=True,
    ablation_study=False
):
    """
    gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
    gallery_feat (list of ndarray): n_det x D features per image
    query_feat (list of ndarray): D dimensional features per query image
    det_thresh (float): filter out gallery detections whose scores below this
    gallery_size (int): -1 for using full set
    ignore_cam_id (bool): Set to True acoording to CUHK-SYSU,
                        although it's a common practice to focus on cross-cam match only.
    """
    assert len(gallery_dataset) == len(gallery_dets)
    assert len(gallery_dataset) == len(gallery_feats)
    assert len(query_dataset) == len(query_box_feats)

    annos = gallery_dataset.annotations
    name_to_det_feat = {}
    for anno, det, feat in zip(annos, gallery_dets, gallery_feats):
        name = anno["img_name"]

        if len(det) > 0:
            scores = det[:, 4].ravel()
            inds = np.where(scores >= det_thresh)[0]
            if len(inds) > 0:
                name_to_det_feat[name] = (det[inds], feat[inds])

    aps = []
    accs = []
    topk = [1, 5, 10]
    ret = {"image_root": gallery_dataset.img_prefix, "results": []}

    probe_gt_count = 0
    probe_count = len(query_dataset)
    total_gt = 0
    total_tp = 0

    for i in range(len(query_dataset)):
        y_true, y_score = [], []
        imgs, rois = [], []
        count_gt, count_tp = 0, 0

        feat_p = query_box_feats[i].ravel()

        query_imname = query_dataset.annotations[i]["img_name"]
        query_roi = query_dataset.annotations[i]["boxes"]
        query_pid = query_dataset.annotations[i]["pids"]
        query_cam = query_dataset.annotations[i]["cam_id"]

        # Find all occurence of this query
        gallery_imgs = []
        for x in annos:
            if query_pid in x["pids"] and x["img_name"] != query_imname:
                gallery_imgs.append(x)
        probe_gt_count += len(gallery_imgs)

        if ablation_study and len(gallery_imgs) == 0:
            continue

        query_gts = {}
        for item in gallery_imgs:
            query_gts[item["img_name"]] = item["boxes"][item["pids"] == query_pid]

        if len(gallery_imgs) == 0:
            print("SKIP GALLERY IMAGE AS GALLERY DOES NOT CONTAIN QUERY")
            continue

        # Construct gallery set for this query
        # if ignore_cam_id:
        gallery_imgs = []
        for x in annos:
            if x["img_name"] != query_imname:
                gallery_imgs.append(x)

        name2sim = {}
        sims = []
        imgs_cbgm = []
        # 1. Go through all gallery samples
        for item in gallery_imgs:
            gallery_imname = item["img_name"]
            # some contain the query (gt not empty), some not
            count_gt += gallery_imname in query_gts
            # compute distance between query and gallery dets
            if gallery_imname not in name_to_det_feat:
                continue
            det, feat_g = name_to_det_feat[gallery_imname]
            # get L2-normalized feature matrix NxD
            assert feat_g.size == np.prod(feat_g.shape[:2])
            feat_g = feat_g.reshape(feat_g.shape[:2])
            # compute cosine similarities
            sim = feat_g.dot(feat_p).ravel()

            if gallery_imname in name2sim:
                continue
            name2sim[gallery_imname] = sim
            sims.extend(list(sim))
            imgs_cbgm.extend([gallery_imname] * len(sim))

        if cbgm:
            sims = np.array(sims)
            imgs_cbgm = np.array(imgs_cbgm)
            inds = np.argsort(sims)[-k1:]
            imgs_cbgm = set(imgs_cbgm[inds])
            for img in imgs_cbgm:
                sim = name2sim[img]
                det, feat_g = name_to_det_feat[img]
                qboxes = query_dets[i][:k2]
                qfeats = query_feats[i][:k2]
                assert (
                    query_roi - qboxes[0][:4]
                ).sum() <= 0.001, "query_roi must be the first one in pboxes"

                graph = []
                for indx_i, pfeat in enumerate(qfeats):
                    for indx_j, gfeat in enumerate(feat_g):
                        graph.append((indx_i, indx_j, (pfeat * gfeat).sum()))
                km_res, max_val = run_kuhn_munkres(graph)

                for indx_i, indx_j, _ in km_res:
                    if indx_i == 0:
                        sim[indx_j] = max_val
                        break

        for gallery_imname, sim in name2sim.items():
            det, feat_g = name_to_det_feat[gallery_imname]
            # assign label for each det
            label = np.zeros(len(sim), dtype=np.int32)
            if gallery_imname in query_gts:
                gt = query_gts[gallery_imname].ravel()
                w, h = gt[2] - gt[0], gt[3] - gt[1]
                iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                inds = np.argsort(sim)[::-1]
                sim = sim[inds]
                det = det[inds]
                # only set the first matched det as true positive
                for j, roi in enumerate(det[:, :4]):
                    if _compute_iou(roi, gt) >= iou_thresh:
                        label[j] = 1
                        count_tp += 1
                        break
            y_true.extend(list(label))
            y_score.extend(list(sim))
            imgs.extend([gallery_imname] * len(sim))
            rois.extend(list(det))

        # 2. Compute AP for this query (need to scale by recall rate)
        y_score = np.asarray(y_score)
        y_true = np.asarray(y_true)
        assert count_tp <= count_gt

        recall_rate = count_tp * 1.0 / count_gt
        total_gt += count_gt
        total_tp += count_tp

        ap = 0 if count_tp == 0 or len(y_score) == 0 else average_precision_score(y_true, y_score) * recall_rate
        aps.append(ap)
        inds = np.argsort(y_score)[::-1]
        y_score = y_score[inds]
        y_true = y_true[inds]
        if len(y_true) > 0:
            accs.append([min(1, sum(y_true[:k])) for k in topk])
        else:
            accs.append([0 for k in topk])

        # 4. Save result for JSON dump
        new_entry = {
            "query_img": str(query_imname),
            "query_roi": list(map(float, query_roi.squeeze().tolist())),
            "query_gt": query_gts,
            "gallery": [],
        }
        # only save top-10 predictions
        if not ablation_study:
            for k in range(10):
                new_entry["gallery"].append(
                    {
                        "img": str(imgs[inds[k]]),
                        "roi": list(map(float, rois[inds[k]].tolist())),
                        "score": float(y_score[k]),
                        "correct": int(y_true[k]),
                    }
                )
            ret["results"].append(new_entry)

    print("search ranking:")
    if len(aps) > 0:
        mAP = float(np.mean(aps))
        print("  mAP = {:.2%}".format(mAP))
        accs = np.mean(accs, axis=0)
        for i, k in enumerate(topk):
            print("  top-{:2d} = {:.2%}".format(k, accs[i]))

        # write_json(ret, "vis/results.json")

        ret["mAP"] = float(np.mean(aps))
        ret["accs"] = accs
    else:
        ret['mAP'] = 0
        ret['accs'] = 0

    ret['probe_gt_count'] = probe_gt_count
    ret['probe_count'] = probe_count
    ret['total_gt'] = total_gt
    ret['total_tp'] = total_tp
    return ret


