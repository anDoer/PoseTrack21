import math
import sys
import json
import numpy as np
from copy import deepcopy
import os
import os.path as osp
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from eval_func import eval_detection, eval_search_pts, vis_detections 
from utils.utils import MetricLogger, SmoothedValue, mkdir, reduce_dict, warmup_lr_scheduler

to_do_shown = False


def to_device(images, targets, device):
    images = [image.to(device) for image in images]
    for t in targets:
        t["boxes"] = t["boxes"].to(device)
        t["labels"] = t["labels"].to(device)

    return images, targets


def perform_optimizer_step(cfg, gradient_accumulation_ctr):
    return not cfg.SOLVER.ACCUMULATE_GRADIENTS or (
                cfg.SOLVER.ACCUMULATE_GRADIENTS and gradient_accumulation_ctr % cfg.SOLVER.GRADIENT_ACC_INTERVAL == 0)


def train_one_epoch(cfg, model, optimizer, data_loader, device, epoch, tfboard=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    # warmup learning rate in the first epoch
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        # FIXME: min(1000, len(data_loader) - 1)
        warmup_iters = len(data_loader) - 1
        warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    gradient_accumulation_ctr = 0
    loss_sum = 0

    for i, (images, targets) in enumerate(
        metric_logger.log_every(data_loader, cfg.DISP_PERIOD, header)
    ):
        images, targets = to_device(images, targets, device)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        # _C.SOLVER.ACCUMULATE_GRADIENTS = False
        # _C.SOLVER.GRADIENT_ACC_INTERVAL = 2

        optimizer.zero_grad()

        losses.backward()
        if cfg.SOLVER.CLIP_GRADIENTS > 0:
            clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRADIENTS)

        optimizer.step()

        if epoch == 0:
            warmup_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if tfboard:
            iter = epoch * len(data_loader) + i
            for k, v in loss_dict_reduced.items():
                tfboard.add_scalars(f"train/{k}", {k: v}, iter)


@torch.no_grad()
def evaluate_performance(model, gallery_loader, query_loader, device, use_gt=False, use_cache=False, use_cbgm=False,
                         cache_file_name='eval_cache_use_gt_False.pth', k1=30, k2=4, **kwargs):
    """
    Args:
        use_gt (bool, optional): Whether to use GT as detection results to verify the upper
                                bound of person search performance. Defaults to False.
        use_cache (bool, optional): Whether to use the cached features. Defaults to False.
        use_cbgm (bool, optional): Whether to use Context Bipartite Graph Matching algorithm.
                                Defaults to False.
    """
    model.eval()
    cache_file = f"data/eval_cache/{cache_file_name}"

    if use_cache and os.path.isfile(cache_file):
        eval_cache = torch.load(cache_file)
        gallery_dets = eval_cache["gallery_dets"]
        gallery_feats = eval_cache["gallery_feats"]
        query_dets = eval_cache["query_dets"]
        query_feats = eval_cache["query_feats"]
        query_box_feats = eval_cache["query_box_feats"]
    else:

        # regarding query image as gallery to detect all people
        # i.e. query person + surrounding people (context information)
        query_dets, query_feats = [], []
        for images, targets in tqdm(query_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            # targets will be modified in the model, so deepcopy it
            outputs = model(images, deepcopy(targets), query_img_as_gallery=True)

            # consistency check
            gt_box = targets[0]["boxes"].squeeze()
            assert (
                gt_box - outputs[0]["boxes"][0]
            ).sum() <= 0.001, "GT box must be the first one in the detected boxes of query image"

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                query_dets.append(box_w_scores.cpu().numpy())
                query_feats.append(output["embeddings"].cpu().numpy())

        # extract the features of query boxes
        query_box_feats = []
        for images, targets in tqdm(query_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            embeddings = model(images, targets)
            assert len(embeddings) == 1, "batch size in test phase should be 1"
            query_box_feats.append(embeddings[0].cpu().numpy())

        gallery_dets, gallery_feats = [], []
        for images, targets in tqdm(gallery_loader, ncols=0):
            assert len(images) == 1

            images, targets = to_device(images, targets, device)
            if not use_gt:
                outputs = model(images)
            else:
                boxes = targets[0]["boxes"]
                n_boxes = boxes.size(0)
                embeddings = model(images, targets)
                if not to_do_shown:
                    to_do_shown = True
                    print("TODO: INCLUDE BOX DENSITY")

                outputs = [
                    {
                        "boxes": boxes,
                        "embeddings": torch.cat(embeddings),
                        "labels": torch.ones(n_boxes).to(device),
                        "scores": torch.ones(n_boxes).to(device),
                    }
                ]


            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                gallery_dets.append(box_w_scores.cpu().numpy())
                gallery_feats.append(output["embeddings"].cpu().numpy())

        save_dict = {
            "gallery_dets": gallery_dets,
            "gallery_feats": gallery_feats,
            "query_dets": query_dets,
            "query_feats": query_feats,
            "query_box_feats": query_box_feats,
        }

        cache_path = '../data/eval_cache'

        mkdir(f"{cache_path}")

        torch.save(save_dict, f"{cache_path}/{cache_file_name}")

    det_rate, ap, detections_to_ignore = eval_detection(gallery_loader.dataset, gallery_dets, det_thresh=0.01)

    removed_detections = 0
    for idx in range(len(detections_to_ignore)):
        ignore_boxes = detections_to_ignore[idx]

        if len(ignore_boxes) > 0:
            dets = gallery_dets[idx]
            refined_dets = []
            refined_feats = []
            for det_idx in range(len(dets)):
                if det_idx not in ignore_boxes:
                    refined_dets.append(dets[det_idx].tolist())
                    refined_feats.append(gallery_feats[idx][det_idx].tolist())
                else:
                    removed_detections += 1

            if len(refined_dets) > 0 or len(refined_feats) > 0:
                refined_dets = np.array(refined_dets).astype(dets.dtype)
                refined_feats = np.array(refined_feats)

                if len(refined_feats.shape) == 1 or len(refined_dets.shape) == 1:
                    import pdb; pdb.set_trace()
            gallery_dets[idx] = refined_dets
            gallery_feats[idx] = refined_feats

   
    with_kwargs = False 
    eval_search_func = eval_search_pts
    
    if not with_kwargs:
        ret = eval_search_func(
                gallery_loader.dataset,
                query_loader.dataset,
                gallery_dets,
                gallery_feats,
                query_box_feats,
                query_dets,
                query_feats,
                cbgm=use_cbgm,
                k1=k1,
                k2=k2,
            )
    else:
        ret = eval_search_func(
            gallery_loader.dataset,
            query_loader.dataset,
            gallery_dets,
            gallery_feats,
            query_box_feats,
            query_dets,
            query_feats,
            cbgm=use_cbgm,
            k1=k1,
            k2=k2,
            **kwargs
        )

    return ret


