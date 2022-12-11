import torch

from utils.transforms import build_transforms
from utils.utils import create_small_table

from .posetrack_psearch import PoseTrackPSearch
from .posetrack_reid import PoseTrackReid

def print_statistics(dataset):
    """
    Print dataset statistics.
    """
    num_imgs = len(dataset.annotations)
    num_boxes = 0
    pid_set = set()
    for anno in dataset.annotations:
        num_boxes += anno["boxes"].shape[0]
        for pid in anno["pids"]:
            pid_set.add(pid)
    statistics = {
        "dataset": dataset.name,
        "split": dataset.split,
        "num_images": num_imgs,
        "num_boxes": num_boxes,
    }
    if dataset.name != "CUHK-SYSU" or dataset.split != "query":
        pid_list = sorted(list(pid_set))
        if dataset.split == "query":
            if len(pid_list) > 0:
                num_pids, min_pid, max_pid = len(pid_list), min(pid_list), max(pid_list)
            else:
                num_pids, min_pid, max_pid = 0, 0, 0
            statistics.update(
                {
                    "num_labeled_pids": num_pids,
                    "min_labeled_pid": int(min_pid),
                    "max_labeled_pid": int(max_pid),
                }
            )
        else:
            if dataset.name != 'PoseTrackPSearch':
                unlabeled_pid = pid_list[-1]
                pid_list = pid_list[:-1]  # remove unlabeled pid
                num_pids, min_pid, max_pid = len(pid_list), min(pid_list), max(pid_list)
                statistics.update(
                    {
                        "num_labeled_pids": num_pids,
                        "min_labeled_pid": int(min_pid),
                        "max_labeled_pid": int(max_pid),
                        "unlabeled_pid": int(unlabeled_pid),
                    }
                )
            else:
                num_pids, min_pid, max_pid = len(pid_list), min(pid_list), max(pid_list)
                statistics.update(
                    {
                        "num_labeled_pids": num_pids,
                        "min_labeled_pid": int(min_pid),
                        "max_labeled_pid": int(max_pid)
                    }
                )
    print(f"=> {dataset.name}-{dataset.split} loaded:\n" + create_small_table(statistics))


def build_dataset(dataset_name, root, transforms, split, verbose=True, **kwargs):
    cfg = None
    if 'cfg' in kwargs:
        cfg = kwargs['cfg']

    if dataset_name == "PoseTrackPSearch":
        dataset = PoseTrackPSearch(root, transforms, split, **kwargs)
    elif dataset_name == "PoseTrackReid":
        dataset = PoseTrackReid(root, transforms, split, **kwargs)
    else:
        raise NotImplementedError(f"Unknow dataset: {dataset_name}")
    if verbose:
        print_statistics(dataset)
    return dataset


def collate_fn(batch):
    return tuple(zip(*batch))


def build_collate_fn(cfg):
    if cfg.CUSTOM_COLLATE_FN:
        if 'CUSTOM_COLLATE_FN_TYPE' not in cfg:
            return collate_fn
        else:
            if cfg.CUSTOM_COLLATE_FN_TYPE == 'standard':
                return collate_fn
            else:
                raise NotImplementedError("Unknown collate function name")
    else:
        return None


def build_train_loader(cfg, mask_images=False):
    transforms = build_transforms(is_train=True, mask_images=mask_images, color_jitter=cfg.AUGMENTATION.COLOR_JITTER)
    dataset = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train", cfg=cfg)

    c_fn = build_collate_fn(cfg)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.INPUT.BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=cfg.INPUT.NUM_WORKERS_TRAIN,
        pin_memory=True,
        drop_last=True,
        collate_fn=c_fn,
    )


def build_test_loader(cfg, **kwargs):
    transforms = build_transforms(is_train=False, mask_images=False)
    query_set = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "query", cfg=cfg, **kwargs)
    gallery_set = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "gallery", cfg=cfg, **kwargs)

    c_fn = build_collate_fn(cfg)
    gallery_loader = torch.utils.data.DataLoader(
        gallery_set,
        batch_size=cfg.INPUT.BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=cfg.INPUT.NUM_WORKERS_TEST,
        pin_memory=True,
        collate_fn=c_fn,
    )
    query_loader = torch.utils.data.DataLoader(
        query_set,
        batch_size=cfg.INPUT.BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=cfg.INPUT.NUM_WORKERS_TEST,
        pin_memory=True,
        collate_fn=c_fn,
    )
    return gallery_loader, query_loader

