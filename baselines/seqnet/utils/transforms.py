import random
import torch
from PIL import ImageDraw, Image
from torchvision.transforms import functional as F, ColorJitter as ColorJitterOrig
import cv2


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ColorJitter(ColorJitterOrig):
    def __init__(self, brightness, contrast, saturation, hue):
        super(ColorJitter, self).__init__(brightness, contrast, saturation, hue)

    def transform_single(self, image, target):
        image = super(ColorJitter, self).__call__(image)

        return image, target

    def __call__(self, image, target):

        raise NotImplementedError("Needs refinement!")
        if isinstance(image, Image.Image):
            return self.transform_single(image, target)
        else:
            images = []
            targets = []
            for img, tgt in zip(image, target):
                img, tgt = self.transform_single(img, tgt)
                images.append(img)
                targets.append(tgt)

            return images, targets


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def transform_single(self, image, target, random_prob):
        if random_prob < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]] - 1
            target["boxes"] = bbox

            if "box_centers" in target:
                centers = target['box_centers']
                centers[:, 0] = width - centers[:, 0] - 1

            if "box_crops" in target:
                for idx in range(len(target['box_crops'])):
                    target['box_crops'][idx] = target['box_crops'][idx].flip(-1)

            if "normalized_boxes" in target:
                normalized_boxes = target['normalized_boxes']
                normalized_boxes[:, [0, 2]] = 1 - normalized_boxes[:, [0, 2]]

            if "ignore_region" in target:
                target['ignore_region'] = target['ignore_region'].flip(-1)
        return image, target

    def __call__(self, image, target):

        transform_prob = random.random()

        if isinstance(image, torch.Tensor):
            return self.transform_single(image, target, transform_prob)
        elif isinstance(image, list):
            images = []
            targets = []
            for img, tgt in zip(image, target):
                img, tgt = self.transform_single(img, tgt, transform_prob)
                images.append(img)
                targets.append(tgt)

            return images, targets
        else:
            raise NotImplementedError(f"Unknown input class {type(image)}")


class ToTensor:

    def transform_single(self, image, target):
        # convert [0, 255] to [0, 1]
        image = F.to_tensor(image)

        if "box_crops" in target:
            for idx in range(len(target['box_crops'])):
                target['box_crops'][idx] = F.to_tensor(target['box_crops'][idx])

        if "ignore_region" in target:
            target['ignore_region'] = F.to_tensor(target['ignore_region'])
        return image, target

    def __call__(self, image, target):
        if isinstance(image, list):
            images, targets = [], []

            for img, tgt in zip(image, target):
                img, tgt = self.transform_single(img, tgt)
                images.append(img)
                targets.append(tgt)
            return images, targets
        else:
            return self.transform_single(image, target)


def build_transforms(is_train, mask_images, color_jitter=False):
    transforms = []

    if is_train and color_jitter:
        jitter = ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.3)
        transforms.append(jitter)

    transforms.append(ToTensor())
    if is_train:
        transforms.append(RandomHorizontalFlip())
    return Compose(transforms)

