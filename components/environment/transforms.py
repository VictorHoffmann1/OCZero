import torch


# TODO: Implement object-based augmentations
class Transforms(object):
    def __init__(self, augmentation):
        self.augmentation = augmentation
        self.transforms = []
        for aug in self.augmentation:
            raise NotImplementedError()
        pass

    def apply_transforms(self, transforms, objects):
        raise NotImplementedError()

    @torch.no_grad()
    def transform(self, objects):
        raise NotImplementedError()
