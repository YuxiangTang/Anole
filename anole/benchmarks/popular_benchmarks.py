import numpy as np
from .base_dataset import BaseDataset, generate_dataset
from .builder import DATASET

__all__ = ['CCD', 'NUS', 'Cube']


class RAWDataset(BaseDataset):

    def __init__(self,
                 data_dir,
                 dataset_name,
                 fold_idx=0,
                 minik=1,
                 aug_num=1,
                 training=False,
                 input_size=512,
                 statistic_mode=False):
        super().__init__(
            dataset_name=dataset_name,
            data_dir=data_dir,
            input_size=input_size,
            training=training,
            fold_idx=fold_idx,
            aug_num=aug_num,
            statistic_mode=statistic_mode,
            minik=minik,
        )

    def load_data(self, img_path):
        img = np.load(img_path + '.npy').astype(np.float32)[:, :, ::-1]
        mask = np.load(img_path + '_mask.npy').astype(np.bool)
        ill = np.load(img_path + '_gt.npy').astype(np.float32)
        camera = str(np.load(img_path + '_camera.npy'))
        # preprocess raw
        img = img / np.max(img)
        idx1, idx2, _ = np.where(mask == False)
        img[idx1, idx2, :] = 0
        return img, ill, camera


@DATASET.register_obj
def CCD(**kwargs):
    kwargs["dataset_name"] = 'CC_ori'
    return generate_dataset(RAWDataset, **kwargs)


@DATASET.register_obj
def NUS(**kwargs):
    kwargs["dataset_name"] = 'NUS_half'
    return generate_dataset(RAWDataset, **kwargs)


@DATASET.register_obj
def Cube(**kwargs):
    kwargs["dataset_name"] = 'Cube_half'
    return generate_dataset(RAWDataset, **kwargs)
