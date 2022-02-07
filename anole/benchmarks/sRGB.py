import numpy as np
from .base_dataset import BaseDataset, generate_dataset
from .builder import DATASET
from ..raw_processing.color_correction import sRGB2camera
from ..raw_processing.gamma_correction import remove_gamma

__all__ = ['sRGB']


class sRGBDataset(BaseDataset):
    def __init__(self,
                 data_dir,
                 dataset_name,
                 fold_idx=0,
                 minik=1,
                 aug_num=1,
                 training=False,
                 input_size=512,
                 statistic_mode=False,
                 camera_trans="sRGB"):
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
        self.camera_trans = camera_trans

    def load_data(self, img_path):
        img = np.load(img_path).astype(np.float32)[:, :, ::-1]
        img = img / np.max(img)
        R = remove_gamma(img)
        if self.mode == 'train' and self.camera_trans is not None:
            if self.camera_trans == 'all':
                camera = self.camera_type[np.random.randint(0, 11)]
            else:
                assert self.camera_trans in self.camera_type
                camera = self.camera_trans
        else:
            camera = 'sRGB'
        R = sRGB2camera(R, camera)
        return R, camera


@DATASET.register_obj
def sRGB(**kwargs):
    kwargs["dataset_name"] = 'Place205'
    return generate_dataset(sRGBDataset, **kwargs)
