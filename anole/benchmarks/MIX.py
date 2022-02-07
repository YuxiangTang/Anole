import numpy as np
from .base_dataset import generate_dataset
from .popular_benchmarks import RAWDataset
from .builder import DATASET

__all__ = ['custom_dataset']


def mix_dataset(data_dir,
                dataset_list,
                fold_idx=0,
                minik=1,
                aug_num=1,
                training=False,
                input_size=512,
                statistic_mode=False):
    assert (dataset_list is list)
    total_img_list = []
    for dataset_name in dataset_list:
        tmp_list = RAWDataset(
            dataset_name=dataset_name,
            data_dir=data_dir,
            input_size=input_size,
            training=training,
            fold_idx=fold_idx,
            aug_num=aug_num,
            statistic_mode=statistic_mode,
            minik=minik,
        ).img_list
        total_img_list += tmp_list
    return RAWDataset(
        data_dir=data_dir,
        input_size=input_size,
        training=training,
        fold_idx=fold_idx,
        aug_num=aug_num,
        statistic_mode=statistic_mode,
        minik=minik,
        defalut_list=total_img_list,
    )


@DATASET.register_obj
def custom_dataset(**kwargs):
    if type(kwargs["dataset_name"]) is not list:
        kwargs["dataset_name"] = [kwargs["dataset_name"]]
    return generate_dataset(mix_dataset, **kwargs)
