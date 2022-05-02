import os

from torch.utils.tensorboard import SummaryWriter
import weakref

from .base_logger import BaseLogger
from ...builder import LOGGERPLUGIN

__all__ = ["TensorboardLogger"]


class TensorboardLogger(BaseLogger):
    """
    The `TensorboardLogger` provides an easy integration with
    Tensorboard logging. Each monitored metric is automatically
    logged to Tensorboard.
    The user can inspect results in real time by appropriately launching
    tensorboard with `tensorboard --logdir=/path/to/tb_log_exp_name`.
    AWS's S3 buckets and (if tensorflow is installed) GCloud storage url are
    supported.
    If no parameters are provided, the default folder in which tensorboard
    log files are placed is "./runs/".
    .. note::
        We rely on PyTorch implementation of Tensorboard. If you
        don't have Tensorflow installed in your environment,
        tensorboard will tell you that it is running with reduced
        feature set. This should not impact on the logger performance.
    """

    def __init__(
        self,
        tb_log_dir: str = ".",
    ):
        """
        Creates an instance of the `TensorboardLogger`.
        :param tb_log_dir: path to the directory where tensorboard log file
            will be stored. Default to "./tb_data".
        :param filename_suffix: string suffix to append at the end of
            tensorboard log file. Default ''.
        """
        super().__init__()
        if not os.path.exists(f"{tb_log_dir}"):
            os.makedirs(f"{tb_log_dir}")
        self.writer = SummaryWriter(f"{tb_log_dir}")

        # Shuts down the writer gracefully on process exit
        # or when this logger gets GCed. Fixes issue #864.
        # For more info see:
        # https://docs.python.org/3/library/weakref.html#comparing-finalizers-with-del-methods
        weakref.finalize(self, SummaryWriter.close, self.writer)

    def log_single_metric(self, name, value, x_plot):
        self.writer.add_scalar(name, value, global_step=x_plot)


@LOGGERPLUGIN.register_obj
def tensorboard_logger(**kwargs):
    return TensorboardLogger(**kwargs)
