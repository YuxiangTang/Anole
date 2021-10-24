from .base_plugin import BasePlugin


class Clock(BasePlugin):
    def __init__(self, iterations_interval):
        """ Counter for strategy events. """
        super().__init__()
        # train
        self.train_epochs = 0
        """ Number of training epochs for the current experience. """

        self.train_iterations = 0
        """ Number of training iterations for the current experience. """

        self.total_iterations = 0
        """ Total number of iterations in training and eval mode. """

        self.iterations_interval = iterations_interval

    def before_training(self, strategy, **kwargs):
        self.train_epochs = 0
        self.total_iterations = 0

    def before_training_epoch(self, strategy, **kwargs):
        self.train_iterations = 0

    def after_training_iteration(self, strategy, **kwargs):
        self.train_iterations += 1
        self.total_iterations += 1

    def after_training_epoch(self, strategy, **kwargs):
        self.train_epochs += 1

    def check_iteration(self, strategy, before):
        iteration_step = self.train_iterations
        if not before:
            iteration_step += 1
        if iteration_step == strategy.dataset.real_len:
            return True

        if iteration_step % self.iterations_interval == 0:
            return True

        return False
