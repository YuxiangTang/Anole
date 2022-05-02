import logging

logger = logging.getLogger(__name__)


def load_state_dict(model, model_urls, model_root):
    if model_urls == '':
        logger.info("nothing to load...")
        return model

    from torch.utils import model_zoo
    from torch import nn
    import re
    from collections import OrderedDict
    own_state_old = model.state_dict()
    # remove all 'group' string
    own_state = OrderedDict()
    for k, v in own_state_old.items():
        k = re.sub('group\d+\.', '', k)
        own_state[k] = v

    state_dict = model_zoo.load_url(model_urls, model_root)

    for name, param in state_dict.items():
        if name not in own_state:
            logger.info(own_state.keys())
            raise KeyError('unexpected key "{}" in state_dict'.format(name))
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

    missing = set(own_state.keys()) - set(state_dict.keys())
    no_use = set(state_dict.keys()) - set(own_state.keys())

    if len(missing) > 0:
        logger.info('some keys are missing: "{}"'.format(no_use))
    if len(no_use) > 0:
        raise KeyError('some keys are not used: "{}"'.format(no_use))
    logger.info('Load pretrained model successfully.')
