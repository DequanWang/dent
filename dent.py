from copy import deepcopy
from sys import modules
import math
import torch
import torch.nn as nn
import torch.jit

from adamod import AdaMod

class Dent(nn.Module):
    """Dent adapts a model by entropy minimization during testing.
    Once dented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, opt_cfg):
        super().__init__()
        self.model = convert_batchnorm(model, globals()[opt_cfg.BN_FUN])
        self.opt_cfg = opt_cfg
        self.steps = opt_cfg.STEPS
        self.criterion = globals()[opt_cfg.LOSS]
        assert self.steps > 0, "dent requires >= 1 step(s) to forward and update"

    def forward(self, x):
        model = configure_batchnorm(x, self.model)
        optimizer = setup_optimizer(
                collect_params(model), self.opt_cfg)

        for _ in range(self.steps):
            forward_and_adapt(x, model, optimizer, self.criterion)
        model.eval()
        y = model(x)
        return y


class SampleAwareStaticBatchNorm2d(nn.BatchNorm2d):

    def forward(self, x):
        scale = self.weight * ((self.running_var + self.eps).rsqrt()).reshape(1, -1)
        bias = self.bias - self.running_mean.reshape(1, -1) * scale
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        return x * scale + bias


class SampleAwareOnlineBatchNorm2d(nn.BatchNorm2d):

    def forward(self, x):
        current_mean = x.mean([0, 2, 3])
        current_var = x.var([0, 2, 3], unbiased=False)
        scale = self.weight * ((current_var + self.eps).rsqrt()).reshape(1, -1)
        bias = self.bias - current_mean.reshape(1, -1) * scale
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        return x * scale + bias


@torch.jit.script
def tent(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean(0)


@torch.jit.script
def shot(x: torch.Tensor) -> torch.Tensor:
    loss_ent = x.softmax(1)
    loss_ent = -torch.sum(loss_ent * torch.log(loss_ent + 1e-5), dim=1).mean(0)
    loss_div = x.softmax(1).mean(0)
    loss_div = torch.sum(loss_div * torch.log(loss_div + 1e-5))
    return loss_ent + loss_div


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, criterion):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    loss = criterion(outputs)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()


def setup_optimizer(params, config):
    """Set up optimizer for dent adaptation.
    Dent needs an optimizer for test-time entropy minimization.
    In principle, dent could make use of any gradient optimizer.
    In practice, we advise choosing AdaMod.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.
    For best results, try tuning the learning rate and batch size.
    """
    if config.METHOD == 'AdaMod':
        return AdaMod(params,
                lr=config.LR,
                betas=(config.BETA, 0.999),
                beta3=config.BETA3,
                weight_decay=config.WD)
    elif config.METHOD == 'Adam':
        return torch.optim.Adam(params,
                lr=config.LR,
                betas=(config.BETA, 0.999),
                weight_decay=config.WD)
    elif config.METHOD == 'SGD':
        return torch.optim.SGD(params,
                lr=config.LR,
                momentum=config.MOMENTUM,
                dampening=config.DAMPENING,
                weight_decay=config.WD,
                nesterov=config.NESTEROV)
    else:
        raise NotImplementedError


def copy_model_state(model):
    """Copy the model states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    return model_state


def load_model_state(model, model_state):
    """Restore the model states from copies."""
    model.load_state_dict(model_state, strict=True)


def collect_params(model):
    """Collect optim params for use with dent."""
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)
    return params


def configure_batchnorm(x, model):
    """Configure model for use with dent."""
    bs = x.size(0)
    # train mode, because dent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what dent updates
    model.requires_grad_(False)
    # configure norm for dent updates:
    # enable grad + keep statisics + repeat affine params
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight = nn.Parameter(m.ckpt_weight.unsqueeze(0).repeat(bs, 1))
            m.bias = nn.Parameter(m.ckpt_bias.unsqueeze(0).repeat(bs, 1))
            m.requires_grad_(True)
    return model


def convert_batchnorm(module, bn_fun):
    module_output = module
    if isinstance(module, nn.BatchNorm2d):
        module_output = bn_fun(module.num_features, module.eps)
        with torch.no_grad():
            module_output.weight = module.weight
            module_output.bias = module.bias
            module_output.register_buffer("ckpt_weight", module.weight)
            module_output.register_buffer("ckpt_bias", module.bias)
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var

    for name, child in module.named_children():
        module_output.add_module(name, convert_batchnorm(child, bn_fun))
    del module
    return module_output
