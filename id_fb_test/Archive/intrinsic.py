import torch
import numpy as np
import os
import torch.autograd.profiler as profiler

from torch import nn
from torch.nn import functional as F
#from fairseq.models.roberta import RobertaModel
from tqdm import tqdm
from typing import Tuple, Optional, Set

from fwh_cuda import fast_walsh_hadamard_transform as fast_walsh_hadamard_transform_cuda


def fast_walsh_hadamard_torched(x, axis: int = 0, normalize: bool = True):
    orig_shape = x.size()
    assert axis >= 0 and axis < len(orig_shape), (
        "For a vector of shape %s, axis must be in [0, %d] but it is %d"
        % (orig_shape, len(orig_shape) - 1, axis)
    )
    h_dim = orig_shape[axis]
    h_dim_exp = int(round(np.log(h_dim) / np.log(2)))
    assert h_dim == 2 ** h_dim_exp, (
        "hadamard can only be computed over axis with size that is a power of two, but"
        " chosen axis %d has size %d" % (axis, h_dim)
    )

    working_shape_pre = [int(torch.prod(torch.tensor(orig_shape[:axis])))]
    working_shape_post = [
        int(torch.prod(torch.tensor(orig_shape[axis + 1:])))
    ]
    working_shape_mid = [2] * h_dim_exp
    working_shape = working_shape_pre + working_shape_mid + working_shape_post

    ret = x.view(working_shape)

    for ii in range(h_dim_exp):
        dim = ii + 1
        arrs = torch.chunk(ret, 2, dim=dim)
        assert len(arrs) == 2
        ret = torch.cat((arrs[0] + arrs[1], arrs[0] - arrs[1]), axis=dim)

    if normalize:
        ret = ret / np.sqrt(float(h_dim))

    ret = ret.view(orig_shape)

    return ret


def fastfood_vars(DD, device=0):
    """
    Returns parameters for fast food transform
    :param DD: desired dimension
    :return:
    """
    ll = int(np.ceil(np.log(DD) / np.log(2)))
    LL = 2 ** ll

    # Binary scaling matrix where $B_{i,i} \in \{\pm 1 \}$ drawn iid
    BB = torch.FloatTensor(LL).uniform_(0, 2).type(torch.LongTensor)
    BB = (BB * 2 - 1)
    BB.requires_grad_(False)

    # Random permutation matrix
    Pi = torch.LongTensor(np.random.permutation(LL))
    Pi.requires_grad_(False)

    # Gaussian scaling matrix, whose elements $G_{i,i} \sim \mathcal{N}(0, 1)$
    GG = torch.FloatTensor(LL,).normal_()
    GG.requires_grad_(False)

    divisor = torch.sqrt(LL * torch.sum(torch.pow(GG, 2)))

    return [BB.to(device), Pi.to(device), GG.to(device), divisor.to(device), LL]


def fastfood_torched(x, DD: int, param_list: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]):
    """
    Fastfood transform
    :param x: array of dd dimension
    :param DD: desired dimension
    :return:
    """
    dd = x.size(0)

    BB, Pi, GG, divisor, LL = param_list
    # Padd x if needed
    dd_pad = F.pad(x, pad=(0, LL - dd), value=0.0, mode="constant")
    # From left to right HGPiH(BX), where H is Walsh-Hadamard matrix
    dd_pad = dd_pad * BB

    # HGPi(HBX)
    mul_2 = FastWalshHadamard.apply(dd_pad)

    # HG(PiHBX)
    mul_3 = mul_2[Pi]

    # H(GPiHBX)
    mul_3 = mul_3 * GG

    # (HGPiHBX)
    mul_5 = FastWalshHadamard.apply(mul_3)

    ret = mul_5[:int(DD)]
    ret = ret / \
        (divisor * np.sqrt(float(DD) / LL))
    return ret


class FastWalshHadamard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(torch.tensor(
            [1 / np.sqrt(float(input.size(0)))]).to(input))
        if input.is_cuda:
            return fast_walsh_hadamard_transform_cuda(input.float(), False)
        else:
            return fast_walsh_hadamard_torched(input.float(), normalize=False)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        if grad_output.is_cuda:
            return input*fast_walsh_hadamard_transform_cuda(grad_output.clone().float(), False).to(grad_output)
        else:
            return input*fast_walsh_hadamard_torched(grad_output.clone().float(), normalize=False).to(grad_output)


class IntrinsicDimensionLight:
    def __init__(self, module: nn.Module, intrinsic_dimension: int, str_filter: Set[str] = set(), said=False, random_seed=1997):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.name_base_localname = []

        self.initial_value = dict()

        self.fastfood_params = {}
        self.said = said
        self.said_size = len(list(module.named_parameters()))
        if self.said:
            assert intrinsic_dimension > self.said_size
            intrinsic_dimension -= self.said_size

        self.intrinsic_parameter = nn.Parameter(
            torch.zeros((intrinsic_dimension)).cpu())
        module.register_parameter(
            "intrinsic_parameter", self.intrinsic_parameter)
        setattr(module, "intrinsic_parameter", self.intrinsic_parameter)

        length = 0
        for name, param in module.named_parameters():
            if param.requires_grad and all([x not in name for x in str_filter]):
                length += 1
                self.initial_value[name] = v0 = (
                    param.clone().detach().requires_grad_(False).to(self.intrinsic_parameter.device)
                )
                DD = np.prod(v0.size())
                self.fastfood_params[name] = fastfood_vars(
                    np.prod(v0.size()), self.intrinsic_parameter.device)
                base, localname = module, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = base.__getattr__(prefix)
                self.name_base_localname.append((name, base, localname))
                if "intrinsic_parameter" not in name:
                    param.requires_grad_(False)
        if said:
            self.intrinsic_parameter_said = nn.Parameter(
                torch.ones((length)).cpu())
            module.register_parameter(
                "intrinsic_parameter_said", self.intrinsic_parameter_said)
            setattr(module, "intrinsic_parameter_said",
                    self.intrinsic_parameter_said)

    def move_to(self, x_tuple, target):
        if isinstance(x_tuple, torch.Tensor):
            return x_tuple.to(target)
        a = []
        for x in x_tuple:
            if isinstance(x, torch.Tensor):
                a.append(x.to(target))
            else:
                a.append(x)
        return tuple(a)

    def requires_to(self, x_tuple, target):
        if isinstance(x_tuple, torch.Tensor):
            x_tuple.requires_grad_(target)
        for x in x_tuple:
            if isinstance(x, torch.Tensor):
                x.requires_grad_(target)

    def fastfood_vars_requires_grad_(self, requires_grad):
        for item in self.fastfood_params.items():
            self.requires_to(item, requires_grad)

    def __call__(self, module, inputs):
        index = 0
        with torch.enable_grad():
            for name, base, localname in self.name_base_localname:
                if localname == "intrinsic_parameter":
                    continue
                self.initial_value[name] = self.initial_value[name].to(
                    getattr(base, localname))
                device_dtype = getattr(base, localname).dtype
                init_shape = self.initial_value[name].size()

                DD = np.prod(init_shape)
                self.fastfood_params[name] = self.move_to(
                    self.fastfood_params[name], module.intrinsic_parameter.device)

                # Fastfood transform te replace dence P
                ray = fastfood_torched(module.intrinsic_parameter, DD, self.fastfood_params[name]).view(
                    init_shape
                )
                if self.said:
                    ray = ray * self.intrinsic_parameter_said[index]
                param = (self.initial_value[name] + ray).to(device_dtype)
                delattr(base, localname)
                setattr(base, localname, param)
                index += 1

    @staticmethod
    def apply(module, intrinsic_dimension, str_filter=set(), said=False):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, IntrinsicDimensionLight) and hook.name == name:
                raise RuntimeError("Cannot register two intrinsic dimension hooks on "
                                   "the same parameter {}".format(name))

        fn = IntrinsicDimensionLight(
            module, intrinsic_dimension, str_filter, said)
        module.register_forward_pre_hook(fn)
        return fn

    @staticmethod
    def apply_with_tensor(module, intrinsic_vector, str_filter=set()):
        assert isinstance(intrinsic_vector,
                          torch.Tensor) and intrinsic_vector.ndim == 1

        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, IntrinsicDimensionLight) and hook.name == name:
                raise RuntimeError("Cannot register two intrinsic dimension hooks on "
                                   "the same parameter {}".format(name))

        fn = IntrinsicDimensionLight(
            module, intrinsic_vector.size(0), str_filter, False)
        fn.intrinsic_parameter = intrinsic_vector
        module.register_forward_pre_hook(fn)
        return fn


def intrinsic_dimension(module, intrinsic_dimension, str_filter):
    IntrinsicDimensionLight.apply(
        module, intrinsic_dimension, str_filter, False)
    return module


def intrinsic_dimension_said(module, intrinsic_dimension, str_filter):
    IntrinsicDimensionLight.apply(
        module, intrinsic_dimension, str_filter, True)
    return module
