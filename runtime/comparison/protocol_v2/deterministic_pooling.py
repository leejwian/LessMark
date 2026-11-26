"""Exact identity-size adaptive-average-pool gradients for native classifiers.

PyTorch 2.5.1 rejects CUDA adaptive_avg_pool2d backward under strict
determinism even when every pooling bin contains exactly one input element.
AlexNet's 224px input reaches a 6x6 map; VGG13/16 reach a 7x7 map. Their
native adaptive pools request those same spatial sizes, so each derivative
is exactly one and no gradient reduction or overlapping-bin sum is needed.

Keep the original module, its native forward output, and every checkpoint
key. A forward hook only replaces this identity-size gradient edge. Any
other spatial size uses the complete native operation, including its strict
determinism error where applicable; it is never silently treated as identity.
The CUDA path is limited to the experiment's FP32 contiguous NCHW tensors.
"""
from __future__ import annotations

import torch


_NATIVE_OUTPUT_SIZES = {"alexnet": (6, 6), "vgg13": (7, 7), "vgg16": (7, 7)}


def _native_cuda_fp32_identity_gradient(gradient):
    # PyTorch v2.5.1 AdaptiveAveragePooling.cu uses global atomicAdd even for
    # a one-element bin. PTX atom/red.add.f32 flushes subnormal inputs to
    # signed zero; adding those to the initial +0 produces +0. Reproduce it
    # without atomics so even this floating-point boundary remains unchanged.
    # https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/AdaptiveAveragePooling.cu
    # https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-atom
    result = gradient + 0
    # Native autograd differentiates the one-element mean as identity, not
    # the floating-point flush boundary. Preserve that higher-order edge.
    with torch.no_grad():
        result.masked_fill_(gradient.abs() < torch.finfo(torch.float32).tiny, 0)
    return result


class _IdentitySpatialAveragePoolBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, native_output):
        # native_output is detached by the hook, so no edge to the rejected
        # native CUDA backward remains. Forward values/layout come from the
        # unmodified native pool, not a separately implemented approximation.
        ctx.native_cuda_fp32 = input.is_cuda
        return native_output

    @staticmethod
    def backward(ctx, grad_output):
        # Native backward initializes the input gradient to +0 then adds the
        # sole output gradient / 1 / 1. Adding +0 also preserves that signed
        # zero convention; simply returning grad_output would retain -0.
        gradient = (_native_cuda_fp32_identity_gradient(grad_output)
                    if ctx.native_cuda_fp32 else grad_output + 0)
        return gradient, None


def _identity_spatial_backward(module, args, kwargs, native_output):
    input = args[0] if args else kwargs["input"]
    if input.is_cuda and (input.dtype != torch.float32 or not input.is_contiguous()
                          or input.is_contiguous(memory_format=torch.channels_last)):
        # Do not extrapolate the reviewed NCHW/FP32 CUDA arithmetic to a
        # different native kernel or precision. Keep its native behavior.
        return native_output
    if (torch.is_grad_enabled() and input.requires_grad
            and tuple(input.shape[-2:]) == tuple(module.output_size)):
        return _IdentitySpatialAveragePoolBackward.apply(input, native_output.detach())
    return native_output


def install_identity_pool_backward(model, architecture):
    """Attach once to the exact native AlexNet/VGG pool, without RNG use.

The model class, module tree/types/repr, parameters, buffers and state_dict
are unchanged. The hook is a module-level function so deepcopy is safe.
Code revision in normal execution/cache contracts records the implementation
change independently of the unchanged scientific model specification.
"""
    expected = _NATIVE_OUTPUT_SIZES[architecture]
    pool = getattr(model, "avgpool", None)
    if type(pool) is not torch.nn.AdaptiveAvgPool2d or tuple(pool.output_size) != expected:
        raise ValueError(f"{architecture}: expected the native AdaptiveAvgPool2d{expected}")
    if _identity_spatial_backward not in pool._forward_hooks.values():
        pool.register_forward_hook(_identity_spatial_backward, with_kwargs=True)
    return model
