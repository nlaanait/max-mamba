from typing import Optional

import numpy as np
from max.graph import TensorValue, ops


def softplus(x: TensorValue) -> TensorValue:
    return ops.log1p(ops.exp(x))


def pad_tensor(
    x: TensorValue, pad: tuple, value: float = 0.0, mode: str = "constant"
) -> TensorValue:
    if mode != "constant":
        raise NotImplementedError("mode != 'constant' not implemented.")

    # store pad dimensions, starting from last axis
    pad_dim = {
        -1 - idx // 2: [pad[idx], pad[idx + 1]] for idx in range(0, len(pad) - 1, 2)
    }

    for dim, val in pad_dim.items():
        if sum(val) < 1:
            continue

        # compute shapes
        pad_shape_left = list(x.shape)
        pad_shape_right = list(x.shape)
        pad_shape_left[dim] = val[0]
        pad_shape_right[dim] = val[1]
        pad_shape_left = [int(itm) for itm in pad_shape_left]
        pad_shape_right = [int(itm) for itm in pad_shape_right]

        # Create padding arrays
        pad_left = ops.constant(
            value=np.full(shape=pad_shape_left, fill_value=value), dtype=x.dtype
        )
        pad_right = ops.constant(
            value=np.full(shape=pad_shape_right, fill_value=value), dtype=x.dtype
        )
        x = ops.concat([pad_left, x, pad_right], axis=dim)
    return x


def clamp_tensor(
    x: TensorValue, min_val: Optional[float], max_val: Optional[float]
) -> TensorValue:
    if min_val and max_val and min_val > max_val:
        raise ValueError("min_val should be less or equal to max_val")
    if not min_val and not max_val:
        raise ValueError("One of 'min_val' or 'max_value' must not be None")
    if min_val:
        x = ops.select(x > min_val, x, min_val)
    if max_val:
        x = ops.select(x < max_val, x, max_val)
    return x


def tile_tensor(x: TensorValue, dims: tuple) -> TensorValue:
    """tile op. Follows pytorch's conventions.

    Args:
        x (TensorValue): input tensor
        dims (tuple):

    Returns:
        (TensorValue): tiled tensor
    """
    while len(dims) < len(x.shape):
        dims = (1,) + dims
    while len(x.shape) < len(dims):
        x = ops.unsqueeze(x, axis=0)
    for idx, dim in enumerate(dims):
        new_shape = (
            (
                tuple(x.shape)[0:idx]
                + (tuple(x.shape)[idx] * dim,)
                + tuple(x.shape)[idx + 1 :]
            )
            if idx != 0
            else (tuple(x.shape)[idx] * dim,) + tuple(x.shape)[idx + 1 :]
        )
        if dim > 1:
            x = ops.stack([x] * dim, axis=idx).reshape(new_shape)
    return x


def roll_tensor(
    x: TensorValue, shifts: tuple[int] | int, dims: tuple[int] | int
) -> TensorValue:
    if isinstance(shifts, int):
        shifts = (shifts,)
    if isinstance(dims, int):
        dims = (dims,)
    if len(shifts) != len(dims):
        raise ValueError("len(shifts) != len(dims)")
    if len(dims) > len(x.shape):
        raise ValueError("len(dims) cannot be greater than len(x.shape)")
    for shift, dim in zip(shifts, dims):
        shift = shift % int(x.shape[dim])
        x = x.transpose(dim_1=0, dim_2=dim)
        if shift > 0:
            x = ops.concat([x[-shift:, ...], x[:-shift, ...]], axis=0)
        else:
            x = ops.concat([x[shift:], x[:shift]], axis=0)
        x = x.transpose(dim_1=dim, dim_2=0)
    return x
