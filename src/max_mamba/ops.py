from typing import Optional
from max.graph import ops, TensorValue
import numpy as np


def softplus(x: TensorValue) -> TensorValue:
    return ops.log1p(ops.exp(x))


def pad_tensor(
    x: TensorValue, pad: tuple, value=0, mode: str = "constant"
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
