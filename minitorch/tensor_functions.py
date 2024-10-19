"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            # print(f"result of mulreduce/apply is {a.f.mul_reduce(a, int(dim.item()))}")
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)
        
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # Since the derivative of the logical 'All' function is zero almost everywhere,
        # we return a zero tensor with the same shape as the input.
        input_shape, dim = ctx.saved_values
        return Tensor.zeros(input_shape, backend=grad_output.backend)


# TODO: Implement for Task 2.3.
class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        t1, t2 = ctx.saved_values
        return grad_output * t2, grad_output * t1
    
class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the output with respect to the input"""
        (sigmoid_a,) = ctx.saved_values
        one = minitorch.Tensor.make(
            [1.0] * sigmoid_a._tensor.size, sigmoid_a.shape, backend=sigmoid_a.backend
        )
        one_minus_sigmoid = one - sigmoid_a
        sigmoid_derivative = sigmoid_a * one_minus_sigmoid
        grad_input = grad_output * sigmoid_derivative
        return grad_input

class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return t1.f.relu_back_zip(t1, grad_output)

class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return t1.f.log_back_zip(t1, grad_output)
    
class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (exp_t1,) = ctx.saved_values
        grad_input = grad_output.f.mul_zip(grad_output, exp_t1)
        return grad_input

# class Sum(Function):
#     @staticmethod
#     def forward(ctx: Context, a: Tensor, dims: Optional[int] = None) -> Tensor:
#         ctx.save_for_backward(a.shape, dims)
#         print(f"dims is {dims.to_numpy()}")
#         if dims is None:
#             # Sum over all dimensions
#             return a.f.add_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)
#         else:
#             result = a
#             print(f"dims is {dims.to_numpy()}")
#             print(type(dims))
#             count = 0
#             for dim in dims:
#                 count += 1
#                 print(dim)
#                 print(f"on pass {count}, tensor")
#                 print(result.to_numpy)
#                 result = result.f.add_reduce(result, int(dim))
#                 print(result.to_numpy)            
#             return result

#     @staticmethod
#     def backward(ctx: Context, grad_output: Tensor) -> Tensor:
#         original_shape, dims = ctx.saved_values
        
#         if dims is None:
#             grad = grad_output * Tensor.ones(original_shape, backend=grad_output.backend)
#         else:
#             grad_shape = list(original_shape)
#             for dim in dims:
#                 grad_shape[dim] = 1  # Set the summed dimensions to 1
#             grad = grad_output.view(*grad_shape) * Tensor.ones(original_shape, backend=grad_output.backend)
        
#         return grad, 0.0

class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim_tensor: Tensor) -> Tensor:
        # print(f"dim tensor is {dim_tensor.to_numpy()}")
        # print(f" size of dimtensor is {dim_tensor.size()}")
        # dim = int(dim_tensor[0]) #uses .item instead
        # ctx.save_for_backward(a, dim_tensor)
        # return a.f.add_reduce(a, dim)
        dims = [int(d) for d in dim_tensor.to_numpy()]
        ctx.save_for_backward(a, dim_tensor)
        
        result = a
        for dim in dims:
            result = result.f.add_reduce(result, dim)
        
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # a, dim_tensor = ctx.saved_values
        # dim = int(dim_tensor.item())
        # shape = list(a.shape)
        # shape[dim] = 1

        # grad_output_reshaped = grad_output.view(*shape)

        # ones_tensor = ones(a.shape, backend=a.backend)
        # grad_input = grad_output_reshaped * ones_tensor
        # zero_grad = zeros(dim_tensor.shape, backend=dim_tensor.backend)
        # return grad_input, zero_grad
        a, dim = ctx.saved_values
        grad_a = a.expand(grad_output)
        grad_dim = grad_output.zeros(dim.shape)
        return grad_a, grad_dim
    
class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output.zeros(grad_output.shape), grad_output.zeros(grad_output.shape)

class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output.zeros(grad_output.shape), grad_output.zeros(grad_output.shape)

class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.is_close_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[None, None]:
        # No backward needed
        return None, None
    
class Permute(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tensor) -> Tensor:
        ctx.save_for_backward(order)
        order2 = [int(order[i]) for i in range(order.size)]

        if len(order2) != len(t1.shape):
            raise ValueError(f"Permutation order length {len(order2)} does not match tensor dimensions {len(a.shape)}.")
        
        if sorted(order2) != list(range(len(t1.shape))):
            raise ValueError(f"Invalid permutation order: {order2}. Must be a permutation of {list(range(len(a.shape)))}.")

        
        permuted_shape = tuple(t1.shape[i] for i in order2)
        permuted_strides = tuple(t1._tensor.strides[i] for i in order2)
        return minitorch.Tensor.make(
            t1._tensor._storage, permuted_shape, permuted_strides, backend=t1.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        order_list = ctx.saved_values[0]
        inverse_order_storage = [0] * order_list.size
        for i in range(order_list.size):
            index = int(order_list[i])  
            inverse_order_storage[index] = i
        
        inverse_order = tensor(inverse_order_storage)

        grad_input = Permute.apply(grad_output, inverse_order)
        zero_grad = zeros(order_list.shape, backend=order_list.backend)
        return grad_input, zero_grad

class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        
        shape2 = [int(shape[i]) for i in range(shape.shape[0])] 
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
# def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
#     """Produce a zero tensor of size `shape`."""
#     return Tensor.make([0.0] * int(operators.prod(shape)), shape, backend=backend)

# def ones(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
#     """Produce a tensor of ones of size `shape`."""
#     return Tensor.make([1.0] * int(operators.prod(shape)), shape, backend=backend)

def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )








