"""The experts' grouped products as Triton kernels.

PyTorch's grouped product has single kernels only for compute capability 9
to 11. On an RTX 5090 (12.0) it multiplies one expert at a time, and each
call waits for the device to send back where each expert's rows end. These
kernels multiply every expert in one launch and read the ends on the device.
They also gather each sorted row's token as they load it, and can turn the
experts' first products into their hidden units on the way in, so neither
the tokens' copies nor the units are written to memory.

The rows come sorted by expert: ends holds where each expert's rows end, and
rows, when given, holds the input row each sorted row reads. scores, when
given, makes each loaded row the expert's hidden units scaled by the row's
score: a squared ReLU of the row, or with gates a SwiGLU of the gate's row and
the row.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

# Rows, columns and inner steps of a block, warps and pipeline stages.
PRODUCT_BLOCKS = {"cuda": (128, 128, 64, 8, 3), "cpu": (16, 16, 16, 1, 1)}
OUTER_BLOCKS = {"cuda": (128, 128, 64, 8, 3), "cpu": (16, 16, 16, 1, 1)}


@triton.jit
def _hidden_units(raised, gated, scores, UNITS: tl.constexpr):
    """The experts' hidden units of the loaded rows, scaled by their scores, in single precision."""
    # Each form in its own branch: Triton compiles only the branch a constant picks, but
    # compiles what follows a return in it, where the squared ReLU has no gate.
    raised = raised.to(tl.float32)
    if UNITS == 1:
        raised = tl.maximum(raised, 0.0)
        units = raised * raised
    else:
        gated = gated.to(tl.float32)
        units = gated * tl.sigmoid(gated) * raised
    return units * scores


@triton.jit
def _products_kernel(
    inputs, rows, gates, scores, matrices, ends, out, columns, inner,
    input_stride, input_inner_stride, matrix_stride, matrix_inner_stride, matrix_column_stride,
    out_stride, out_column_stride,
    EXPERTS: tl.constexpr, GATHER: tl.constexpr, UNITS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_COLUMNS: tl.constexpr, BLOCK_INNER: tl.constexpr,
):
    # The blocks of rows run expert by expert, so a block never spans two experts.
    tile = tl.program_id(0)
    expert, first, end, tiles = EXPERTS, 0, 0, 0
    start = 0
    for index in tl.static_range(EXPERTS):
        stop = tl.load(ends + index)
        count = tl.cdiv(stop - start, BLOCK_ROWS)
        inside = (tile >= tiles) & (tile < tiles + count)
        expert = tl.where(inside, index, expert)
        first = tl.where(inside, start + (tile - tiles) * BLOCK_ROWS, first)
        end = tl.where(inside, stop, end)
        tiles += count
        start = stop
    if expert == EXPERTS:
        return
    row = first + tl.arange(0, BLOCK_ROWS)
    row_mask = row < end
    source = tl.load(rows + row, mask=row_mask, other=0).to(tl.int64) if GATHER else row.to(tl.int64)
    column = tl.program_id(1) * BLOCK_COLUMNS + tl.arange(0, BLOCK_COLUMNS)
    column_mask = column < columns
    score = tl.load(scores + row, mask=row_mask, other=0.0).to(tl.float32)[:, None] if UNITS else None
    matrix = matrices + expert.to(tl.int64) * matrix_stride
    total = tl.zeros((BLOCK_ROWS, BLOCK_COLUMNS), dtype=tl.float32)
    for step in range(0, inner, BLOCK_INNER):
        position = step + tl.arange(0, BLOCK_INNER)
        position_mask = position < inner
        offsets = source[:, None] * input_stride + position[None, :] * input_inner_stride
        mask = row_mask[:, None] & position_mask[None, :]
        left = tl.load(inputs + offsets, mask=mask, other=0.0)
        if UNITS:
            gated = tl.load(gates + offsets, mask=mask, other=0.0) if UNITS == 2 else None
            left = _hidden_units(left, gated, score, UNITS).to(matrices.dtype.element_ty)
        right = tl.load(
            matrix + position[:, None] * matrix_inner_stride + column[None, :] * matrix_column_stride,
            mask=position_mask[:, None] & column_mask[None, :], other=0.0,
        )
        total = tl.dot(left, right, total)
    tl.store(
        out + row[:, None].to(tl.int64) * out_stride + column[None, :] * out_column_stride,
        total.to(out.dtype.element_ty), mask=row_mask[:, None] & column_mask[None, :],
    )


@triton.jit
def _outer_kernel(
    inputs, input_rows, gates, scores, gradients, gradient_rows, ends, out, width, columns,
    input_stride, input_column_stride, gradient_stride, gradient_column_stride,
    out_stride, out_row_stride, out_column_stride,
    GATHER_INPUTS: tl.constexpr, GATHER_GRADIENTS: tl.constexpr, UNITS: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr, BLOCK_COLUMNS: tl.constexpr, BLOCK_ROWS: tl.constexpr,
):
    expert = tl.program_id(0)
    start = tl.load(ends + expert - 1, mask=expert > 0, other=0)
    end = tl.load(ends + expert)
    position = tl.program_id(1) * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
    position_mask = position < width
    column = tl.program_id(2) * BLOCK_COLUMNS + tl.arange(0, BLOCK_COLUMNS)
    column_mask = column < columns
    total = tl.zeros((BLOCK_WIDTH, BLOCK_COLUMNS), dtype=tl.float32)
    for step in range(start, end, BLOCK_ROWS):
        row = step + tl.arange(0, BLOCK_ROWS)
        row_mask = row < end
        source = tl.load(input_rows + row, mask=row_mask, other=0).to(tl.int64) if GATHER_INPUTS else row.to(tl.int64)
        # The inputs' rows, transposed: each column of this block is one row.
        offsets = source[None, :] * input_stride + position[:, None] * input_column_stride
        mask = position_mask[:, None] & row_mask[None, :]
        left = tl.load(inputs + offsets, mask=mask, other=0.0)
        if UNITS:
            gated = tl.load(gates + offsets, mask=mask, other=0.0) if UNITS == 2 else None
            score = tl.load(scores + row, mask=row_mask, other=0.0).to(tl.float32)[None, :]
            left = _hidden_units(left, gated, score, UNITS).to(gradients.dtype.element_ty)
        target = tl.load(gradient_rows + row, mask=row_mask, other=0).to(tl.int64) if GATHER_GRADIENTS else row.to(tl.int64)
        right = tl.load(
            gradients + target[:, None] * gradient_stride + column[None, :] * gradient_column_stride,
            mask=row_mask[:, None] & column_mask[None, :], other=0.0,
        )
        total = tl.dot(left, right, total)
    tl.store(
        out + expert.to(tl.int64) * out_stride + position[:, None] * out_row_stride + column[None, :] * out_column_stride,
        total, mask=position_mask[:, None] & column_mask[None, :],
    )


def _units(gates, scores):
    if scores is None:
        return 0
    return 1 if gates is None else 2


@torch.library.custom_op("pretrain::expert_products", mutates_args=())
def expert_products(
    inputs: Tensor, matrices: Tensor, ends: Tensor,
    rows: Tensor | None = None, gates: Tensor | None = None, scores: Tensor | None = None,
) -> Tensor:
    """Each sorted row times its expert's matrix, in the matrices' precision."""
    count = inputs.shape[0] if rows is None else rows.shape[0]
    experts, inner, columns = matrices.shape
    out = inputs.new_empty((count, columns), dtype=matrices.dtype)
    block_rows, block_columns, block_inner, warps, stages = PRODUCT_BLOCKS[inputs.device.type]
    grid = (triton.cdiv(count, block_rows) + experts, triton.cdiv(columns, block_columns))
    _products_kernel[grid](
        inputs, ends if rows is None else rows, inputs if gates is None else gates,
        ends if scores is None else scores, matrices, ends, out, columns, inner,
        inputs.stride(0), inputs.stride(1), *matrices.stride(), *out.stride(),
        EXPERTS=experts, GATHER=rows is not None, UNITS=_units(gates, scores),
        BLOCK_ROWS=block_rows, BLOCK_COLUMNS=block_columns, BLOCK_INNER=block_inner,
        num_warps=warps, num_stages=stages,
    )
    return out


@expert_products.register_fake
def _(inputs, matrices, ends, rows=None, gates=None, scores=None):
    count = inputs.shape[0] if rows is None else rows.shape[0]
    return inputs.new_empty((count, matrices.shape[-1]), dtype=matrices.dtype)


@torch.library.custom_op("pretrain::expert_outer_products", mutates_args=())
def expert_outer_products(
    inputs: Tensor, gradients: Tensor, ends: Tensor,
    input_rows: Tensor | None = None, gradient_rows: Tensor | None = None,
    gates: Tensor | None = None, scores: Tensor | None = None,
) -> Tensor:
    """For each expert, the sum over its sorted rows of the input row's
    transpose times the gradient row: the gradient of its matrix, in single
    precision."""
    experts, width, columns = ends.shape[0], inputs.shape[1], gradients.shape[1]
    out = inputs.new_empty((experts, width, columns), dtype=torch.float32)
    block_width, block_columns, block_rows, warps, stages = OUTER_BLOCKS[inputs.device.type]
    grid = (experts, triton.cdiv(width, block_width), triton.cdiv(columns, block_columns))
    _outer_kernel[grid](
        inputs, ends if input_rows is None else input_rows, inputs if gates is None else gates,
        ends if scores is None else scores, gradients, ends if gradient_rows is None else gradient_rows,
        ends, out, width, columns, inputs.stride(0), inputs.stride(1), gradients.stride(0), gradients.stride(1),
        *out.stride(),
        GATHER_INPUTS=input_rows is not None, GATHER_GRADIENTS=gradient_rows is not None,
        UNITS=_units(gates, scores), BLOCK_WIDTH=block_width, BLOCK_COLUMNS=block_columns, BLOCK_ROWS=block_rows,
        num_warps=warps, num_stages=stages,
    )
    return out


@expert_outer_products.register_fake
def _(inputs, gradients, ends, input_rows=None, gradient_rows=None, gates=None, scores=None):
    return inputs.new_empty((ends.shape[0], inputs.shape[1], gradients.shape[1]), dtype=torch.float32)
