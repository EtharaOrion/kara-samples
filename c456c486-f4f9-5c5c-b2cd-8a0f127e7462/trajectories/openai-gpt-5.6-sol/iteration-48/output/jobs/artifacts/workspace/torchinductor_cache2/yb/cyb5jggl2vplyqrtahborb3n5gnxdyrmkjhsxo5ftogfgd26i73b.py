# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /workspace/torchinductor_cache2/fk/cfk7rbvjpz2snlt6tsknfu2r3zldjrf2afa5isrzccxp65eftfzk.py
# Topologically Sorted Source Nodes: [embedding, embedding_1, add, src, z], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add => add
#   embedding => embedding
#   embedding_1 => embedding_1
#   src => add_1
#   z => add_2, add_3, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_1, %primals_2), kwargs = {})
#   %embedding_1 : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_3, %primals_4), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %unsqueeze), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_6), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_7), kwargs = {})
#   %div_80 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 10), kwargs = {})
triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_0 = async_compile.triton('triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 57344
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    x3 = xindex
    r2 = rindex
    x0 = (xindex % 14)
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([XBLOCK, RBLOCK], 10, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 10), "index out of bounds: 0 <= tmp4 < 10")
    tmp6 = tl.load(in_ptr1 + (r2 + 10*tmp4), rmask, other=0.0)
    tmp8 = tmp7 + tmp1
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert((0 <= tmp10) & (tmp10 < 10), "index out of bounds: 0 <= tmp10 < 10")
    tmp12 = tl.load(in_ptr3 + (r2 + 10*tmp10), rmask, other=0.0)
    tmp13 = tmp6 + tmp12
    tmp14 = r2
    tmp15 = tl.full([1, 1], 2, tl.int64)
    tmp16 = tmp14 < tmp15
    tmp17 = tl.load(in_ptr4 + (r2 + 2*x0), rmask & tmp16, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp13 + tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tl.full([XBLOCK, 1], 10, tl.int32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 / tmp27
    tmp29 = tmp19 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(rmask, tmp31, 0)
    tmp34 = tl.sum(tmp33, 1)[:, None]
    tmp35 = tmp18 - tmp28
    tmp36 = 10.0
    tmp37 = tmp34 / tmp36
    tmp38 = 1e-05
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.rsqrt(tmp39)
    tmp41 = tmp35 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = 0.1
    tmp47 = tmp40 * tmp46
    tl.store(out_ptr0 + (r2 + 10*x3), tmp18, rmask)
    tl.store(out_ptr3 + (r2 + 10*x3), tmp41, rmask)
    tl.store(out_ptr4 + (r2 + 10*x3), tmp45, rmask)
    tl.store(out_ptr5 + (x3), tmp47, None)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/3w/c3w6iusdcfipgqakgstmlgdfu77wsz2wvdddb5t27jreb26twb3l.py
# Topologically Sorted Source Nodes: [mul, clone], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   clone => clone_default_3
#   mul => mul_scalar_4
# Graph fragment:
#   %mul_scalar_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_default_12, 0.668740304976422), kwargs = {})
#   %clone_default_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_default_4,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_1 = async_compile.triton('triton_poi_fused_clone_mul_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 573440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 14)
    x2 = ((xindex // 70) % 2)
    x3 = xindex // 140
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 5*x2 + 10*x1 + 140*x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 5*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.668740304976422
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/c4/cc4abylkyl36zz6daeemyc2kgbomk2a3hud3mhvlhggsnkwklasc.py
# Topologically Sorted Source Nodes: [mul_1, clone_1], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   clone_1 => clone_default_4
#   mul_1 => mul_scalar_5
# Graph fragment:
#   %mul_scalar_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_default_15, 0.668740304976422), kwargs = {})
#   %clone_default_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_default_5,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_2 = async_compile.triton('triton_poi_fused_clone_mul_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40960
    xnumel = 14
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 10)
    y1 = yindex // 10
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 10*x2 + 140*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.668740304976422
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x2 + 14*y3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/zw/czwr3kqlckobdggbfhx4lr2umji4aag7smpcpd3k7y2pz56zpdd6.py
# Topologically Sorted Source Nodes: [amax, sub, exp, sum_1, div, eq, logical_not, any_1, logical_not_1, full, where], Original ATen: [aten._safe_softmax]
# Source node to ATen node mapping:
#   amax => amax_default_1
#   any_1 => any_dim_1
#   div => div_tensor_1
#   eq => eq_scalar_1
#   exp => exp_default_1
#   full => full_default_33
#   logical_not => logical_not_default_2
#   logical_not_1 => logical_not_default_3
#   sub => sub_tensor_1
#   sum_1 => sum_dim_int_list_2
#   where => where_self_1
# Graph fragment:
#   %amax_default_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_default_14, [-1], True), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_default_14, %amax_default_1), kwargs = {})
#   %exp_default_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_1,), kwargs = {})
#   %sum_dim_int_list_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_default_1, [-1], True), kwargs = {})
#   %div_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_1, %sum_dim_int_list_2), kwargs = {})
#   %eq_scalar_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%view_default_14, -inf), kwargs = {})
#   %logical_not_default_2 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_scalar_1,), kwargs = {})
#   %any_dim_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not_default_2, -1, True), kwargs = {})
#   %logical_not_default_3 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_dim_1,), kwargs = {})
#   %full_default_33 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4096, 2, 14, 14], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_self_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_default_3, %full_default_33, %div_tensor_1), kwargs = {})
triton_per_fused__safe_softmax_3 = async_compile.triton('triton_per_fused__safe_softmax_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__safe_softmax_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__safe_softmax_3(in_out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 114688
    rnumel = 14
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 14*x0), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = float("-inf")
    tmp12 = tmp0 == tmp11
    tmp13 = tmp12 == 0
    tmp14 = tmp13.to(tl.int64)
    tmp15 = (tmp14 != 0)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.any(tmp18, 1)[:, None]
    tmp20 = tmp19 == 0
    tmp21 = tmp6 / tmp10
    tmp22 = 0.0
    tmp23 = tl.where(tmp20, tmp22, tmp21)
    tl.store(in_out_ptr0 + (r1 + 14*x0), tmp23, rmask)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/ty/cty2gp4bvymqqfzjhvgisauwbwg4bj6y3gyw4owcd43m7i7x2wq5.py
# Topologically Sorted Source Nodes: [clone_2], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   clone_2 => clone_default_5
# Graph fragment:
#   %clone_default_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_default_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 573440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 14)
    x2 = ((xindex // 70) % 2)
    x3 = xindex // 140
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 5*x2 + 10*x1 + 140*x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 5*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/x4/cx4q6teyza6qwuiukc2yjilh7mrwljcwre52bs6ootrrzof6myxp.py
# Topologically Sorted Source Nodes: [reshape], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   reshape => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 573440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 2)
    x2 = ((xindex // 10) % 14)
    x3 = xindex // 140
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 5*x2 + 70*x1 + 140*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/pn/cpnohlasqx4imp3atadjpoefequup7it6ncj3sj57gij2iv5rywd.py
# Topologically Sorted Source Nodes: [x, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_1 => add_5, add_6, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
#   x => add_4
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %view_17), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_3), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_16), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_17), kwargs = {})
#   %div_78 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 10), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 57344
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 10*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 10*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 10, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 10.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.1
    tmp33 = tmp26 * tmp32
    tl.store(out_ptr2 + (r1 + 10*x0), tmp27, rmask)
    tl.store(out_ptr3 + (r1 + 10*x0), tmp31, rmask)
    tl.store(out_ptr4 + (x0), tmp33, None)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/dq/cdqrkdtrolxhnbp3b5rr737oecrxaqq2r7p6kcr6yo4nbzxebjqp.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_2 => add_7, erf, mul_4, mul_5, mul_6
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, 0.5), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_5,), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_7), kwargs = {})
triton_poi_fused_gelu_7 = async_compile.triton('triton_poi_fused_gelu_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 229376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/fv/cfv6kpa62wg5fwhwkpyj5ohelkvdowbnx7upvfxaic2nnl7bfbit.py
# Topologically Sorted Source Nodes: [x, src_1, z_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   src_1 => add_8
#   x => add_4
#   z_1 => add_11, add_12, mul_10, mul_9, rsqrt_3, sub_4, var_mean_3
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %view_17), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_21), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_7), kwargs = {})
#   %mul_9 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %primals_25), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %primals_26), kwargs = {})
#   %div_76 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_3, 10), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 57344
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 10*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + 10*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + 10*x0), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 10, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 10.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = 0.1
    tmp37 = tmp30 * tmp36
    tl.store(in_out_ptr0 + (r1 + 10*x0), tmp31, rmask)
    tl.store(out_ptr2 + (r1 + 10*x0), tmp35, rmask)
    tl.store(out_ptr3 + (x0), tmp37, None)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/yn/cynruh4qefvigokxhawpshsc3ldu67qwu6i2hv6w4mlcxnpfb3zk.py
# Topologically Sorted Source Nodes: [pad_1], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   pad_1 => constant_pad_nd_1
# Graph fragment:
#   %constant_pad_nd_1 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%primals_22, [0, 8], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_9 = async_compile.triton('triton_poi_fused_constant_pad_nd_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 10)
    x1 = xindex // 10
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + 2*x1), tmp2 & xmask, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/li/cligwrxru2ulp3rqh2gkxlcxy2576nzbew5zwqojisqkbfkxmi63.py
# Topologically Sorted Source Nodes: [layer_norm_2], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_2 => add_10, add_9, clone_4, mul_7, mul_8, rsqrt_2, sub_3, var_mean_2
# Graph fragment:
#   %clone_4 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_4, %getitem_5), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %primals_23), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %primals_24), kwargs = {})
triton_per_fused_native_layer_norm_10 = async_compile.triton('triton_per_fused_native_layer_norm_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = (xindex % 15)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + 10*x0), rmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 10, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 10.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp0 - tmp10
    tmp23 = tmp22 * tmp21
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp21, None)
    tl.store(out_ptr1 + (r2 + 10*x3), tmp27, rmask)
    tl.store(out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/cx/ccxtvw3p25bpwlemi3chlom735dnaaq5uyk65fuwfurzwzsvguyo.py
# Topologically Sorted Source Nodes: [mul, clone], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   clone => clone_default
#   mul => mul_scalar
# Graph fragment:
#   %mul_scalar : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_default, 0.668740304976422), kwargs = {})
#   %clone_default : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_default,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_11 = async_compile.triton('triton_poi_fused_clone_mul_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_11(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 614400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 15)
    x2 = ((xindex // 75) % 2)
    x3 = xindex // 150
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 5*x2 + 10*x1 + 150*x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 5*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.668740304976422
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x4), tmp4, None)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/3n/c3nuh55kgkbf2xuwxrsxim3ckt7fuk75dx3vomz3uczqiixze7sq.py
# Topologically Sorted Source Nodes: [amax, sub, exp, sum_1, div, eq, logical_not, any_1, logical_not_1, full, where], Original ATen: [aten._safe_softmax]
# Source node to ATen node mapping:
#   amax => amax_default
#   any_1 => any_dim
#   div => div_tensor
#   eq => eq_scalar
#   exp => exp_default
#   full => full_default_32
#   logical_not => logical_not_default
#   logical_not_1 => logical_not_default_1
#   sub => sub_tensor
#   sum_1 => sum_dim_int_list
#   where => where_self
# Graph fragment:
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_default_2, [-1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_default_2, %amax_default), kwargs = {})
#   %exp_default : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %sum_dim_int_list : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_default, [-1], True), kwargs = {})
#   %div_tensor : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %sum_dim_int_list), kwargs = {})
#   %eq_scalar : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%view_default_2, -inf), kwargs = {})
#   %logical_not_default : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_scalar,), kwargs = {})
#   %any_dim : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not_default, -1, True), kwargs = {})
#   %logical_not_default_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_dim,), kwargs = {})
#   %full_default_32 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4096, 2, 15, 14], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_self : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_default_1, %full_default_32, %div_tensor), kwargs = {})
triton_per_fused__safe_softmax_12 = async_compile.triton('triton_per_fused__safe_softmax_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__safe_softmax_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__safe_softmax_12(in_out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 122880
    rnumel = 14
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 14*x0), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = float("-inf")
    tmp12 = tmp0 == tmp11
    tmp13 = tmp12 == 0
    tmp14 = tmp13.to(tl.int64)
    tmp15 = (tmp14 != 0)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.any(tmp18, 1)[:, None]
    tmp20 = tmp19 == 0
    tmp21 = tmp6 / tmp10
    tmp22 = 0.0
    tmp23 = tl.where(tmp20, tmp22, tmp21)
    tl.store(in_out_ptr0 + (r1 + 14*x0), tmp23, rmask)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/kj/ckjcfzwxxvanhhbznnvvyzhy5oolqua52r4sqoyw2ge2iiomuen6.py
# Topologically Sorted Source Nodes: [reshape_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   reshape_1 => clone_8
# Graph fragment:
#   %clone_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_18,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_13 = async_compile.triton('triton_poi_fused_clone_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 614400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 2)
    x2 = ((xindex // 10) % 15)
    x3 = xindex // 150
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 5*x2 + 75*x1 + 150*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/ot/cotcjr3k5w4hqcilrqnxy7i4jsssoymajrsyy6v5r4u7ahert5dp.py
# Topologically Sorted Source Nodes: [q_3, z_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   q_3 => add_13
#   z_2 => add_14, add_15, mul_11, mul_12, rsqrt_4, sub_6, var_mean_4
# Graph fragment:
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand_4, %view_39), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_13, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_13, %getitem_9), kwargs = {})
#   %mul_11 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_4), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %primals_35), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %primals_36), kwargs = {})
#   %div_74 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_4, 10), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = (xindex % 15)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + 10*x0), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + 10*x3), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 10, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 10.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.1
    tmp33 = tmp26 * tmp32
    tl.store(out_ptr2 + (r2 + 10*x3), tmp27, rmask)
    tl.store(out_ptr3 + (r2 + 10*x3), tmp31, rmask)
    tl.store(out_ptr4 + (x3), tmp33, None)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/mq/cmqjbjvtiit7ctgxzkwpnvh7ckxocdnjebc2nehzdazl46yxtsn7.py
# Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_4 => clone_9
# Graph fragment:
#   %clone_9 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_9,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_15 = async_compile.triton('triton_poi_fused_clone_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_15(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 614400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 15)
    x2 = ((xindex // 75) % 2)
    x3 = xindex // 150
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 5*x2 + 10*x1 + 150*x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 5*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/fo/cfoqlvsxp5frdtpa3sl6h76mpzd75cc423wtt3ucptwu7hmn3ewo.py
# Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_4 => clone_10
# Graph fragment:
#   %clone_10 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_10,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_16 = async_compile.triton('triton_poi_fused_clone_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_16(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 40960
    xnumel = 15
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 10)
    y1 = yindex // 10
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 10*x2 + 150*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 15*y3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/bd/cbdgjkujqzz2ytnx4j52km6wkbvutvewewedqfldym23gvwdixuo.py
# Topologically Sorted Source Nodes: [s_2, gt, s_3, softmax_2], Original ATen: [aten.div, aten.gt, aten.masked_fill, aten._softmax]
# Source node to ATen node mapping:
#   gt => gt
#   s_2 => div_4
#   s_3 => full_default, where
#   softmax_2 => amax_2, div_5, exp_2, sub_7, sum_3
# Graph fragment:
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_51, 2.23606797749979), kwargs = {})
#   %gt : [num_users=14] = call_function[target=torch.ops.aten.gt.Tensor](args = (%unsqueeze_2, %unsqueeze_3), kwargs = {})
#   %full_default : [num_users=14] = call_function[target=torch.ops.aten.full.default](args = ([], -10000.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %full_default, %div_4), kwargs = {})
#   %amax_2 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where, [-1], True), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %amax_2), kwargs = {})
#   %exp_2 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_7,), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [-1], True), kwargs = {})
#   %div_5 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_2, %sum_3), kwargs = {})
triton_per_fused__softmax_div_gt_masked_fill_17 = async_compile.triton('triton_per_fused__softmax_div_gt_masked_fill_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_div_gt_masked_fill_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_div_gt_masked_fill_17(in_out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 122880
    rnumel = 15
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = (xindex % 15)
    x3 = xindex
    tmp3 = tl.load(in_out_ptr0 + (r2 + 15*x3), rmask, other=0.0)
    tmp0 = r2
    tmp1 = x0
    tmp2 = tmp0 > tmp1
    tmp4 = 0.4472135954999579
    tmp5 = tmp3 * tmp4
    tmp6 = -10000.0
    tmp7 = tl.where(tmp2, tmp6, tmp5)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, float("-inf"))
    tmp11 = triton_helpers.max2(tmp10, 1)[:, None]
    tmp12 = tmp7 - tmp11
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tmp13 / tmp17
    tl.store(in_out_ptr0 + (r2 + 15*x3), tmp18, rmask)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/wv/cwv7qm6g7krceurbrbccnnncxxm7xadvgvin5u5i6jc6vphisojo.py
# Topologically Sorted Source Nodes: [q_3, x_1, layer_norm_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_5 => add_17, add_18, mul_13, mul_14, rsqrt_5, sub_8, var_mean_5
#   q_3 => add_13
#   x_1 => add_16
# Graph fragment:
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand_4, %view_39), kwargs = {})
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_57), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_16, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_16, %getitem_11), kwargs = {})
#   %mul_13 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_5), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %primals_45), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %primals_46), kwargs = {})
#   %div_72 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_5, 10), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_18 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = (xindex % 15)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + 10*x0), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r2 + 10*x3), rmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r2 + 10*x3), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 10, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 10.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = 0.1
    tmp37 = tmp30 * tmp36
    tl.store(in_out_ptr0 + (r2 + 10*x3), tmp8, rmask)
    tl.store(out_ptr2 + (r2 + 10*x3), tmp31, rmask)
    tl.store(out_ptr3 + (r2 + 10*x3), tmp35, rmask)
    tl.store(out_ptr4 + (x3), tmp37, None)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/il/cilu7qlk6xl5bvkubxdjwghv5s37mospjcmh26pm33plavcelhr6.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_5 => add_19, erf_1, mul_15, mul_16, mul_17
# Graph fragment:
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_59, 0.5), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_59, 0.7071067811865476), kwargs = {})
#   %erf_1 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_16,), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_1, 1), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %add_19), kwargs = {})
triton_poi_fused_gelu_19 = async_compile.triton('triton_poi_fused_gelu_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 245760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/rz/crzazawvi64dxbwzwnpk5iy5qnbpdvdncaevwsj5hvojgdfrblvk.py
# Topologically Sorted Source Nodes: [q_5, z_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   q_5 => add_20
#   z_3 => add_21, add_22, mul_18, mul_19, rsqrt_6, sub_9, var_mean_6
# Graph fragment:
#   %add_20 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %view_61), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_20, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_6 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_21,), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_20, %getitem_13), kwargs = {})
#   %mul_18 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %rsqrt_6), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %primals_51), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %primals_52), kwargs = {})
#   %div_71 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_6, 10), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 10*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 10*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 10, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 10.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.1
    tmp33 = tmp26 * tmp32
    tl.store(out_ptr2 + (r1 + 10*x0), tmp27, rmask)
    tl.store(out_ptr3 + (r1 + 10*x0), tmp31, rmask)
    tl.store(out_ptr4 + (x0), tmp33, None)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/3n/c3nrf3pum4y23gbovxrmz5rifkrxqmygsbkwyedzum4eqmzrlrs6.py
# Topologically Sorted Source Nodes: [q_5, x_2, layer_norm_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_7 => add_24, add_25, mul_20, mul_21, rsqrt_7, sub_11, var_mean_7
#   q_5 => add_20
#   x_2 => add_23
# Graph fragment:
#   %add_20 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %view_61), kwargs = {})
#   %add_23 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %view_79), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_23, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_7 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_24,), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_23, %getitem_15), kwargs = {})
#   %mul_20 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %rsqrt_7), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %primals_61), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %primals_62), kwargs = {})
#   %div_69 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_7, 10), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 10*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + 10*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + 10*x0), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 10, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 10.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = 0.1
    tmp37 = tmp30 * tmp36
    tl.store(in_out_ptr0 + (r1 + 10*x0), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + 10*x0), tmp31, rmask)
    tl.store(out_ptr3 + (r1 + 10*x0), tmp35, rmask)
    tl.store(out_ptr4 + (x0), tmp37, None)
''', device_str='cuda')


# kernel path: /workspace/torchinductor_cache2/v3/cv3mmcirhufe7bcddjm6qk74vpvaqqk7qw25vgkqpdlz7clmiiqd.py
# Topologically Sorted Source Nodes: [q_31, layer_norm_32], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_32 => add_112, add_113, mul_109, mul_110, rsqrt_32, sub_48, var_mean_32
#   q_31 => add_111
# Graph fragment:
#   %add_111 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_107, %view_347), kwargs = {})
#   %var_mean_32 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_111, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_112 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_64, 1e-05), kwargs = {})
#   %rsqrt_32 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_112,), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_111, %getitem_65), kwargs = {})
#   %mul_109 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %rsqrt_32), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_109, %primals_67), kwargs = {})
#   %add_113 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_110, %primals_68), kwargs = {})
#   %div_32 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_32, 10), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_22 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '3D6C7E806835DAE8C493444FDE5AEB5724933C8733627B626E0ED6C26E0D3D9F', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    rnumel = 10
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 10*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + 10*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 10, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 10.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.1
    tmp33 = tmp26 * tmp32
    tl.store(in_out_ptr0 + (r1 + 10*x0), tmp27, rmask)
    tl.store(out_ptr2 + (r1 + 10*x0), tmp31, rmask)
    tl.store(out_ptr3 + (x0), tmp33, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70 = args
    args.clear()
    assert_size_stride(primals_1, (10, 10), (10, 1))
    assert_size_stride(primals_2, (4096, 14), (14, 1))
    assert_size_stride(primals_3, (10, 10), (10, 1))
    assert_size_stride(primals_4, (4096, 14), (14, 1))
    assert_size_stride(primals_5, (14, 2), (2, 1))
    assert_size_stride(primals_6, (10, ), (1, ))
    assert_size_stride(primals_7, (10, ), (1, ))
    assert_size_stride(primals_8, (10, 10), (10, 1))
    assert_size_stride(primals_9, (10, ), (1, ))
    assert_size_stride(primals_10, (10, 10), (10, 1))
    assert_size_stride(primals_11, (10, ), (1, ))
    assert_size_stride(primals_12, (10, 10), (10, 1))
    assert_size_stride(primals_13, (10, ), (1, ))
    assert_size_stride(primals_14, (10, 10), (10, 1))
    assert_size_stride(primals_15, (10, ), (1, ))
    assert_size_stride(primals_16, (10, ), (1, ))
    assert_size_stride(primals_17, (10, ), (1, ))
    assert_size_stride(primals_18, (4, 10), (10, 1))
    assert_size_stride(primals_19, (4, ), (1, ))
    assert_size_stride(primals_20, (10, 4), (4, 1))
    assert_size_stride(primals_21, (10, ), (1, ))
    assert_size_stride(primals_22, (15, 2), (2, 1))
    assert_size_stride(primals_23, (10, ), (1, ))
    assert_size_stride(primals_24, (10, ), (1, ))
    assert_size_stride(primals_25, (10, ), (1, ))
    assert_size_stride(primals_26, (10, ), (1, ))
    assert_size_stride(primals_27, (10, 10), (10, 1))
    assert_size_stride(primals_28, (10, ), (1, ))
    assert_size_stride(primals_29, (10, 10), (10, 1))
    assert_size_stride(primals_30, (10, ), (1, ))
    assert_size_stride(primals_31, (10, 10), (10, 1))
    assert_size_stride(primals_32, (10, ), (1, ))
    assert_size_stride(primals_33, (10, 10), (10, 1))
    assert_size_stride(primals_34, (10, ), (1, ))
    assert_size_stride(primals_35, (10, ), (1, ))
    assert_size_stride(primals_36, (10, ), (1, ))
    assert_size_stride(primals_37, (10, 10), (10, 1))
    assert_size_stride(primals_38, (10, ), (1, ))
    assert_size_stride(primals_39, (10, 10), (10, 1))
    assert_size_stride(primals_40, (10, ), (1, ))
    assert_size_stride(primals_41, (10, 10), (10, 1))
    assert_size_stride(primals_42, (10, ), (1, ))
    assert_size_stride(primals_43, (10, 10), (10, 1))
    assert_size_stride(primals_44, (10, ), (1, ))
    assert_size_stride(primals_45, (10, ), (1, ))
    assert_size_stride(primals_46, (10, ), (1, ))
    assert_size_stride(primals_47, (4, 10), (10, 1))
    assert_size_stride(primals_48, (4, ), (1, ))
    assert_size_stride(primals_49, (10, 4), (4, 1))
    assert_size_stride(primals_50, (10, ), (1, ))
    assert_size_stride(primals_51, (10, ), (1, ))
    assert_size_stride(primals_52, (10, ), (1, ))
    assert_size_stride(primals_53, (10, 10), (10, 1))
    assert_size_stride(primals_54, (10, ), (1, ))
    assert_size_stride(primals_55, (10, 10), (10, 1))
    assert_size_stride(primals_56, (10, ), (1, ))
    assert_size_stride(primals_57, (10, 10), (10, 1))
    assert_size_stride(primals_58, (10, ), (1, ))
    assert_size_stride(primals_59, (10, 10), (10, 1))
    assert_size_stride(primals_60, (10, ), (1, ))
    assert_size_stride(primals_61, (10, ), (1, ))
    assert_size_stride(primals_62, (10, ), (1, ))
    assert_size_stride(primals_63, (4, 10), (10, 1))
    assert_size_stride(primals_64, (4, ), (1, ))
    assert_size_stride(primals_65, (10, 4), (4, 1))
    assert_size_stride(primals_66, (10, ), (1, ))
    assert_size_stride(primals_67, (10, ), (1, ))
    assert_size_stride(primals_68, (10, ), (1, ))
    assert_size_stride(primals_69, (10, 10), (10, 1))
    assert_size_stride(primals_70, (10, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 14, 10), (140, 10, 1), torch.float32)
        buf4 = empty_strided_cuda((4096, 14, 10), (140, 10, 1), torch.float32)
        buf5 = empty_strided_cuda((4096, 14, 10), (140, 10, 1), torch.float32)
        buf469 = empty_strided_cuda((4096, 14, 1), (14, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding, embedding_1, add, src, z], Original ATen: [aten.embedding, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_embedding_native_layer_norm_native_layer_norm_backward_0.run(primals_2, primals_1, primals_4, primals_3, primals_5, primals_6, primals_7, buf0, buf4, buf5, buf469, 57344, 10, grid=grid(57344), stream=stream0)
        del primals_1
        del primals_3
        del primals_5
        del primals_7
        buf6 = empty_strided_cuda((57344, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf5, (57344, 10), (10, 1), 0), reinterpret_tensor(primals_8, (10, 10), (1, 10), 0), out=buf6)
        buf7 = empty_strided_cuda((57344, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf5, (57344, 10), (10, 1), 0), reinterpret_tensor(primals_10, (10, 10), (1, 10), 0), out=buf7)
        buf8 = empty_strided_cuda((57344, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf5, (57344, 10), (10, 1), 0), reinterpret_tensor(primals_12, (10, 10), (1, 10), 0), out=buf8)
        buf9 = empty_strided_cuda((4096, 2, 14, 5), (140, 70, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, clone], Original ATen: [aten.mul, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_mul_1.run(buf6, primals_9, buf9, 573440, grid=grid(573440), stream=stream0)
        del primals_9
        buf10 = reinterpret_tensor(buf6, (4096, 2, 5, 14), (140, 70, 14, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [mul_1, clone_1], Original ATen: [aten.mul, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_mul_2.run(buf7, primals_11, buf10, 40960, 14, grid=grid(40960, 14), stream=stream0)
        del primals_11
        buf11 = empty_strided_cuda((8192, 14, 14), (196, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (8192, 14, 5), (70, 5, 1), 0), reinterpret_tensor(buf10, (8192, 5, 14), (70, 14, 1), 0), out=buf11)
        buf15 = reinterpret_tensor(buf11, (4096, 2, 14, 14), (392, 196, 14, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [amax, sub, exp, sum_1, div, eq, logical_not, any_1, logical_not_1, full, where], Original ATen: [aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__safe_softmax_3.run(buf15, 114688, 14, grid=grid(114688), stream=stream0)
        buf16 = reinterpret_tensor(buf7, (4096, 2, 14, 5), (140, 70, 5, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [clone_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf8, primals_13, buf16, 573440, grid=grid(573440), stream=stream0)
        del primals_13
        buf17 = reinterpret_tensor(buf8, (8192, 14, 5), (70, 5, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf15, (8192, 14, 14), (196, 14, 1), 0), reinterpret_tensor(buf16, (8192, 14, 5), (70, 5, 1), 0), out=buf17)
        buf18 = empty_strided_cuda((4096, 14, 2, 5), (140, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf17, buf18, 573440, grid=grid(573440), stream=stream0)
        buf19 = reinterpret_tensor(buf17, (57344, 10), (10, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf18, (57344, 10), (10, 1), 0), reinterpret_tensor(primals_14, (10, 10), (1, 10), 0), out=buf19)
        buf23 = empty_strided_cuda((4096, 14, 10), (140, 10, 1), torch.float32)
        buf24 = empty_strided_cuda((4096, 14, 10), (140, 10, 1), torch.float32)
        buf468 = empty_strided_cuda((4096, 14, 1), (14, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_6.run(buf0, buf19, primals_15, primals_16, primals_17, buf23, buf24, buf468, 57344, 10, grid=grid(57344), stream=stream0)
        del primals_17
        buf25 = empty_strided_cuda((57344, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, reinterpret_tensor(buf24, (57344, 10), (10, 1), 0), reinterpret_tensor(primals_18, (10, 4), (1, 10), 0), alpha=1, beta=1, out=buf25)
        del primals_19
        buf26 = empty_strided_cuda((4096, 14, 4), (56, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_7.run(buf25, buf26, 229376, grid=grid(229376), stream=stream0)
        buf27 = empty_strided_cuda((57344, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf26, (57344, 4), (4, 1), 0), reinterpret_tensor(primals_20, (4, 10), (1, 4), 0), out=buf27)
        buf28 = buf0; del buf0  # reuse
        buf37 = buf28; del buf28  # reuse
        buf40 = empty_strided_cuda((4096, 14, 10), (140, 10, 1), torch.float32)
        buf467 = empty_strided_cuda((4096, 14, 1), (14, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, src_1, z_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf37, buf19, primals_15, buf27, primals_21, primals_25, primals_26, buf40, buf467, 57344, 10, grid=grid(57344), stream=stream0)
        del primals_15
        del primals_21
        del primals_26
        buf29 = empty_strided_cuda((15, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_1], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_9.run(primals_22, buf29, 150, grid=grid(150), stream=stream0)
        del primals_22
        buf30 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        buf31 = empty_strided_cuda((4096, 15, 1), (15, 1, 61440), torch.float32)
        buf33 = reinterpret_tensor(buf31, (4096, 15, 1), (15, 1, 1), 0); del buf31  # reuse
        buf38 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_2], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_10.run(buf33, buf29, primals_23, primals_24, buf30, buf38, 61440, 10, grid=grid(61440), stream=stream0)
        del primals_24
        buf39 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf38, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_27, (10, 10), (1, 10), 0), out=buf39)
        buf41 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf40, (57344, 10), (10, 1), 0), reinterpret_tensor(primals_29, (10, 10), (1, 10), 0), out=buf41)
        buf42 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf40, (57344, 10), (10, 1), 0), reinterpret_tensor(primals_31, (10, 10), (1, 10), 0), out=buf42)
        buf43 = empty_strided_cuda((4096, 2, 15, 5), (150, 75, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, clone], Original ATen: [aten.mul, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_mul_11.run(buf39, primals_28, buf43, 614400, grid=grid(614400), stream=stream0)
        del primals_28
        buf44 = empty_strided_cuda((4096, 2, 5, 14), (140, 70, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1, clone_1], Original ATen: [aten.mul, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_mul_2.run(buf41, primals_30, buf44, 40960, 14, grid=grid(40960, 14), stream=stream0)
        del primals_30
        buf45 = empty_strided_cuda((8192, 15, 14), (210, 14, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bmm], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf43, (8192, 15, 5), (75, 5, 1), 0), reinterpret_tensor(buf44, (8192, 5, 14), (70, 14, 1), 0), out=buf45)
        buf49 = reinterpret_tensor(buf45, (4096, 2, 15, 14), (420, 210, 14, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [amax, sub, exp, sum_1, div, eq, logical_not, any_1, logical_not_1, full, where], Original ATen: [aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__safe_softmax_12.run(buf49, 122880, 14, grid=grid(122880), stream=stream0)
        buf50 = reinterpret_tensor(buf41, (4096, 2, 14, 5), (140, 70, 5, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [clone_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf42, primals_32, buf50, 573440, grid=grid(573440), stream=stream0)
        del buf42
        del primals_32
        buf51 = reinterpret_tensor(buf39, (8192, 15, 5), (75, 5, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf49, (8192, 15, 14), (210, 14, 1), 0), reinterpret_tensor(buf50, (8192, 14, 5), (70, 5, 1), 0), out=buf51)
        buf52 = empty_strided_cuda((4096, 15, 2, 5), (150, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf51, buf52, 614400, grid=grid(614400), stream=stream0)
        buf53 = reinterpret_tensor(buf51, (61440, 10), (10, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf52, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_33, (10, 10), (1, 10), 0), out=buf53)
        buf57 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf58 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf466 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_3, z_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14.run(buf29, buf53, primals_34, primals_35, primals_36, buf57, buf58, buf466, 61440, 10, grid=grid(61440), stream=stream0)
        buf59 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf58, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_37, (10, 10), (1, 10), 0), out=buf59)
        buf60 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf58, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_39, (10, 10), (1, 10), 0), out=buf60)
        buf61 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf58, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_41, (10, 10), (1, 10), 0), out=buf61)
        buf62 = empty_strided_cuda((4096, 2, 15, 5), (150, 75, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf59, primals_38, buf62, 614400, grid=grid(614400), stream=stream0)
        buf63 = reinterpret_tensor(buf59, (4096, 2, 5, 15), (150, 75, 15, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_16.run(buf60, primals_40, buf63, 40960, 15, grid=grid(40960, 15), stream=stream0)
        buf64 = empty_strided_cuda((8192, 15, 15), (225, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf62, (8192, 15, 5), (75, 5, 1), 0), reinterpret_tensor(buf63, (8192, 5, 15), (75, 15, 1), 0), out=buf64)
        buf67 = reinterpret_tensor(buf64, (4096, 2, 15, 15), (450, 225, 15, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [s_2, gt, s_3, softmax_2], Original ATen: [aten.div, aten.gt, aten.masked_fill, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_div_gt_masked_fill_17.run(buf67, 122880, 15, grid=grid(122880), stream=stream0)
        buf68 = reinterpret_tensor(buf60, (4096, 2, 15, 5), (150, 75, 5, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf61, primals_42, buf68, 614400, grid=grid(614400), stream=stream0)
        buf69 = reinterpret_tensor(buf61, (8192, 15, 5), (75, 5, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf67, (8192, 15, 15), (225, 15, 1), 0), reinterpret_tensor(buf68, (8192, 15, 5), (75, 5, 1), 0), out=buf69)
        buf70 = empty_strided_cuda((4096, 15, 2, 5), (150, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf69, buf70, 614400, grid=grid(614400), stream=stream0)
        buf71 = reinterpret_tensor(buf69, (61440, 10), (10, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf70, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_43, (10, 10), (1, 10), 0), out=buf71)
        buf72 = reinterpret_tensor(buf53, (4096, 15, 10), (150, 10, 1), 0); del buf53  # reuse
        buf76 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf77 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf465 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_3, x_1, layer_norm_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_18.run(buf72, buf29, primals_34, buf71, primals_44, primals_45, primals_46, buf76, buf77, buf465, 61440, 10, grid=grid(61440), stream=stream0)
        del primals_34
        buf78 = empty_strided_cuda((61440, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_48, reinterpret_tensor(buf77, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_47, (10, 4), (1, 10), 0), alpha=1, beta=1, out=buf78)
        buf79 = empty_strided_cuda((4096, 15, 4), (60, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_19.run(buf78, buf79, 245760, grid=grid(245760), stream=stream0)
        buf80 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf79, (61440, 4), (4, 1), 0), reinterpret_tensor(primals_49, (4, 10), (1, 4), 0), out=buf80)
        buf84 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf85 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf464 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_5, z_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20.run(buf72, buf80, primals_50, primals_51, primals_52, buf84, buf85, buf464, 61440, 10, grid=grid(61440), stream=stream0)
        buf86 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf85, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_53, (10, 10), (1, 10), 0), out=buf86)
        buf87 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf85, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_55, (10, 10), (1, 10), 0), out=buf87)
        buf88 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf85, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_57, (10, 10), (1, 10), 0), out=buf88)
        buf89 = empty_strided_cuda((4096, 2, 15, 5), (150, 75, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf86, primals_54, buf89, 614400, grid=grid(614400), stream=stream0)
        buf90 = reinterpret_tensor(buf86, (4096, 2, 5, 15), (150, 75, 15, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_16.run(buf87, primals_56, buf90, 40960, 15, grid=grid(40960, 15), stream=stream0)
        buf91 = empty_strided_cuda((8192, 15, 15), (225, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf89, (8192, 15, 5), (75, 5, 1), 0), reinterpret_tensor(buf90, (8192, 5, 15), (75, 15, 1), 0), out=buf91)
        buf94 = reinterpret_tensor(buf91, (4096, 2, 15, 15), (450, 225, 15, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [gt, s_3, s_4, s_5, softmax_3], Original ATen: [aten.gt, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_div_gt_masked_fill_17.run(buf94, 122880, 15, grid=grid(122880), stream=stream0)
        buf95 = reinterpret_tensor(buf87, (4096, 2, 15, 5), (150, 75, 5, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf88, primals_58, buf95, 614400, grid=grid(614400), stream=stream0)
        buf96 = reinterpret_tensor(buf88, (8192, 15, 5), (75, 5, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf94, (8192, 15, 15), (225, 15, 1), 0), reinterpret_tensor(buf95, (8192, 15, 5), (75, 5, 1), 0), out=buf96)
        buf97 = empty_strided_cuda((4096, 15, 2, 5), (150, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf96, buf97, 614400, grid=grid(614400), stream=stream0)
        buf98 = reinterpret_tensor(buf96, (61440, 10), (10, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf97, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_59, (10, 10), (1, 10), 0), out=buf98)
        buf99 = buf72; del buf72  # reuse
        buf103 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf104 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf463 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_5, x_2, layer_norm_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf99, buf80, primals_50, buf98, primals_60, primals_61, primals_62, buf103, buf104, buf463, 61440, 10, grid=grid(61440), stream=stream0)
        buf105 = empty_strided_cuda((61440, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_64, reinterpret_tensor(buf104, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_63, (10, 4), (1, 10), 0), alpha=1, beta=1, out=buf105)
        buf106 = empty_strided_cuda((4096, 15, 4), (60, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_19.run(buf105, buf106, 245760, grid=grid(245760), stream=stream0)
        buf107 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf106, (61440, 4), (4, 1), 0), reinterpret_tensor(primals_65, (4, 10), (1, 4), 0), out=buf107)
        buf111 = reinterpret_tensor(buf80, (4096, 15, 10), (150, 10, 1), 0); del buf80  # reuse
        buf112 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf462 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_7, z_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20.run(buf99, buf107, primals_66, primals_35, primals_36, buf111, buf112, buf462, 61440, 10, grid=grid(61440), stream=stream0)
        buf113 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf112, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_37, (10, 10), (1, 10), 0), out=buf113)
        buf114 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf112, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_39, (10, 10), (1, 10), 0), out=buf114)
        buf115 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf112, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_41, (10, 10), (1, 10), 0), out=buf115)
        buf116 = empty_strided_cuda((4096, 2, 15, 5), (150, 75, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_8], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf113, primals_38, buf116, 614400, grid=grid(614400), stream=stream0)
        buf117 = reinterpret_tensor(buf113, (4096, 2, 5, 15), (150, 75, 15, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [matmul_8], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_16.run(buf114, primals_40, buf117, 40960, 15, grid=grid(40960, 15), stream=stream0)
        buf118 = empty_strided_cuda((8192, 15, 15), (225, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf116, (8192, 15, 5), (75, 5, 1), 0), reinterpret_tensor(buf117, (8192, 5, 15), (75, 15, 1), 0), out=buf118)
        buf121 = reinterpret_tensor(buf118, (4096, 2, 15, 15), (450, 225, 15, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [gt, s_3, s_6, s_7, softmax_4], Original ATen: [aten.gt, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_div_gt_masked_fill_17.run(buf121, 122880, 15, grid=grid(122880), stream=stream0)
        buf122 = reinterpret_tensor(buf114, (4096, 2, 15, 5), (150, 75, 5, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf115, primals_42, buf122, 614400, grid=grid(614400), stream=stream0)
        buf123 = reinterpret_tensor(buf115, (8192, 15, 5), (75, 5, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf121, (8192, 15, 15), (225, 15, 1), 0), reinterpret_tensor(buf122, (8192, 15, 5), (75, 5, 1), 0), out=buf123)
        buf124 = empty_strided_cuda((4096, 15, 2, 5), (150, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf123, buf124, 614400, grid=grid(614400), stream=stream0)
        buf125 = reinterpret_tensor(buf123, (61440, 10), (10, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf124, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_43, (10, 10), (1, 10), 0), out=buf125)
        buf126 = buf99; del buf99  # reuse
        buf130 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf131 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf461 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_7, x_3, layer_norm_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf126, buf107, primals_66, buf125, primals_44, primals_45, primals_46, buf130, buf131, buf461, 61440, 10, grid=grid(61440), stream=stream0)
        buf132 = empty_strided_cuda((61440, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_48, reinterpret_tensor(buf131, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_47, (10, 4), (1, 10), 0), alpha=1, beta=1, out=buf132)
        buf133 = empty_strided_cuda((4096, 15, 4), (60, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_19.run(buf132, buf133, 245760, grid=grid(245760), stream=stream0)
        buf134 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf133, (61440, 4), (4, 1), 0), reinterpret_tensor(primals_49, (4, 10), (1, 4), 0), out=buf134)
        buf138 = reinterpret_tensor(buf107, (4096, 15, 10), (150, 10, 1), 0); del buf107  # reuse
        buf139 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf460 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_9, z_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20.run(buf126, buf134, primals_50, primals_51, primals_52, buf138, buf139, buf460, 61440, 10, grid=grid(61440), stream=stream0)
        buf140 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf139, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_53, (10, 10), (1, 10), 0), out=buf140)
        buf141 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf139, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_55, (10, 10), (1, 10), 0), out=buf141)
        buf142 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf139, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_57, (10, 10), (1, 10), 0), out=buf142)
        buf143 = empty_strided_cuda((4096, 2, 15, 5), (150, 75, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf140, primals_54, buf143, 614400, grid=grid(614400), stream=stream0)
        buf144 = reinterpret_tensor(buf140, (4096, 2, 5, 15), (150, 75, 15, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_16.run(buf141, primals_56, buf144, 40960, 15, grid=grid(40960, 15), stream=stream0)
        buf145 = empty_strided_cuda((8192, 15, 15), (225, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf143, (8192, 15, 5), (75, 5, 1), 0), reinterpret_tensor(buf144, (8192, 5, 15), (75, 15, 1), 0), out=buf145)
        buf148 = reinterpret_tensor(buf145, (4096, 2, 15, 15), (450, 225, 15, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [gt, s_3, s_8, s_9, softmax_5], Original ATen: [aten.gt, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_div_gt_masked_fill_17.run(buf148, 122880, 15, grid=grid(122880), stream=stream0)
        buf149 = reinterpret_tensor(buf141, (4096, 2, 15, 5), (150, 75, 5, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf142, primals_58, buf149, 614400, grid=grid(614400), stream=stream0)
        buf150 = reinterpret_tensor(buf142, (8192, 15, 5), (75, 5, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf148, (8192, 15, 15), (225, 15, 1), 0), reinterpret_tensor(buf149, (8192, 15, 5), (75, 5, 1), 0), out=buf150)
        buf151 = empty_strided_cuda((4096, 15, 2, 5), (150, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf150, buf151, 614400, grid=grid(614400), stream=stream0)
        buf152 = reinterpret_tensor(buf150, (61440, 10), (10, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf151, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_59, (10, 10), (1, 10), 0), out=buf152)
        buf153 = buf126; del buf126  # reuse
        buf157 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf158 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf459 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_9, x_4, layer_norm_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf153, buf134, primals_50, buf152, primals_60, primals_61, primals_62, buf157, buf158, buf459, 61440, 10, grid=grid(61440), stream=stream0)
        buf159 = empty_strided_cuda((61440, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_64, reinterpret_tensor(buf158, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_63, (10, 4), (1, 10), 0), alpha=1, beta=1, out=buf159)
        buf160 = empty_strided_cuda((4096, 15, 4), (60, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_19.run(buf159, buf160, 245760, grid=grid(245760), stream=stream0)
        buf161 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf160, (61440, 4), (4, 1), 0), reinterpret_tensor(primals_65, (4, 10), (1, 4), 0), out=buf161)
        buf165 = reinterpret_tensor(buf134, (4096, 15, 10), (150, 10, 1), 0); del buf134  # reuse
        buf166 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf458 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_11, z_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20.run(buf153, buf161, primals_66, primals_35, primals_36, buf165, buf166, buf458, 61440, 10, grid=grid(61440), stream=stream0)
        buf167 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_34], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf166, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_37, (10, 10), (1, 10), 0), out=buf167)
        buf168 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf166, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_39, (10, 10), (1, 10), 0), out=buf168)
        buf169 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf166, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_41, (10, 10), (1, 10), 0), out=buf169)
        buf170 = empty_strided_cuda((4096, 2, 15, 5), (150, 75, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_12], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf167, primals_38, buf170, 614400, grid=grid(614400), stream=stream0)
        buf171 = reinterpret_tensor(buf167, (4096, 2, 5, 15), (150, 75, 15, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [matmul_12], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_16.run(buf168, primals_40, buf171, 40960, 15, grid=grid(40960, 15), stream=stream0)
        buf172 = empty_strided_cuda((8192, 15, 15), (225, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf170, (8192, 15, 5), (75, 5, 1), 0), reinterpret_tensor(buf171, (8192, 5, 15), (75, 15, 1), 0), out=buf172)
        buf175 = reinterpret_tensor(buf172, (4096, 2, 15, 15), (450, 225, 15, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [gt, s_3, s_10, s_11, softmax_6], Original ATen: [aten.gt, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_div_gt_masked_fill_17.run(buf175, 122880, 15, grid=grid(122880), stream=stream0)
        buf176 = reinterpret_tensor(buf168, (4096, 2, 15, 5), (150, 75, 5, 1), 0); del buf168  # reuse
        # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf169, primals_42, buf176, 614400, grid=grid(614400), stream=stream0)
        buf177 = reinterpret_tensor(buf169, (8192, 15, 5), (75, 5, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf175, (8192, 15, 15), (225, 15, 1), 0), reinterpret_tensor(buf176, (8192, 15, 5), (75, 5, 1), 0), out=buf177)
        buf178 = empty_strided_cuda((4096, 15, 2, 5), (150, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf177, buf178, 614400, grid=grid(614400), stream=stream0)
        buf179 = reinterpret_tensor(buf177, (61440, 10), (10, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf178, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_43, (10, 10), (1, 10), 0), out=buf179)
        buf180 = buf153; del buf153  # reuse
        buf184 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf185 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf457 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_11, x_5, layer_norm_13], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf180, buf161, primals_66, buf179, primals_44, primals_45, primals_46, buf184, buf185, buf457, 61440, 10, grid=grid(61440), stream=stream0)
        buf186 = empty_strided_cuda((61440, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_48, reinterpret_tensor(buf185, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_47, (10, 4), (1, 10), 0), alpha=1, beta=1, out=buf186)
        buf187 = empty_strided_cuda((4096, 15, 4), (60, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_19.run(buf186, buf187, 245760, grid=grid(245760), stream=stream0)
        buf188 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf187, (61440, 4), (4, 1), 0), reinterpret_tensor(primals_49, (4, 10), (1, 4), 0), out=buf188)
        buf192 = reinterpret_tensor(buf161, (4096, 15, 10), (150, 10, 1), 0); del buf161  # reuse
        buf193 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf456 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_13, z_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20.run(buf180, buf188, primals_50, primals_51, primals_52, buf192, buf193, buf456, 61440, 10, grid=grid(61440), stream=stream0)
        buf194 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf193, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_53, (10, 10), (1, 10), 0), out=buf194)
        buf195 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf193, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_55, (10, 10), (1, 10), 0), out=buf195)
        buf196 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf193, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_57, (10, 10), (1, 10), 0), out=buf196)
        buf197 = empty_strided_cuda((4096, 2, 15, 5), (150, 75, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_14], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf194, primals_54, buf197, 614400, grid=grid(614400), stream=stream0)
        buf198 = reinterpret_tensor(buf194, (4096, 2, 5, 15), (150, 75, 15, 1), 0); del buf194  # reuse
        # Topologically Sorted Source Nodes: [matmul_14], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_16.run(buf195, primals_56, buf198, 40960, 15, grid=grid(40960, 15), stream=stream0)
        buf199 = empty_strided_cuda((8192, 15, 15), (225, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf197, (8192, 15, 5), (75, 5, 1), 0), reinterpret_tensor(buf198, (8192, 5, 15), (75, 15, 1), 0), out=buf199)
        buf202 = reinterpret_tensor(buf199, (4096, 2, 15, 15), (450, 225, 15, 1), 0); del buf199  # reuse
        # Topologically Sorted Source Nodes: [gt, s_3, s_12, s_13, softmax_7], Original ATen: [aten.gt, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_div_gt_masked_fill_17.run(buf202, 122880, 15, grid=grid(122880), stream=stream0)
        buf203 = reinterpret_tensor(buf195, (4096, 2, 15, 5), (150, 75, 5, 1), 0); del buf195  # reuse
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf196, primals_58, buf203, 614400, grid=grid(614400), stream=stream0)
        buf204 = reinterpret_tensor(buf196, (8192, 15, 5), (75, 5, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf202, (8192, 15, 15), (225, 15, 1), 0), reinterpret_tensor(buf203, (8192, 15, 5), (75, 5, 1), 0), out=buf204)
        buf205 = empty_strided_cuda((4096, 15, 2, 5), (150, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_7], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf204, buf205, 614400, grid=grid(614400), stream=stream0)
        buf206 = reinterpret_tensor(buf204, (61440, 10), (10, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf205, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_59, (10, 10), (1, 10), 0), out=buf206)
        buf207 = buf180; del buf180  # reuse
        buf211 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf212 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf455 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_13, x_6, layer_norm_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf207, buf188, primals_50, buf206, primals_60, primals_61, primals_62, buf211, buf212, buf455, 61440, 10, grid=grid(61440), stream=stream0)
        buf213 = empty_strided_cuda((61440, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_64, reinterpret_tensor(buf212, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_63, (10, 4), (1, 10), 0), alpha=1, beta=1, out=buf213)
        buf214 = empty_strided_cuda((4096, 15, 4), (60, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_19.run(buf213, buf214, 245760, grid=grid(245760), stream=stream0)
        buf215 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf214, (61440, 4), (4, 1), 0), reinterpret_tensor(primals_65, (4, 10), (1, 4), 0), out=buf215)
        buf219 = reinterpret_tensor(buf188, (4096, 15, 10), (150, 10, 1), 0); del buf188  # reuse
        buf220 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf454 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_15, z_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20.run(buf207, buf215, primals_66, primals_35, primals_36, buf219, buf220, buf454, 61440, 10, grid=grid(61440), stream=stream0)
        buf221 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf220, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_37, (10, 10), (1, 10), 0), out=buf221)
        buf222 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf220, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_39, (10, 10), (1, 10), 0), out=buf222)
        buf223 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf220, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_41, (10, 10), (1, 10), 0), out=buf223)
        buf224 = empty_strided_cuda((4096, 2, 15, 5), (150, 75, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_16], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf221, primals_38, buf224, 614400, grid=grid(614400), stream=stream0)
        buf225 = reinterpret_tensor(buf221, (4096, 2, 5, 15), (150, 75, 15, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [matmul_16], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_16.run(buf222, primals_40, buf225, 40960, 15, grid=grid(40960, 15), stream=stream0)
        buf226 = empty_strided_cuda((8192, 15, 15), (225, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf224, (8192, 15, 5), (75, 5, 1), 0), reinterpret_tensor(buf225, (8192, 5, 15), (75, 15, 1), 0), out=buf226)
        buf229 = reinterpret_tensor(buf226, (4096, 2, 15, 15), (450, 225, 15, 1), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [gt, s_3, s_14, s_15, softmax_8], Original ATen: [aten.gt, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_div_gt_masked_fill_17.run(buf229, 122880, 15, grid=grid(122880), stream=stream0)
        buf230 = reinterpret_tensor(buf222, (4096, 2, 15, 5), (150, 75, 5, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [matmul_17], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf223, primals_42, buf230, 614400, grid=grid(614400), stream=stream0)
        buf231 = reinterpret_tensor(buf223, (8192, 15, 5), (75, 5, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf229, (8192, 15, 15), (225, 15, 1), 0), reinterpret_tensor(buf230, (8192, 15, 5), (75, 5, 1), 0), out=buf231)
        buf232 = empty_strided_cuda((4096, 15, 2, 5), (150, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_8], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf231, buf232, 614400, grid=grid(614400), stream=stream0)
        buf233 = reinterpret_tensor(buf231, (61440, 10), (10, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf232, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_43, (10, 10), (1, 10), 0), out=buf233)
        buf234 = buf207; del buf207  # reuse
        buf238 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf239 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf453 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_15, x_7, layer_norm_17], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf234, buf215, primals_66, buf233, primals_44, primals_45, primals_46, buf238, buf239, buf453, 61440, 10, grid=grid(61440), stream=stream0)
        buf240 = empty_strided_cuda((61440, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_48, reinterpret_tensor(buf239, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_47, (10, 4), (1, 10), 0), alpha=1, beta=1, out=buf240)
        buf241 = empty_strided_cuda((4096, 15, 4), (60, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_19.run(buf240, buf241, 245760, grid=grid(245760), stream=stream0)
        buf242 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf241, (61440, 4), (4, 1), 0), reinterpret_tensor(primals_49, (4, 10), (1, 4), 0), out=buf242)
        buf246 = reinterpret_tensor(buf215, (4096, 15, 10), (150, 10, 1), 0); del buf215  # reuse
        buf247 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf452 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_17, z_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20.run(buf234, buf242, primals_50, primals_51, primals_52, buf246, buf247, buf452, 61440, 10, grid=grid(61440), stream=stream0)
        buf248 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_52], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf247, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_53, (10, 10), (1, 10), 0), out=buf248)
        buf249 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_53], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf247, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_55, (10, 10), (1, 10), 0), out=buf249)
        buf250 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf247, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_57, (10, 10), (1, 10), 0), out=buf250)
        buf251 = empty_strided_cuda((4096, 2, 15, 5), (150, 75, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_18], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf248, primals_54, buf251, 614400, grid=grid(614400), stream=stream0)
        buf252 = reinterpret_tensor(buf248, (4096, 2, 5, 15), (150, 75, 15, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [matmul_18], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_16.run(buf249, primals_56, buf252, 40960, 15, grid=grid(40960, 15), stream=stream0)
        buf253 = empty_strided_cuda((8192, 15, 15), (225, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf251, (8192, 15, 5), (75, 5, 1), 0), reinterpret_tensor(buf252, (8192, 5, 15), (75, 15, 1), 0), out=buf253)
        buf256 = reinterpret_tensor(buf253, (4096, 2, 15, 15), (450, 225, 15, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [gt, s_3, s_16, s_17, softmax_9], Original ATen: [aten.gt, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_div_gt_masked_fill_17.run(buf256, 122880, 15, grid=grid(122880), stream=stream0)
        buf257 = reinterpret_tensor(buf249, (4096, 2, 15, 5), (150, 75, 5, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [matmul_19], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf250, primals_58, buf257, 614400, grid=grid(614400), stream=stream0)
        buf258 = reinterpret_tensor(buf250, (8192, 15, 5), (75, 5, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf256, (8192, 15, 15), (225, 15, 1), 0), reinterpret_tensor(buf257, (8192, 15, 5), (75, 5, 1), 0), out=buf258)
        buf259 = empty_strided_cuda((4096, 15, 2, 5), (150, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_9], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf258, buf259, 614400, grid=grid(614400), stream=stream0)
        buf260 = reinterpret_tensor(buf258, (61440, 10), (10, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf259, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_59, (10, 10), (1, 10), 0), out=buf260)
        buf261 = buf234; del buf234  # reuse
        buf265 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf266 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf451 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_17, x_8, layer_norm_19], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf261, buf242, primals_50, buf260, primals_60, primals_61, primals_62, buf265, buf266, buf451, 61440, 10, grid=grid(61440), stream=stream0)
        buf267 = empty_strided_cuda((61440, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_64, reinterpret_tensor(buf266, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_63, (10, 4), (1, 10), 0), alpha=1, beta=1, out=buf267)
        buf268 = empty_strided_cuda((4096, 15, 4), (60, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_19.run(buf267, buf268, 245760, grid=grid(245760), stream=stream0)
        buf269 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf268, (61440, 4), (4, 1), 0), reinterpret_tensor(primals_65, (4, 10), (1, 4), 0), out=buf269)
        buf273 = reinterpret_tensor(buf242, (4096, 15, 10), (150, 10, 1), 0); del buf242  # reuse
        buf274 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf450 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_19, z_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20.run(buf261, buf269, primals_66, primals_35, primals_36, buf273, buf274, buf450, 61440, 10, grid=grid(61440), stream=stream0)
        buf275 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_58], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf274, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_37, (10, 10), (1, 10), 0), out=buf275)
        buf276 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_59], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf274, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_39, (10, 10), (1, 10), 0), out=buf276)
        buf277 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf274, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_41, (10, 10), (1, 10), 0), out=buf277)
        buf278 = empty_strided_cuda((4096, 2, 15, 5), (150, 75, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_20], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf275, primals_38, buf278, 614400, grid=grid(614400), stream=stream0)
        buf279 = reinterpret_tensor(buf275, (4096, 2, 5, 15), (150, 75, 15, 1), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [matmul_20], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_16.run(buf276, primals_40, buf279, 40960, 15, grid=grid(40960, 15), stream=stream0)
        buf280 = empty_strided_cuda((8192, 15, 15), (225, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf278, (8192, 15, 5), (75, 5, 1), 0), reinterpret_tensor(buf279, (8192, 5, 15), (75, 15, 1), 0), out=buf280)
        buf283 = reinterpret_tensor(buf280, (4096, 2, 15, 15), (450, 225, 15, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [gt, s_3, s_18, s_19, softmax_10], Original ATen: [aten.gt, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_div_gt_masked_fill_17.run(buf283, 122880, 15, grid=grid(122880), stream=stream0)
        buf284 = reinterpret_tensor(buf276, (4096, 2, 15, 5), (150, 75, 5, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [matmul_21], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf277, primals_42, buf284, 614400, grid=grid(614400), stream=stream0)
        buf285 = reinterpret_tensor(buf277, (8192, 15, 5), (75, 5, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf283, (8192, 15, 15), (225, 15, 1), 0), reinterpret_tensor(buf284, (8192, 15, 5), (75, 5, 1), 0), out=buf285)
        buf286 = empty_strided_cuda((4096, 15, 2, 5), (150, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_10], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf285, buf286, 614400, grid=grid(614400), stream=stream0)
        buf287 = reinterpret_tensor(buf285, (61440, 10), (10, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf286, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_43, (10, 10), (1, 10), 0), out=buf287)
        buf288 = buf261; del buf261  # reuse
        buf292 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf293 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf449 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_19, x_9, layer_norm_21], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf288, buf269, primals_66, buf287, primals_44, primals_45, primals_46, buf292, buf293, buf449, 61440, 10, grid=grid(61440), stream=stream0)
        buf294 = empty_strided_cuda((61440, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_48, reinterpret_tensor(buf293, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_47, (10, 4), (1, 10), 0), alpha=1, beta=1, out=buf294)
        buf295 = empty_strided_cuda((4096, 15, 4), (60, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_19.run(buf294, buf295, 245760, grid=grid(245760), stream=stream0)
        buf296 = buf287; del buf287  # reuse
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf295, (61440, 4), (4, 1), 0), reinterpret_tensor(primals_49, (4, 10), (1, 4), 0), out=buf296)
        buf300 = reinterpret_tensor(buf269, (4096, 15, 10), (150, 10, 1), 0); del buf269  # reuse
        buf301 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf448 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_21, z_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20.run(buf288, buf296, primals_50, primals_51, primals_52, buf300, buf301, buf448, 61440, 10, grid=grid(61440), stream=stream0)
        buf302 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_64], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf301, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_53, (10, 10), (1, 10), 0), out=buf302)
        buf303 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_65], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf301, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_55, (10, 10), (1, 10), 0), out=buf303)
        buf304 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_66], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf301, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_57, (10, 10), (1, 10), 0), out=buf304)
        buf305 = empty_strided_cuda((4096, 2, 15, 5), (150, 75, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_22], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf302, primals_54, buf305, 614400, grid=grid(614400), stream=stream0)
        buf306 = reinterpret_tensor(buf302, (4096, 2, 5, 15), (150, 75, 15, 1), 0); del buf302  # reuse
        # Topologically Sorted Source Nodes: [matmul_22], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_16.run(buf303, primals_56, buf306, 40960, 15, grid=grid(40960, 15), stream=stream0)
        buf307 = empty_strided_cuda((8192, 15, 15), (225, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf305, (8192, 15, 5), (75, 5, 1), 0), reinterpret_tensor(buf306, (8192, 5, 15), (75, 15, 1), 0), out=buf307)
        buf310 = reinterpret_tensor(buf307, (4096, 2, 15, 15), (450, 225, 15, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [gt, s_3, s_20, s_21, softmax_11], Original ATen: [aten.gt, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_div_gt_masked_fill_17.run(buf310, 122880, 15, grid=grid(122880), stream=stream0)
        buf311 = reinterpret_tensor(buf303, (4096, 2, 15, 5), (150, 75, 5, 1), 0); del buf303  # reuse
        # Topologically Sorted Source Nodes: [matmul_23], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf304, primals_58, buf311, 614400, grid=grid(614400), stream=stream0)
        buf312 = reinterpret_tensor(buf304, (8192, 15, 5), (75, 5, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf310, (8192, 15, 15), (225, 15, 1), 0), reinterpret_tensor(buf311, (8192, 15, 5), (75, 5, 1), 0), out=buf312)
        buf313 = empty_strided_cuda((4096, 15, 2, 5), (150, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_11], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf312, buf313, 614400, grid=grid(614400), stream=stream0)
        buf314 = reinterpret_tensor(buf312, (61440, 10), (10, 1), 0); del buf312  # reuse
        # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf313, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_59, (10, 10), (1, 10), 0), out=buf314)
        buf315 = buf288; del buf288  # reuse
        buf319 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf320 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf447 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_21, x_10, layer_norm_23], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf315, buf296, primals_50, buf314, primals_60, primals_61, primals_62, buf319, buf320, buf447, 61440, 10, grid=grid(61440), stream=stream0)
        buf321 = empty_strided_cuda((61440, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_64, reinterpret_tensor(buf320, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_63, (10, 4), (1, 10), 0), alpha=1, beta=1, out=buf321)
        buf322 = empty_strided_cuda((4096, 15, 4), (60, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_19.run(buf321, buf322, 245760, grid=grid(245760), stream=stream0)
        buf323 = buf314; del buf314  # reuse
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf322, (61440, 4), (4, 1), 0), reinterpret_tensor(primals_65, (4, 10), (1, 4), 0), out=buf323)
        buf327 = reinterpret_tensor(buf296, (4096, 15, 10), (150, 10, 1), 0); del buf296  # reuse
        buf328 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf446 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_23, z_12], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20.run(buf315, buf323, primals_66, primals_35, primals_36, buf327, buf328, buf446, 61440, 10, grid=grid(61440), stream=stream0)
        buf329 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf328, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_37, (10, 10), (1, 10), 0), out=buf329)
        buf330 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_71], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf328, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_39, (10, 10), (1, 10), 0), out=buf330)
        buf331 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_72], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf328, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_41, (10, 10), (1, 10), 0), out=buf331)
        buf332 = empty_strided_cuda((4096, 2, 15, 5), (150, 75, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_24], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf329, primals_38, buf332, 614400, grid=grid(614400), stream=stream0)
        buf333 = reinterpret_tensor(buf329, (4096, 2, 5, 15), (150, 75, 15, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [matmul_24], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_16.run(buf330, primals_40, buf333, 40960, 15, grid=grid(40960, 15), stream=stream0)
        buf334 = empty_strided_cuda((8192, 15, 15), (225, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf332, (8192, 15, 5), (75, 5, 1), 0), reinterpret_tensor(buf333, (8192, 5, 15), (75, 15, 1), 0), out=buf334)
        buf337 = reinterpret_tensor(buf334, (4096, 2, 15, 15), (450, 225, 15, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [gt, s_3, s_22, s_23, softmax_12], Original ATen: [aten.gt, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_div_gt_masked_fill_17.run(buf337, 122880, 15, grid=grid(122880), stream=stream0)
        buf338 = reinterpret_tensor(buf330, (4096, 2, 15, 5), (150, 75, 5, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [matmul_25], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf331, primals_42, buf338, 614400, grid=grid(614400), stream=stream0)
        buf339 = reinterpret_tensor(buf331, (8192, 15, 5), (75, 5, 1), 0); del buf331  # reuse
        # Topologically Sorted Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf337, (8192, 15, 15), (225, 15, 1), 0), reinterpret_tensor(buf338, (8192, 15, 5), (75, 5, 1), 0), out=buf339)
        buf340 = empty_strided_cuda((4096, 15, 2, 5), (150, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_12], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf339, buf340, 614400, grid=grid(614400), stream=stream0)
        buf341 = reinterpret_tensor(buf339, (61440, 10), (10, 1), 0); del buf339  # reuse
        # Topologically Sorted Source Nodes: [linear_73], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf340, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_43, (10, 10), (1, 10), 0), out=buf341)
        buf342 = buf315; del buf315  # reuse
        buf346 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf347 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf445 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_23, x_11, layer_norm_25], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf342, buf323, primals_66, buf341, primals_44, primals_45, primals_46, buf346, buf347, buf445, 61440, 10, grid=grid(61440), stream=stream0)
        buf348 = empty_strided_cuda((61440, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_48, reinterpret_tensor(buf347, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_47, (10, 4), (1, 10), 0), alpha=1, beta=1, out=buf348)
        buf349 = empty_strided_cuda((4096, 15, 4), (60, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_19.run(buf348, buf349, 245760, grid=grid(245760), stream=stream0)
        buf350 = buf341; del buf341  # reuse
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf349, (61440, 4), (4, 1), 0), reinterpret_tensor(primals_49, (4, 10), (1, 4), 0), out=buf350)
        buf354 = reinterpret_tensor(buf323, (4096, 15, 10), (150, 10, 1), 0); del buf323  # reuse
        buf355 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf444 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_25, z_13], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20.run(buf342, buf350, primals_50, primals_51, primals_52, buf354, buf355, buf444, 61440, 10, grid=grid(61440), stream=stream0)
        buf356 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_76], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf355, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_53, (10, 10), (1, 10), 0), out=buf356)
        buf357 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_77], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf355, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_55, (10, 10), (1, 10), 0), out=buf357)
        buf358 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_78], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf355, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_57, (10, 10), (1, 10), 0), out=buf358)
        buf359 = empty_strided_cuda((4096, 2, 15, 5), (150, 75, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_26], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf356, primals_54, buf359, 614400, grid=grid(614400), stream=stream0)
        buf360 = reinterpret_tensor(buf356, (4096, 2, 5, 15), (150, 75, 15, 1), 0); del buf356  # reuse
        # Topologically Sorted Source Nodes: [matmul_26], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_16.run(buf357, primals_56, buf360, 40960, 15, grid=grid(40960, 15), stream=stream0)
        buf361 = empty_strided_cuda((8192, 15, 15), (225, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf359, (8192, 15, 5), (75, 5, 1), 0), reinterpret_tensor(buf360, (8192, 5, 15), (75, 15, 1), 0), out=buf361)
        buf364 = reinterpret_tensor(buf361, (4096, 2, 15, 15), (450, 225, 15, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [gt, s_3, s_24, s_25, softmax_13], Original ATen: [aten.gt, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_div_gt_masked_fill_17.run(buf364, 122880, 15, grid=grid(122880), stream=stream0)
        buf365 = reinterpret_tensor(buf357, (4096, 2, 15, 5), (150, 75, 5, 1), 0); del buf357  # reuse
        # Topologically Sorted Source Nodes: [matmul_27], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf358, primals_58, buf365, 614400, grid=grid(614400), stream=stream0)
        buf366 = reinterpret_tensor(buf358, (8192, 15, 5), (75, 5, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf364, (8192, 15, 15), (225, 15, 1), 0), reinterpret_tensor(buf365, (8192, 15, 5), (75, 5, 1), 0), out=buf366)
        buf367 = empty_strided_cuda((4096, 15, 2, 5), (150, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_13], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf366, buf367, 614400, grid=grid(614400), stream=stream0)
        buf368 = reinterpret_tensor(buf366, (61440, 10), (10, 1), 0); del buf366  # reuse
        # Topologically Sorted Source Nodes: [linear_79], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf367, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_59, (10, 10), (1, 10), 0), out=buf368)
        buf369 = buf342; del buf342  # reuse
        buf373 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf374 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf443 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_25, x_12, layer_norm_27], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf369, buf350, primals_50, buf368, primals_60, primals_61, primals_62, buf373, buf374, buf443, 61440, 10, grid=grid(61440), stream=stream0)
        buf375 = empty_strided_cuda((61440, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_64, reinterpret_tensor(buf374, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_63, (10, 4), (1, 10), 0), alpha=1, beta=1, out=buf375)
        buf376 = empty_strided_cuda((4096, 15, 4), (60, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_19.run(buf375, buf376, 245760, grid=grid(245760), stream=stream0)
        buf377 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf376, (61440, 4), (4, 1), 0), reinterpret_tensor(primals_65, (4, 10), (1, 4), 0), out=buf377)
        buf381 = reinterpret_tensor(buf350, (4096, 15, 10), (150, 10, 1), 0); del buf350  # reuse
        buf382 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf442 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_27, z_14], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20.run(buf369, buf377, primals_66, primals_35, primals_36, buf381, buf382, buf442, 61440, 10, grid=grid(61440), stream=stream0)
        del primals_36
        buf383 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf382, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_37, (10, 10), (1, 10), 0), out=buf383)
        buf384 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_83], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf382, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_39, (10, 10), (1, 10), 0), out=buf384)
        buf385 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_84], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf382, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_41, (10, 10), (1, 10), 0), out=buf385)
        buf386 = empty_strided_cuda((4096, 2, 15, 5), (150, 75, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_28], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf383, primals_38, buf386, 614400, grid=grid(614400), stream=stream0)
        del primals_38
        buf387 = reinterpret_tensor(buf383, (4096, 2, 5, 15), (150, 75, 15, 1), 0); del buf383  # reuse
        # Topologically Sorted Source Nodes: [matmul_28], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_16.run(buf384, primals_40, buf387, 40960, 15, grid=grid(40960, 15), stream=stream0)
        del primals_40
        buf388 = empty_strided_cuda((8192, 15, 15), (225, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf386, (8192, 15, 5), (75, 5, 1), 0), reinterpret_tensor(buf387, (8192, 5, 15), (75, 15, 1), 0), out=buf388)
        buf391 = reinterpret_tensor(buf388, (4096, 2, 15, 15), (450, 225, 15, 1), 0); del buf388  # reuse
        # Topologically Sorted Source Nodes: [gt, s_3, s_26, s_27, softmax_14], Original ATen: [aten.gt, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_div_gt_masked_fill_17.run(buf391, 122880, 15, grid=grid(122880), stream=stream0)
        buf392 = reinterpret_tensor(buf384, (4096, 2, 15, 5), (150, 75, 5, 1), 0); del buf384  # reuse
        # Topologically Sorted Source Nodes: [matmul_29], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf385, primals_42, buf392, 614400, grid=grid(614400), stream=stream0)
        del primals_42
        buf393 = reinterpret_tensor(buf385, (8192, 15, 5), (75, 5, 1), 0); del buf385  # reuse
        # Topologically Sorted Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf391, (8192, 15, 15), (225, 15, 1), 0), reinterpret_tensor(buf392, (8192, 15, 5), (75, 5, 1), 0), out=buf393)
        buf394 = empty_strided_cuda((4096, 15, 2, 5), (150, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_14], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf393, buf394, 614400, grid=grid(614400), stream=stream0)
        buf395 = reinterpret_tensor(buf393, (61440, 10), (10, 1), 0); del buf393  # reuse
        # Topologically Sorted Source Nodes: [linear_85], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf394, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_43, (10, 10), (1, 10), 0), out=buf395)
        buf396 = buf369; del buf369  # reuse
        buf400 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf401 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf441 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_27, x_13, layer_norm_29], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf396, buf377, primals_66, buf395, primals_44, primals_45, primals_46, buf400, buf401, buf441, 61440, 10, grid=grid(61440), stream=stream0)
        del primals_44
        del primals_46
        buf402 = empty_strided_cuda((61440, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_48, reinterpret_tensor(buf401, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_47, (10, 4), (1, 10), 0), alpha=1, beta=1, out=buf402)
        del primals_48
        buf403 = empty_strided_cuda((4096, 15, 4), (60, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_19.run(buf402, buf403, 245760, grid=grid(245760), stream=stream0)
        buf404 = buf395; del buf395  # reuse
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf403, (61440, 4), (4, 1), 0), reinterpret_tensor(primals_49, (4, 10), (1, 4), 0), out=buf404)
        buf408 = reinterpret_tensor(buf377, (4096, 15, 10), (150, 10, 1), 0); del buf377  # reuse
        buf409 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf440 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_29, z_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20.run(buf396, buf404, primals_50, primals_51, primals_52, buf408, buf409, buf440, 61440, 10, grid=grid(61440), stream=stream0)
        del primals_52
        buf410 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_88], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf409, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_53, (10, 10), (1, 10), 0), out=buf410)
        buf411 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_89], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf409, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_55, (10, 10), (1, 10), 0), out=buf411)
        buf412 = empty_strided_cuda((61440, 10), (10, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_90], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf409, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_57, (10, 10), (1, 10), 0), out=buf412)
        buf413 = empty_strided_cuda((4096, 2, 15, 5), (150, 75, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_30], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf410, primals_54, buf413, 614400, grid=grid(614400), stream=stream0)
        del primals_54
        buf414 = reinterpret_tensor(buf410, (4096, 2, 5, 15), (150, 75, 15, 1), 0); del buf410  # reuse
        # Topologically Sorted Source Nodes: [matmul_30], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_16.run(buf411, primals_56, buf414, 40960, 15, grid=grid(40960, 15), stream=stream0)
        del primals_56
        buf415 = empty_strided_cuda((8192, 15, 15), (225, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf413, (8192, 15, 5), (75, 5, 1), 0), reinterpret_tensor(buf414, (8192, 5, 15), (75, 15, 1), 0), out=buf415)
        buf418 = reinterpret_tensor(buf415, (4096, 2, 15, 15), (450, 225, 15, 1), 0); del buf415  # reuse
        # Topologically Sorted Source Nodes: [gt, s_3, s_28, s_29, softmax_15], Original ATen: [aten.gt, aten.masked_fill, aten.div, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_div_gt_masked_fill_17.run(buf418, 122880, 15, grid=grid(122880), stream=stream0)
        buf419 = reinterpret_tensor(buf411, (4096, 2, 15, 5), (150, 75, 5, 1), 0); del buf411  # reuse
        # Topologically Sorted Source Nodes: [matmul_31], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf412, primals_58, buf419, 614400, grid=grid(614400), stream=stream0)
        del primals_58
        buf420 = reinterpret_tensor(buf412, (8192, 15, 5), (75, 5, 1), 0); del buf412  # reuse
        # Topologically Sorted Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf418, (8192, 15, 15), (225, 15, 1), 0), reinterpret_tensor(buf419, (8192, 15, 5), (75, 5, 1), 0), out=buf420)
        buf421 = empty_strided_cuda((4096, 15, 2, 5), (150, 10, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape_15], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_13.run(buf420, buf421, 614400, grid=grid(614400), stream=stream0)
        buf422 = reinterpret_tensor(buf420, (61440, 10), (10, 1), 0); del buf420  # reuse
        # Topologically Sorted Source Nodes: [linear_91], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf421, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_59, (10, 10), (1, 10), 0), out=buf422)
        buf423 = buf396; del buf396  # reuse
        buf427 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf428 = empty_strided_cuda((4096, 15, 10), (150, 10, 1), torch.float32)
        buf439 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_29, x_14, layer_norm_31], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf423, buf404, primals_50, buf422, primals_60, primals_61, primals_62, buf427, buf428, buf439, 61440, 10, grid=grid(61440), stream=stream0)
        del primals_50
        del primals_60
        del primals_62
        buf429 = empty_strided_cuda((61440, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_64, reinterpret_tensor(buf428, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_63, (10, 4), (1, 10), 0), alpha=1, beta=1, out=buf429)
        del primals_64
        buf430 = empty_strided_cuda((4096, 15, 4), (60, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_19.run(buf429, buf430, 245760, grid=grid(245760), stream=stream0)
        buf431 = buf422; del buf422  # reuse
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf430, (61440, 4), (4, 1), 0), reinterpret_tensor(primals_65, (4, 10), (1, 4), 0), out=buf431)
        buf435 = buf423; del buf423  # reuse
        buf436 = reinterpret_tensor(buf404, (4096, 15, 10), (150, 10, 1), 0); del buf404  # reuse
        buf438 = empty_strided_cuda((4096, 15, 1), (15, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q_31, layer_norm_32], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_22.run(buf435, buf431, primals_66, primals_67, primals_68, buf436, buf438, 61440, 10, grid=grid(61440), stream=stream0)
        del primals_66
        del primals_68
        buf437 = buf431; del buf431  # reuse
        # Topologically Sorted Source Nodes: [linear_94], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_70, reinterpret_tensor(buf436, (61440, 10), (10, 1), 0), reinterpret_tensor(primals_69, (10, 10), (1, 10), 0), alpha=1, beta=1, out=buf437)
        del primals_70
    return (reinterpret_tensor(buf437, (4096, 15, 10), (150, 10, 1), 0), primals_2, primals_4, primals_6, primals_16, primals_23, primals_25, primals_35, primals_45, primals_51, primals_61, primals_67, buf4, reinterpret_tensor(buf5, (57344, 10), (10, 1), 0), buf15, reinterpret_tensor(buf16, (8192, 5, 14), (70, 1, 5), 0), reinterpret_tensor(buf9, (8192, 5, 14), (70, 1, 5), 0), reinterpret_tensor(buf10, (8192, 14, 5), (70, 1, 14), 0), reinterpret_tensor(buf18, (57344, 10), (10, 1), 0), buf23, reinterpret_tensor(buf24, (57344, 10), (10, 1), 0), buf25, reinterpret_tensor(buf26, (57344, 4), (4, 1), 0), buf29, buf30, buf33, buf37, reinterpret_tensor(buf38, (61440, 10), (10, 1), 0), reinterpret_tensor(buf40, (57344, 10), (10, 1), 0), buf49, reinterpret_tensor(buf50, (8192, 5, 14), (70, 1, 5), 0), reinterpret_tensor(buf43, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf44, (8192, 14, 5), (70, 1, 14), 0), reinterpret_tensor(buf52, (61440, 10), (10, 1), 0), buf57, reinterpret_tensor(buf58, (61440, 10), (10, 1), 0), buf67, reinterpret_tensor(buf70, (61440, 10), (10, 1), 0), buf76, reinterpret_tensor(buf77, (61440, 10), (10, 1), 0), buf78, reinterpret_tensor(buf79, (61440, 4), (4, 1), 0), buf84, reinterpret_tensor(buf85, (61440, 10), (10, 1), 0), buf94, reinterpret_tensor(buf97, (61440, 10), (10, 1), 0), buf103, reinterpret_tensor(buf104, (61440, 10), (10, 1), 0), buf105, reinterpret_tensor(buf106, (61440, 4), (4, 1), 0), buf111, reinterpret_tensor(buf112, (61440, 10), (10, 1), 0), buf121, reinterpret_tensor(buf124, (61440, 10), (10, 1), 0), buf130, reinterpret_tensor(buf131, (61440, 10), (10, 1), 0), buf132, reinterpret_tensor(buf133, (61440, 4), (4, 1), 0), buf138, reinterpret_tensor(buf139, (61440, 10), (10, 1), 0), buf148, reinterpret_tensor(buf151, (61440, 10), (10, 1), 0), buf157, reinterpret_tensor(buf158, (61440, 10), (10, 1), 0), buf159, reinterpret_tensor(buf160, (61440, 4), (4, 1), 0), buf165, reinterpret_tensor(buf166, (61440, 10), (10, 1), 0), buf175, reinterpret_tensor(buf178, (61440, 10), (10, 1), 0), buf184, reinterpret_tensor(buf185, (61440, 10), (10, 1), 0), buf186, reinterpret_tensor(buf187, (61440, 4), (4, 1), 0), buf192, reinterpret_tensor(buf193, (61440, 10), (10, 1), 0), buf202, reinterpret_tensor(buf205, (61440, 10), (10, 1), 0), buf211, reinterpret_tensor(buf212, (61440, 10), (10, 1), 0), buf213, reinterpret_tensor(buf214, (61440, 4), (4, 1), 0), buf219, reinterpret_tensor(buf220, (61440, 10), (10, 1), 0), buf229, reinterpret_tensor(buf232, (61440, 10), (10, 1), 0), buf238, reinterpret_tensor(buf239, (61440, 10), (10, 1), 0), buf240, reinterpret_tensor(buf241, (61440, 4), (4, 1), 0), buf246, reinterpret_tensor(buf247, (61440, 10), (10, 1), 0), buf256, reinterpret_tensor(buf259, (61440, 10), (10, 1), 0), buf265, reinterpret_tensor(buf266, (61440, 10), (10, 1), 0), buf267, reinterpret_tensor(buf268, (61440, 4), (4, 1), 0), buf273, reinterpret_tensor(buf274, (61440, 10), (10, 1), 0), buf283, reinterpret_tensor(buf286, (61440, 10), (10, 1), 0), buf292, reinterpret_tensor(buf293, (61440, 10), (10, 1), 0), buf294, reinterpret_tensor(buf295, (61440, 4), (4, 1), 0), buf300, reinterpret_tensor(buf301, (61440, 10), (10, 1), 0), buf310, reinterpret_tensor(buf313, (61440, 10), (10, 1), 0), buf319, reinterpret_tensor(buf320, (61440, 10), (10, 1), 0), buf321, reinterpret_tensor(buf322, (61440, 4), (4, 1), 0), buf327, reinterpret_tensor(buf328, (61440, 10), (10, 1), 0), buf337, reinterpret_tensor(buf340, (61440, 10), (10, 1), 0), buf346, reinterpret_tensor(buf347, (61440, 10), (10, 1), 0), buf348, reinterpret_tensor(buf349, (61440, 4), (4, 1), 0), buf354, reinterpret_tensor(buf355, (61440, 10), (10, 1), 0), buf364, reinterpret_tensor(buf367, (61440, 10), (10, 1), 0), buf373, reinterpret_tensor(buf374, (61440, 10), (10, 1), 0), buf375, reinterpret_tensor(buf376, (61440, 4), (4, 1), 0), buf381, reinterpret_tensor(buf382, (61440, 10), (10, 1), 0), buf391, reinterpret_tensor(buf394, (61440, 10), (10, 1), 0), buf400, reinterpret_tensor(buf401, (61440, 10), (10, 1), 0), buf402, reinterpret_tensor(buf403, (61440, 4), (4, 1), 0), buf408, reinterpret_tensor(buf409, (61440, 10), (10, 1), 0), buf418, reinterpret_tensor(buf421, (61440, 10), (10, 1), 0), buf427, reinterpret_tensor(buf428, (61440, 10), (10, 1), 0), buf429, reinterpret_tensor(buf430, (61440, 4), (4, 1), 0), buf435, reinterpret_tensor(buf436, (61440, 10), (10, 1), 0), primals_69, buf438, primals_65, primals_63, buf439, primals_59, reinterpret_tensor(buf419, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf413, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf414, (8192, 15, 5), (75, 1, 15), 0), primals_57, primals_55, primals_53, buf440, primals_49, primals_47, buf441, primals_43, reinterpret_tensor(buf392, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf386, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf387, (8192, 15, 5), (75, 1, 15), 0), primals_41, primals_39, primals_37, buf442, buf443, reinterpret_tensor(buf365, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf359, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf360, (8192, 15, 5), (75, 1, 15), 0), buf444, buf445, reinterpret_tensor(buf338, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf332, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf333, (8192, 15, 5), (75, 1, 15), 0), buf446, buf447, reinterpret_tensor(buf311, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf305, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf306, (8192, 15, 5), (75, 1, 15), 0), buf448, buf449, reinterpret_tensor(buf284, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf278, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf279, (8192, 15, 5), (75, 1, 15), 0), buf450, buf451, reinterpret_tensor(buf257, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf251, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf252, (8192, 15, 5), (75, 1, 15), 0), buf452, buf453, reinterpret_tensor(buf230, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf224, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf225, (8192, 15, 5), (75, 1, 15), 0), buf454, buf455, reinterpret_tensor(buf203, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf197, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf198, (8192, 15, 5), (75, 1, 15), 0), buf456, buf457, reinterpret_tensor(buf176, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf170, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf171, (8192, 15, 5), (75, 1, 15), 0), buf458, buf459, reinterpret_tensor(buf149, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf143, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf144, (8192, 15, 5), (75, 1, 15), 0), buf460, buf461, reinterpret_tensor(buf122, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf116, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf117, (8192, 15, 5), (75, 1, 15), 0), buf462, buf463, reinterpret_tensor(buf95, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf89, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf90, (8192, 15, 5), (75, 1, 15), 0), buf464, buf465, reinterpret_tensor(buf68, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf62, (8192, 5, 15), (75, 1, 5), 0), reinterpret_tensor(buf63, (8192, 15, 5), (75, 1, 15), 0), buf466, primals_33, primals_31, primals_29, primals_27, buf467, primals_20, primals_18, buf468, primals_14, primals_12, primals_10, primals_8, buf469, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4096, 14), (14, 1), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4096, 14), (14, 1), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((14, 2), (2, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((10, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((15, 2), (2, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((4, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((10, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((4, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((10, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
