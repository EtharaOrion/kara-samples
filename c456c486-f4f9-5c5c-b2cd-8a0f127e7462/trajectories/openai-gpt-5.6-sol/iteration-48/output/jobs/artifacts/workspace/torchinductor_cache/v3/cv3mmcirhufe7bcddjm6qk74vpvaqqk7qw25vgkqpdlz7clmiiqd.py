
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
