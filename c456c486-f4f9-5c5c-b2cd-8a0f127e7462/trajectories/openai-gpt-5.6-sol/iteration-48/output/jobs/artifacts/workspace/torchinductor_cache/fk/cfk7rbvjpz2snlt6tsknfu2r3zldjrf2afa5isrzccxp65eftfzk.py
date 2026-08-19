
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
