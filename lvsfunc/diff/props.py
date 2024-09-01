from stgpytools import CustomNotImplementedError, FuncExceptT
from vstools import (PlanesT, depth, get_depth, get_neutral_value, get_prop,
                     merge_clip_props, normalize_planes, scale_value, vs)

from .enum import DiffMode
from .types import CallbacksT

__all__: list[str] = []


def _diff_planestats(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode,
    mode: DiffMode = DiffMode.PLANESTATS_MINMAX,
    thr: float | int = 96, planes: PlanesT = None,
    func: FuncExceptT | None = None
) -> tuple[vs.VideoNode, CallbacksT]:
    """Analyze the difference between two clips using PlaneStats."""

    bits = get_depth(clip_a)

    clip_a = depth(clip_a, 8)
    clip_b = depth(clip_b, 8)

    norm_planes = normalize_planes(clip_a, planes)

    def _set_ps_diff(n: int, f: vs.VideoFrame, clip: vs.VideoNode) -> vs.VideoNode:
        return clip.std.SetFrameProps(fd_psDiff=[get_prop(f, f'fd_ps{p}Diff', float) for p in norm_planes])

    def _set_ps_minmax(n: int, f: vs.VideoFrame, clip: vs.VideoNode) -> vs.VideoNode:

        ps_min = [get_prop(f, f'fd_ps{p}Min', int) for p in norm_planes]
        ps_max = [get_prop(f, f'fd_ps{p}Max', int) for p in norm_planes]

        ps_diff = [int(max(abs(neutral - min_val), abs(max_val - neutral))) for min_val, max_val in zip(ps_min, ps_max)]

        return clip.std.SetFrameProps(fd_psMin=ps_min, fd_psMax=ps_max, fd_psDiff=sum(ps_diff))

    callbacks: CallbacksT = []

    if mode is DiffMode.PLANESTATS_DIFF:
        if not 0 <= thr <= 1:
            thr = max(0, min(1, thr / 255))

        ps_diffs = merge_clip_props(*(clip_a.std.PlaneStats(clip_b, p, f'fd_ps{p}') for p in norm_planes))

        ps_comp = clip_a.std.FrameEval(lambda n, f: _set_ps_diff(n, f, clip_a), prop_src=ps_diffs)

        neutral = get_neutral_value(ps_comp)

        callbacks.append(lambda f: get_prop(f, 'fd_psDiff', float) >= neutral - thr)

        return ps_comp, callbacks
    elif mode is not DiffMode.PLANESTATS_MINMAX:
        raise CustomNotImplementedError('Mode is not supported!', func, mode)

    if not (0 <= thr <= 255):
        thr = scale_value(thr, bits, 8, scale_offsets=True)

    diff_clip = clip_a.std.MakeDiff(clip_b, norm_planes)
    neutral = get_neutral_value(diff_clip)

    ps_diffs = merge_clip_props(*(diff_clip.std.PlaneStats(None, p, f'fd_ps{p}') for p in norm_planes))
    ps_comp = diff_clip.std.FrameEval(lambda n, f: _set_ps_minmax(n, f, diff_clip), prop_src=ps_diffs)

    callbacks.append(lambda f: thr <= get_prop(f, 'fd_psDiff', int))

    return ps_comp.std.SetFrameProps(fd_psThr=thr), callbacks


def _diff_vmaf_plugin(  # type:ignore
    clip_a: vs.VideoNode, clip_b: vs.VideoNode,
    mode: DiffMode = DiffMode.PLANESTATS_MINMAX,
    thr: float | int = 96, planes: PlanesT = None,
    func: FuncExceptT | None = None
) -> tuple[vs.VideoNode, CallbacksT]:
    """Analyze the difference between two clips using the VMAF plugin."""

    ...
