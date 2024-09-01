from __future__ import annotations

import warnings
from itertools import groupby
from typing import Iterable, Sequence, TypeVar

from vsrgtools import box_blur
from vstools import (CustomError, CustomNotImplementedError, CustomValueError,
                     FrameRangesN, FuncExceptT, PlanesT, Sentinel,
                     VSFunctionNoArgs, check_ref_clip, clip_async_render, core,
                     get_w, merge_clip_props, normalize_franges,
                     normalize_planes, vs)

from lvsfunc.dependency.plugin import check_installed_plugins

from .enum import DiffMode
from .props import _diff_planestats, _diff_vmaf_plugin
from .types import CallbacksT


class FindDiff:
    """Find the differences between two clips."""

    clip_a: vs.VideoNode
    """The first clip being compared."""

    clip_b: vs.VideoNode
    """The second clip being compared."""

    thresholds: list[float]
    """List of thresholds used for comparison, normalized as needed."""

    diff_modes: list[DiffMode]
    """List of difference modes used for comparison."""

    pre_process: VSFunctionNoArgs | bool
    """Pre-processing function or flag indicating whether to use default pre-processing."""

    exclusion_ranges: FrameRangesN
    """Ranges of frames to exclude from the comparison."""

    planes: PlanesT
    """Planes to consider in the comparison."""

    diff_clips: list[vs.VideoNode]
    """List of difference clips generated during comparison."""

    callbacks: CallbacksT
    """List of callback functions for each comparison method."""

    def __init__(
        self,
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        thresholds: float | Sequence[float] = 24,
        diff_modes: DiffMode | Sequence[DiffMode] = DiffMode.SSIM,
        pre_process: VSFunctionNoArgs | bool = True,
        exclusion_ranges: Sequence[int | tuple[int, int]] | None = None,
        planes: PlanesT = None,
        func_except: FuncExceptT | None = None
    ) -> None:
        """
        Find differences between two clips using various comparison methods.

        This class is useful for:
        - Identifying frames that differ significantly between two versions of a video
        - Detecting scene changes or cuts

        Example usage:

        .. code-block:: python

            from lvsfunc.diff import FindDiff
            from lvsfunc.enum import DiffMode
            import vapoursynth as vs

            # Assume clip_a and clip_b are your input clips
            diff_finder = FindDiff(
                clip_a, clip_b,
                thresholds=24,
                diff_modes=DiffMode.PLANESTATS_MINMAX,
                planes=[0]
            )

            # Get the ranges of frames that differ
            diff_ranges = diff_finder.get_ranges()

            # Get a processed clip highlighting the differences
            diff_clip = diff_finder.get_diff_clip()


        This example compares two clips using both SSIM and PSNR methods on the luma plane,
        with pre-processing enabled. It then provides ways to access the differing frames,
        frame ranges, and a processed clip showing the differences.

        :param clip_a:              The first clip to compare.
        :param clip_b:              The second clip to compare.
        :param thresholds:          The threshold(s) to use for the comparison.
                                    These will be normalized as necessary during processing.
                                    Default: 96.
        :param diff_modes:          The difference mode(s) to use for the comparison.
                                    For more information, please refer to the documentation
                                    of :py:class:`lvsfunc.enum.DiffMode`.
                                    Default: :py:attr:`lvsfunc.enum.DiffMode.SSIM`.
        :param pre_process:         The pre-processing function to use for the comparison.
                                    If True, use box_blur. If False, skip pre-processing.
                                    Default: True.
        :param exclusion_ranges:    Ranges to exclude from the comparison.
                                    These frames will still be processed, but not outputted.
        :param planes:              The planes to compare. Default: all planes.
        :param func_except:         The function exception to use for the comparison.
        """

        self._func_except = func_except or self.__class__.__name__

        self.clip_a = clip_a
        self.clip_b = clip_b

        self.thresholds = list(thresholds) if isinstance(thresholds, Sequence) else [thresholds]
        self.diff_modes = [diff_modes] if not isinstance(diff_modes, Sequence) else diff_modes

        if not self.thresholds:
            raise CustomValueError('You must pass a threshold!', self._func_except, self.thresholds)

        if not self.diff_modes:
            raise CustomValueError('You must pass a diff mode!', self._func_except, self.diff_modes)

        self.pre_process = pre_process if callable(pre_process) else (box_blur if pre_process is True else None)

        self.exclusion_ranges = exclusion_ranges or []

        self.planes = normalize_planes(clip_a, planes)

        self._diff_frames: list[int] | None = None
        self._diff_ranges: list[tuple[int, int]] | None = None
        self._processed_clip: vs.VideoNode | None = None

        self._validate_inputs()
        self._prepare_clips()

        self._process()

    def _validate_inputs(self) -> None:
        check_ref_clip(self.clip_a, self.clip_b, self._func_except)

        if self.clip_a.num_frames != self.clip_b.num_frames:
            warnings.warn(
                f"{self._func_except}: 'The number of frames of the clips don't match! "
                f"({self.clip_a.num_frames=}, {self.clip_b.num_frames=})\n"
                "The function will still work, but your clips may be synced incorrectly!'"
            )

            self.clip_a, self.clip_b = (self.clip_a[:self.clip_b.num_frames], self.clip_b) \
                if self.clip_a.num_frames > self.clip_b.num_frames \
                else (self.clip_a, self.clip_b[:self.clip_a.num_frames])

        if len(self.thresholds) != len(self.diff_modes):
            self.thresholds.extend([self.thresholds[-1]] * (len(self.diff_modes) - len(self.thresholds)))
            self.thresholds = self.thresholds[:len(self.diff_modes)]

    def _prepare_clips(self) -> None:
        if self.pre_process is True:
            self.pre_process = box_blur  # type: ignore

        if callable(self.pre_process):
            self.clip_a, self.clip_b = self.pre_process(self.clip_a), self.pre_process(self.clip_b)

    def _process(self) -> None:
        self._processed_clip = self.clip_a.std.SetFrameProps(fd_DiffModes=[str(x) for x in self.diff_modes])

        callbacks: CallbacksT = []

        for thr, metric in zip(self.thresholds, [DiffMode.from_param(x) for x in self.diff_modes]):
            if metric.is_planestats:
                self._processed_clip, cb = _diff_planestats(
                    self._processed_clip, self.clip_b, metric, thr, self.planes
                )
            elif metric.is_vmaf:
                self._processed_clip, cb = _diff_vmaf_plugin(
                    self._processed_clip, self.clip_b, metric, thr, self.planes
                )

            else:
                raise CustomNotImplementedError(f'Diff mode {metric} is not implemented!', self._func_except)

            callbacks += cb

        self._find_frames(callbacks)

    def _find_frames(self, callbacks: CallbacksT) -> None:
        """Get the frames that are different between two clips."""

        assert isinstance(self._processed_clip, vs.VideoNode)

        frames_render = clip_async_render(
            self._processed_clip, None, "Finding differences between clips...",
            lambda n, f: Sentinel.check(n, any(cb(f) for cb in callbacks))
        )

        self._diff_frames = list(Sentinel.filter(frames_render))

        if not self._diff_frames:
            raise CustomError['StopIteration']('No differences found!', self._func_except)  # type: ignore[index]

        self._diff_frames.sort()

        if self.exclusion_ranges:
            excluded = set(
                frame
                for range_ in normalize_franges(self.exclusion_ranges)
                for frame in range(range_[0], range_[-1] + 1)
            )

            self._diff_frames = [f for f in self._diff_frames if f not in excluded]

        self._diff_ranges = list(self._to_ranges(self._diff_frames))

    @staticmethod
    def _to_ranges(iterable: list[int]) -> Iterable[tuple[int, int]]:
        iterable = sorted(set(iterable))
        for _, group in groupby(enumerate(iterable), lambda t: t[1] - t[0]):
            groupl = list(group)
            yield groupl[0][1], groupl[-1][1]

    @property
    def diff_frames(self) -> list[int]:
        if self._diff_frames is None:
            self._process()

        return self._diff_frames or []

    @property
    def diff_ranges(self) -> list[tuple[int, int]]:
        if self._diff_ranges is None:
            self._process()

        return self._diff_ranges or []

    def get_diff_clip(self, height: int = 288, names: tuple[str, str] = ("Clip A", "Clip B")) -> vs.VideoNode:
        """
        Get a processed clip highlighting the differences between two clips.

        :param height:    The height of the output clip. Default: 288.
        :param names:     The names of the clips. Default: ("Clip A", "Clip B").

        :return:          A clip highlighting the differences between the two clips.
        """

        if not isinstance(names, tuple):
            names = (names, names)
        elif len(names) != 2:
            raise CustomValueError("Names must be a tuple of two strings!", self.get_diff_clip, names)

        scaled_width = get_w(height, mod=1)

        rzs = core.resize2.Bicubic if not check_installed_plugins('resize2', False) else core.resize.Bicubic

        diff_clip = core.std.MakeDiff(self.clip_a, self.clip_b)
        diff_clip = rzs(diff_clip, width=scaled_width * 2, height=height * 2).text.FrameNum(9)

        a, b = (
            rzs(c, width=scaled_width, height=height).text.FrameNum(9)
            for c in (self.clip_a, self.clip_b)
        )

        a = merge_clip_props(a, self._processed_clip)

        diff_stack = core.std.StackHorizontal([
            core.std.Splice([a[f] for f in self._diff_frames]).text.Text(names[0], 7),
            core.std.Splice([b[f] for f in self._diff_frames]).text.Text(names[1], 7),
        ])

        diff_clip = diff_clip.text.Text(text='Differences found:', alignment=8)
        diff_clip = core.std.Splice([diff_clip[f] for f in self._diff_frames])
        diff_clip = core.std.StackVertical([diff_stack, diff_clip])

        return diff_clip

    @classmethod
    def get_diff(
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        thresholds: float | Sequence[float] = 96,
        diff_modes: DiffMode | Sequence[DiffMode] = DiffMode.SSIM,
        pre_process: VSFunctionNoArgs | bool = True,
        exclusion_ranges: Sequence[int | tuple[int, int]] | None = None,
        planes: PlanesT = None,
        func_except: FuncExceptT | None = None
    ) -> vs.VideoNode:
        return FindDiff(
            clip_a, clip_b, thresholds, diff_modes, pre_process, exclusion_ranges, planes, func_except
        )._processed_clip


TFindDiff = TypeVar('TFindDiff', bound='FindDiff')
