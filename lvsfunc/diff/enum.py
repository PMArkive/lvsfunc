from vstools import CustomIntEnum

from .. import check_installed_plugins

__all__: list[str] = [
    'DiffMode',
]


class DiffMode(CustomIntEnum):
    """Different supported difference finding methods."""

    PLANESTATS_MINMAX = -2
    """Get the difference using min/max values of a MakeDiff clip."""

    PLANESTATS_DIFF = -1
    """Get the difference using PlaneStats diff prop."""

    PSNR = 0
    """"""

    PSNR_HVS = 1
    """"""

    SSIM = 2
    """"""

    MS_SSIM = 3
    """"""

    CIEDE2000 = 4
    """"""

    def __post_init__(self) -> None:
        if self.is_vmaf:
            check_installed_plugins('vmaf')

    def __str__(self) -> str:
        return str(self.name).replace('DiffMode.', '')

    @property
    def is_planestats(self) -> bool:
        return self < 0

    @property
    def is_vmaf(self) -> bool:
        return 0 <= self <= 4
