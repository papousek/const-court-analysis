from spiderpig import msg
from threading import RLock
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
import spiderpig as sp


_SNS_PALETTE = None
_LOCK = RLock()


def plot_line(data, color, with_confidence=True, xs=None, **kwargs):
    if xs is None:
        xs = list(range(1, len(data) + 1))
    plt.plot(
        xs,
        [y['value'] for y in data],
        color=color, markersize=5, **kwargs
    )
    if with_confidence:
        plt.fill_between(
            xs,
            [x['confidence_interval']['min'] for x in data],
            [x['confidence_interval']['max'] for x in data],
            color=color, alpha=0.35
        )
    plt.xlim(min(xs), max(xs) + 1)


def palette():
    global _SNS_PALETTE
    if _SNS_PALETTE is None:
        raise Exception("The palette is not initialized!")
    return _SNS_PALETTE


def init_plotting(palette=None, palette_name=None, font_scale=None, style='white'):
    with _LOCK:
        sns_kwargs = {'style': style}
        if font_scale is not None:
            sns_kwargs['font_scale'] = font_scale
        sns.set(**sns_kwargs)
        global _SNS_PALETTE
        if palette is None and palette_name is None:
            _SNS_PALETTE = sns.color_palette()
            return
        if palette is not None and palette_name is not None:
            raise Exception('Both palette itself or palette name can not be given.')
        if palette is not None:
            _SNS_PALETTE = palette
        if palette_name is not None:
            _SNS_PALETTE = sns.color_palette(palette_name)
        if _SNS_PALETTE:
            sns.set_palette(_SNS_PALETTE)


@sp.configured(cached=False)
def savefig(filename, output_dir=None, figure_extension=None, tight_layout=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if tight_layout:
        plt.tight_layout()
    filename = '{}/{}.{}'.format(output_dir, filename, figure_extension)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    msg.print_success(filename)
