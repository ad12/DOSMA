from enum import Enum
import os


class ImageVisFigureFormat(Enum):
    """
    Enum describing format to save visualization images (i.e. figures)
    """
    png=1
    eps=2
    pdf=3
    jpeg=4
    pgf=5
    ps=6
    raw=7
    rgba=8
    svg=9
    svgz=10
    tiff=11


    def get_filepath(self, filepath):
        filename, ext = os.path.splitext(filepath)
        return '%s.%s' % (filename, self.name)


SUPPORTED_VISUALIZATION_FORMATS = [x.name for x in ImageVisFigureFormat]


import defaults
import matplotlib.pyplot as plt


def savefig(filepath, fig_format=None, **kwargs):
    """
    Wrapper method for saving figures created using matplotlib
    :param filepath: the filepath to save image to
    :param fig_format: a ImageVisFigureFormat to save figure in
    :param kwargs: additional keyword arguments used for matplotlib savefig command
    """
    if not fig_format:
        fig_format = defaults.DEFAULT_FIG_FORMAT

    orig_fig_format = fig_format
    if type(fig_format) is not ImageVisFigureFormat:
        members = ImageVisFigureFormat.__members__
        for format_str in members.keys():
            id = members[format_str].value
            match_str = (type(fig_format) is str) and (fig_format.lower() == format_str.lower())
            match_id = (type(fig_format) is int) and (fig_format == id)
            if ((type(fig_format) is str) and (format_str.lower() == fig_format.lower())):
                if match_str or match_id:
                    fig_format = members[format_str]
                    break

    if type(fig_format) is not ImageVisFigureFormat:
        raise TypeError('fig_format should be of type [ImageVisFigureFormat, str, int]. %s did not match' % orig_fig_format)

    filepath = fig_format.get_filepath(filepath)

    plt.savefig(filepath, **kwargs)
