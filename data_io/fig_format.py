import os
from enum import Enum

__all__ = ['ImageVisFigureFormat', 'SUPPORTED_VISUALIZATION_FORMATS']


class ImageVisFigureFormat(Enum):
    """
    Enum describing format to save visualization images (i.e. figures)
    """
    png = 1
    eps = 2
    pdf = 3
    jpeg = 4
    pgf = 5
    ps = 6
    raw = 7
    rgba = 8
    svg = 9
    svgz = 10
    tiff = 11

    def get_filepath(self, filepath):
        filename, ext = os.path.splitext(filepath)
        return '%s.%s' % (filename, self.name)


SUPPORTED_VISUALIZATION_FORMATS = [x.name for x in ImageVisFigureFormat]
