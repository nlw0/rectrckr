import json
import os
import Image
import numpy

import rectrckr.lowlevel as lowlevel

class DataSource:
    def __init__(self, meta_fp):
        self.path = os.path.dirname(meta_fp.name)

        self.meta_fp = meta_fp
        dic = json.load(self.meta_fp)
        self.meta_fp.close()

        for attribute, val in dic.items():
            setattr(self, attribute, val)

        self.Nimages = len(os.listdir(os.path.join(self.path,
                                                   'frames')))

    def get_image(self, n):
        img_path = os.path.join(self.path,
                                'frames',
                                self.image_name_format.format(n))
        img = numpy.array(
            Image.open(img_path).convert('L'), dtype=numpy.float64)
        return ImageAnalyzer(img)

    def get_output_path(self, n):
        return os.path.join(self.path,
                            'output',
                            self.image_name_format.format(n))


class ImageAnalyzer:
    def __init__(self, img):
        self.img = img

    def extract_edgels(self, fx, fy):
        px, py = int(round(fx)), int(round(fy))
        return numpy.array(
            (lowlevel.find_edges(self.img, px, py, 0),
             lowlevel.find_edges(self.img, px, py, 1),
             lowlevel.find_edges(self.img, px, py, 2),
             lowlevel.find_edges(self.img, px, py, 3)
             )
            )
