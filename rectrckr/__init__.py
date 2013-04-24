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
            (lowlevel.find_edge(self.img, px, py, 0),
             lowlevel.find_edge(self.img, px, py, 1),
             lowlevel.find_edge(self.img, px, py, 2),
             lowlevel.find_edge(self.img, px, py, 3)
             )
            )

    def extract_moar_edgels(self, fx, fy, gs=16):
        px, py = int(round(fx)), int(round(fy))
        return numpy.array(
            (
                (lowlevel.find_edge(self.img, px, py, 0),py),
                (lowlevel.find_edge(self.img, px, py, 1),py),
                (px, lowlevel.find_edge(self.img, px, py, 2)),
                (px, lowlevel.find_edge(self.img, px, py, 3)),
                (lowlevel.find_edge(self.img, px, py-gs, 0),py-gs),
                (lowlevel.find_edge(self.img, px, py-gs, 1),py-gs),
                (px-gs, lowlevel.find_edge(self.img, px-gs, py, 2)),
                (px-gs, lowlevel.find_edge(self.img, px-gs, py, 3)),
                (lowlevel.find_edge(self.img, px, py+gs, 0),py+gs),
                (lowlevel.find_edge(self.img, px, py+gs, 1),py+gs),
                (px+gs, lowlevel.find_edge(self.img, px+gs, py, 2)),
                (px+gs, lowlevel.find_edge(self.img, px+gs, py, 3)),
                )
            )

    def estimate_direction_at_point(self, px, py):
        return numpy.array(lowlevel.gradient(self.img, px, py))
        
        
