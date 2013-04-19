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

    def images(self):
        for self.current_image_n in xrange(self.Nimages):
            yield self.get_image(self.current_image_n)
    
    def get_image(self, n):
        img_path = os.path.join(self.path,
                                'frames',
                                self.image_name_format.format(n))
        self.current_image = numpy.array(
            Image.open(img_path).convert('L'), dtype=numpy.float64)
        return self.current_image

    def extract_edgels(self, px, py):
        img = self.current_image
        return numpy.array(
            ((lowlevel.find_edges(img, px, py, 0), py),
             (lowlevel.find_edges(img, px, py, 1), py),
             (px, lowlevel.find_edges(img, px, py, 2)),
             (px, lowlevel.find_edges(img, px, py, 3))))

            
class SimpleTracker:

    def __init__(self, initial_state):        
        self.state = initial_state      

    def update(self, edges):
        self.state[0] = (edges[0,0] + edges[1,0])/2
        self.state[1] = (edges[2,1] + edges[3,1])/2
        self.state[2] = (edges[0,0] - edges[1,0])/2
        self.state[3] = (edges[2,1] - edges[3,1])/2
        
        pass
