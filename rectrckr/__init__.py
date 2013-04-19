import json
import os
import Image
import numpy

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
        for self.current_image in xrange(self.Nimages):
            yield self.get_image(self.current_image)
    
    def get_image(self, n):
        img_path = os.path.join(self.path,
                                'frames',
                                self.image_name_format.format(n))
        return numpy.array(Image.open(img_path).convert('L'), dtype=numpy.float64)
            
class Tracker:

    def __init__(self, initial_state):
        pass
        
    
    


