import json
import os
import Image
import numpy

import rectrckr.lowlevel as lowlevel
from corisco.quaternion import Quat
from corisco import dir_colors

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


class Camera:

    def __init__(self):
        self.fd = 500
        self.shape = numpy.array([640.0, 480.0])
        self.center = numpy.array([319.5, 239.5])

        self.position = numpy.array([0,0,-20.0])
        self.orientation = Quat(1.0,0,0,0)

    def project_points(self, points):
        M = self.orientation.rot()
        tp = numpy.dot((points - self.position), M)
        return self.intrinsic_transform(tp)

    def intrinsic_transform(self, points):
        nf = self.fd / points[:,2]
        return self.center + numpy.c_[points[:,0] * nf, points[:,1] * nf]

    def set_position(self, pos):
        self.position = pos
        
    def set_orientation(self, ori):
        self.orientation = ori
        
    def plot_points(self, ax, scn):
        pp = self.project_points(scn.points)

        ax.plot(pp[:,0], pp[:,1], 'bo')
        ax.axis('equal')
        ax.axis([0,self.shape[0], self.shape[1], 0])

    def plot_edges(self, ax, scn):
        M = self.orientation.rot()
        pp = []
        pp1 = numpy.zeros(3)
        pp2 = numpy.zeros(3)
        for d,x1,x2,y,z in scn.edges:
            if d == 0:
                pp.append([x1,y,z])
                pp.append([x2,y,z])
            elif d == 1:
                pp.append([z,x1,y])
                pp.append([z,x2,y])
            elif d == 2:
                pp.append([y,z,x1])
                pp.append([y,z,x2])

        ip = self.project_points(numpy.array(pp))

        for k in xrange(ip.shape[0]/2):
            ax.plot([ip[2*k,0], ip[2*k+1,0]],
                    [ip[2*k,1], ip[2*k+1,1]],                  
                    'r-', lw=2, color=dir_colors[int(scn.edges[k,0])])

        ax.axis('equal')
        ax.axis([0,self.shape[0], self.shape[1], 0])


class SceneModel:
    def __init__(self, edges, points):
        self.edges = edges
        self.points = points

