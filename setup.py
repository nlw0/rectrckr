import distutils
import setuptools
from Cython.Distutils import build_ext

import numpy

setuptools.setup(
    name="rectrckr",
    version="0.1dev",
    packages=["rectrckr"],
    entry_points={
        "console_scripts": [
            "rtr_test_extractor = rectrckr.tools.rtr_test_extractor:main",
            "rtr_test_edgel_extractor = rectrckr.tools.rtr_test_edgel_extractor:main",
            "rtr_simple_stracker = rectrckr.tools.rtr_simple_tracker:main",
            "rtr_edgel_tracker = rectrckr.tools.rtr_edgel_tracker:main",
            "rtr_analyze_edgel_data = rectrckr.tools.rtr_analyze_edgel_data:main",
            "rtr_simulate_edgels = rectrckr.tools.rtr_simulate_edgels:main",
            "rtr_estimate_pose = rectrckr.tools.rtr_estimate_pose:main",
            "rtr_test_scene_model = rectrckr.tools.rtr_test_scene_model:main",
            "rtr_test_camera_model = rectrckr.tools.rtr_test_camera_model:main",
            "rtr_test_corisco = rectrckr.tools.rtr_test_corisco:main",
            "rtr_test_filter = rectrckr.tools.rtr_test_filter:main",
        ]
    },
    install_requires=[
    ],

    cmdclass = {'build_ext': build_ext},
    ext_modules=[
        distutils.extension.Extension(
            "rectrckr.lowlevel",
            sources=["rectrckr/lowlevel.cc", "rectrckr/target.cc", "rectrckr/camera_models.cc"],
        ),

        distutils.extension.Extension(
            "rectrckr.corisco.corisco_aux",
            sources=["rectrckr/corisco/corisco_aux.pyx",
                     "rectrckr/corisco/corisco_aux_external.c"],

            extra_compile_args=['-msse2', '-mfpmath=sse'],
        ),
    ],
    include_dirs=[numpy.get_include()],
)
