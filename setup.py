import distutils
import setuptools

import numpy

setuptools.setup(
    name="rectrckr",
    version="0.1dev",
    packages=["rectrckr"],
    entry_points={
        "console_scripts": [
            "rtr_init = rectrckr.tools.rtr_init:main",
        ]
    },
    install_requires=[
    ],

    ext_modules=[
        distutils.extension.Extension(
            "rectrckr.lowlevel",
            sources=["rectrckr/lowlevel.c"],
        ),
    ],
    include_dirs=[numpy.get_include()],
)
