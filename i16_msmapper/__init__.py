"""
i16_msmapper
Simple GUI to run the msmapper code on Beamline I16

The Miller Space Mapper (msmapper) program converts x-ray diffraction scans with area detectors into reciprocal space units.
msmapper is developed by Peter Chang & SciSoft Group, Diamond Light Source
https://github.com/DawnScience/scisoft-core/blob/master/uk.ac.diamond.scisoft.analysis/src/uk/ac/diamond/scisoft/analysis/diffraction/MillerSpaceMapper.java
https://alfred.diamond.ac.uk/documentation/javadocs/GDA/master/uk/ac/diamond/scisoft/analysis/diffraction/MillerSpaceMapper.html
https://confluence.diamond.ac.uk/display/I16/HKL+Mapping

Usage:
    $ module load msmapper
    $ python -m i16_msmapper

How it works:
 - uses the msmapper command, using a bean.json file with parameters
 - generates a bean.json file in a writable temporary directory
 - runs the command:
    $ msmapper -bean /location/of/bean.json

Example:
import json
bean = {
    "inputs": ['file1.nxs'],  # Filename of scan file
    "output": 'output_file.nxs',
    # Output filename - must be in processing directory, or somewhere you can write to
    "splitterName": "gaussian",  # one of the following strings "nearest", "gaussian", "negexp", "inverse"
    "splitterParameter": 2.0,
    # splitter's parameter is distance to half-height of the weight function.
    # If you use None or "" then it is treated as "nearest"
    "scaleFactor": 2.0,
    # the oversampling factor for each image; to ensure that are no gaps in between pixels in mapping
    "step": [0.002, 0.002, 0.001],
    # a single value or list if 3 values and determines the lengths of each side of the voxels in the volume
    "start": [0-100*0.002, 0-100*0.002, 1-200*0.001],  # location in HKL space of the bottom corner of the array.
    "shape": [200, 200, 400],  # size of the array to create for reciprocal space volume
    "reduceToNonZero": False  # True/False, if True, attempts to reduce the volume output
}
json.dump(bean, open("/location/of/bean.json", 'w'), indent=4)

# run msmapper
$ msmapper -bean /location/of/bean.json


By Dan Porter, PhD
Diamond Light Source Ltd.
2023
"""
import sys, os
# sys.path.insert(0, '/dls_sw/i16/software/python/babelscan')
sys.path.insert(0, os.path.expanduser('~/OneDrive - Diamond Light Source Ltd/PythonProjects/babelscan'))

from i16_msmapper.mapper_runner import run_msmapper, get_nexus_data, get_pixel_steps
from i16_msmapper.tkmsmapper import MsMapperGui

__version__ = '1.1.0'
__date__ = '21/06/23'


def version_info():
    return 'i16_msmapper version %s (%s)' % (__version__, __date__)


def title():
    return 'I16 MSMapper  version %s' % __version__


def module_info():
    out = 'Python version %s' % sys.version
    out += '\n%s' % version_info()
    # Modules
    import numpy
    out += '\n     numpy version: %s' % numpy.__version__
    import tkinter
    out += '\n   tkinter version: %s' % tkinter.TkVersion
    out += '\n'
    return out


def doc_str():
    return __doc__
