"""
i16_msmapper
Simple GUI to run the msmapper code on Beamline I16

msmapper is developed by Peter Chang & SciSoft Group, Diamond Light Source
https://github.com/DawnScience/scisoft-core/blob/master/uk.ac.diamond.scisoft.analysis/src/uk/ac/diamond/scisoft/analysis/diffraction/MillerSpaceMapper.java
https://alfred.diamond.ac.uk/documentation/javadocs/GDA/master/uk/ac/diamond/scisoft/analysis/diffraction/MillerSpaceMapper.html

Usage:
    $ module load msmapper
    $ python -m i16_msmapper.py


By Dan Porter, PhD
Diamond Light Source Ltd.
2022
"""
import sys, os
sys.path.insert(0, '/dls_sw/i16/software/python/babelscan')
sys.path.insert(0, os.path.expanduser('~/OneDrive - Diamond Light Source Ltd/PythonProjects/babelscan'))

from i16_msmapper.mapper_runner import run_msmapper, get_nexus_data, get_pixel_steps
from i16_msmapper.tkmsmapper import MsMapperGui

__version__ = '0.1.0'
__date__ = '13/12/22'


def version_info():
    return 'i16_msmapper version %s (%s)' % (__version__, __date__)


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
