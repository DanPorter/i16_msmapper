# i16_msmapper
Simple GUI to run the msmapper code on Beamline I16

The Miller Space Mapper (msmapper) program converts x-ray diffraction scans with area detectors into reciprocal space units.
msmapper is developed by Peter Chang & SciSoft Group, Diamond Light Source Ltd.
Links:
 - [GitHub](https://github.com/DawnScience/scisoft-core/blob/master/uk.ac.diamond.scisoft.analysis/src/uk/ac/diamond/scisoft/analysis/diffraction/MillerSpaceMapper.java)
 - [Javadocs](https://alfred.diamond.ac.uk/documentation/javadocs/GDA/master/uk/ac/diamond/scisoft/analysis/diffraction/MillerSpaceMapper.html)
 - [I16 Confluence Page](https://confluence.diamond.ac.uk/display/I16/HKL+Mapping)

By Dan Porter, Diamond Light Source Ltd. 2022

### Usage
```commandline
$ module load msmapper
$ cd i16_msmapper
$ python -m i16_msmapper
```

### Installation
**requirements:** *tkinter, numpy, matplotlib, h5py, babelscan, scisoftpy, msmapper*

**available from: https://github.com/DanPorter/i16_msmapper**

Latest version from github:
```commandline
git clone https://github.com/DanPorter/i16_msmapper.git
```

### Screenshot
![msmapper_gui](https://github.com/DanPorter/i16_msmapper/blob/master/msmapper_gui.png?raw=true)

