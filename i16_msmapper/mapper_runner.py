"""
Runs msmapper on Diamond linux workstations

run with
$ module load msmapper

"""

import sys, os
import tempfile
import subprocess
import json
import h5py
import numpy as np
import babelscan

SHELL_CMD = "msmapper -bean %s"
TEMPDIR = tempfile.gettempdir()
TEMP_BEAN = 'tmp_remap.json'
TEMP_NEXUS = 'tmp_remap.nxs'


def get_nexus_data(nexus_file):
    """
    Get data
    """
    with babelscan.hdf_loader(nexus_file) as hdf:
        if '/entry1/before_scan/diffractometer_sample/h' in hdf:
            h = hdf['/entry1/before_scan/diffractometer_sample/h'][()]
            k = hdf['/entry1/before_scan/diffractometer_sample/k'][()]
            l = hdf['/entry1/before_scan/diffractometer_sample/l'][()]
            return h, k, l
    # if address is wrong, fall back on the dynamic scan class
    scan = babelscan.file_loader(nexus_file)
    h, k, l = scan('h, k, l')
    # cmd = scan.string_format('{cmd}')
    # hkl_str = f"({h:.4g},{k:.4g},{l:.5g})"
    return h, k, l


def get_pixel_steps(nexus_file):
    """
    Get minimum pixel steps for a scan file
     - Creates & stores json bean file with outputMode: 'Coords_HKL' + fixed pixel indexes
     - Runs msmapper (requires msmapper environment)
     - Opens resulting Nexus file and loads coordinates
     - determines variation in hkl between adjacent pixels

    :param nexus_file: str filename of .nxs scan file
    :return: h_diff, k_diff, l_diff
    """

    nxs_file = os.path.join(TEMPDIR, TEMP_NEXUS)
    bean_file = os.path.join(TEMPDIR, TEMP_BEAN)
    bean = {
        "inputs": [nexus_file],
        "output": nxs_file,
        "outputMode": "Coords_HKL",
        "pixelIndexes": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
    }
    json.dump(bean, open(bean_file, 'w'), indent=4)
    print('bean file written to: %s' % bean_file)

    print('\n\n\n################# Starting msmapper ###################\n\n\n')
    output = subprocess.run(SHELL_CMD % bean_file, shell=True, capture_output=True)
    print(output.stdout)
    print('\n\n\n################# msmapper finished ###################\n\n\n')

    # 3. Read nxs file
    print('\nReading %s' % nxs_file)
    with babelscan.hdf_loader(nxs_file) as hdf:
        coords = hdf['processed/reciprocal_space/coordinates'][()]

    hkl_diff = np.abs(np.mean(np.diff(coords, axis=0), axis=0))
    print('\n\n***Results***')
    print('File: %s' % nexus_file)
    print('pixel step (h,k,l) = (%.2g, %.2g, %.2g)\n\n' % (hkl_diff[0], hkl_diff[1], hkl_diff[2]))
    return hkl_diff


def rsmap_command(input_files, output_file, start=None, shape=None, step=0.002):
    """
    Create rsmap command from values
    $ rs_map -s 0.002 -o /dls/i16/data/2022/mm12345-1/processing/12345_remap.nxs /dls/i16/data/2022/mm12345-1/12345.nxs
    :param input_files: list of scan file locations
    :param output_file: str localtion of output file
    :param start: [h, k, l] start of box
    :param shape: [n, m, o] size of box in voxels
    :param step: [dh, dk, dl] step size in each direction - size of voxel in reciprocal lattice units
    :return: str file location of bean file
    """
    input_files = np.asarray(input_files, dtype=str).reshape(-1).tolist()
    return f"rs_map -s {step} -o {output_file} {input_files[0]}"


def create_bean_file(input_files, output_file, start=None, shape=None, step=None):
    """
    Create a bean file for msmapper in a temporary directory
     currently only allows a few standard inputs: hkl_start, shape and step values.
    :param input_files: list of scan file locations
    :param output_file: str localtion of output file
    :param start: [h, k, l] start of box
    :param shape: [n, m, o] size of box in voxels
    :param step: [dh, dk, dl] step size in each direction - size of voxel in reciprocal lattice units
    :return: str file location of bean file
    """
    input_files = np.asarray(input_files, dtype=str).reshape(-1).tolist()
    # Remove empty entries
    while '' in input_files:
        input_files.remove('')
    if step is None:
        step = [0.001, 0.001, 0.001]
    else:
        step = np.asarray(step, dtype=float).reshape(-1).tolist()

    if shape is not None:
        shape = np.asarray(shape, dtype=int).reshape(-1).tolist()

    if start is not None:
        start = np.asarray(start, dtype=float).reshape(-1).tolist()

    bean = {
        "inputs": input_files,  # Filename of scan file
        "output": output_file,
        # Output filename - must be in processing directory, or somewhere you can write to
        "splitterName": "gaussian",  # one of the following strings "nearest", "gaussian", "negexp", "inverse"
        "splitterParameter": 2.0,
        # splitter's parameter is distance to half-height of the weight function.
        # If you use None or "" then it is treated as "nearest"
        "scaleFactor": 2.0,
        # the oversampling factor for each image; to ensure that are no gaps in between pixels in mapping
        "step": step,
        # a single value or list if 3 values and determines the lengths of each side of the voxels in the volume
        "start": start,  # location in HKL space of the bottom corner of the array.
        "shape": shape,  # size of the array to create for reciprocal space volume
        "reduceToNonZero": False  # True/False, if True, attempts to reduce the volume output
    }

    bean_file = os.path.join(TEMPDIR, TEMP_BEAN)
    json.dump(bean, open(bean_file, 'w'), indent=4)
    print('bean file written to: %s' % bean_file)
    return bean_file


def run_msmapper(input_files, output_file, start=None, shape=None, step=None):
    """
    Create the input file and run msmapper
     currently only allows a few standard inputs: hkl_start, shape and step values.
    :param input_files: list of scan file locations
    :param output_file: str localtion of output file
    :param start: [h, k, l] start of box
    :param shape: [n, m, o] size of box in voxels
    :param step: [dh, dk, dl] step size in each direction - size of voxel in reciprocal lattice units
    :return: str file location of bean file
    """
    bean_file = create_bean_file(input_files, output_file, start, shape, step)
    print('bean file: %s' % bean_file)
    # shell_cmd = f"gnome-terminal -- bash -c \"module load msmapper;msmapper -bean {bean_file}; exec bash\""
    print('\n\n\n################# Starting msmapper ###################\n\n\n')
    output = subprocess.run(SHELL_CMD % bean_file, shell=True, capture_output=True)
    print(output.stdout)
    print('\n\n\n################# msmapper finished ###################\n\n\n')


if __name__ == '__main__':
    print(create_bean_file('/dls/i16/data/2022/cm31138-15/954096.nxs', 'test.nxs'))
    # run_msmapper('/dls/i16/data/2022/cm31138-15/954096.nxs', 'test.nxs')
