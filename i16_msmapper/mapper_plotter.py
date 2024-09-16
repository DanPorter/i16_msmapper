"""
Plot completed msmapper files
"""

import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import hdfmap


DEFAULT_FONT = 'Times New Roman'
DEFAULT_FONTSIZE = 14
FIG_SIZE = [12, 8]
FIG_DPI = 80


def set_plot_defaults(rcdefaults=False):
    """
    Set custom matplotlib rcparams, or revert to matplotlib defaults
    These handle the default look of matplotlib plots
    See: https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
    :param rcdefaults: False*/ True, if True, revert to matplotlib defaults
    :return: None
    """
    if rcdefaults:
        print('Return matplotlib rcparams to default settings.')
        plt.rcdefaults()
        return

    plt.rc('figure', figsize=FIG_SIZE, dpi=FIG_DPI, autolayout=False)
    plt.rc('lines', marker='o', color='r', linewidth=2, markersize=6)
    plt.rc('errorbar', capsize=2)
    plt.rc('legend', loc='best', frameon=False, fontsize=DEFAULT_FONTSIZE)
    plt.rc('axes', linewidth=2, titleweight='bold', labelsize='large')
    plt.rc('xtick', labelsize='large')
    plt.rc('ytick', labelsize='large')
    plt.rc('axes.formatter', limits=(-3, 3), offset_threshold=6)
    # default colourmap, see https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.rc('image', cmap='viridis')
    # Note font values appear to only be set when plt.show is called
    plt.rc(
        'font',
        family='serif',
        style='normal',
        weight='bold',
        size=DEFAULT_FONTSIZE,
        serif=[DEFAULT_FONT, 'Times', 'DejaVu Serif']
    )
    # plt.rcParams["savefig.directory"] = os.path.dirname(__file__) # Default save directory for figures


def get_remap(nexus_file):
    """
    Returns HdfMap.NexusLoader object
    :param nexus_file:
    :return: scan
    """
    scan = hdfmap.NexusLoader(nexus_file)
    if 'volume' in scan.map:
        print('File contains remapped data')
        return scan
    # Find remapped files
    path, filename = os.path.split(nexus_file)
    name, ext = os.path.splitext(filename)
    print('Finding remapped files in %s' % path)
    for ntries in range(1):
        # remapping may take some time, so keep checking until finished
        remap_files = glob.glob(path + '/processed/*.nxs')
        rmf = [file for file in remap_files if name in file]
        if len(rmf) > 0:
            scan = hdfmap.NexusLoader(rmf[0])
            if 'volume' in scan.map:
                print(f"Remapped file loaded: {scan.filename}")
                return scan
            print(f"Wait for file to finish writing: {scan.filename}")
            time.sleep(10)  # wait for file to finish writing
        else:
            print('Remapped file does not exist yet, try again in 10 s')
            time.sleep(10)
    raise Exception("Scan does not exist or doesn't contain data")


def plot_scan(nexus_file):
    """Default plot scan"""
    scan = hdfmap.NexusLoader(nexus_file)
    xdata, ydata = scan('axes, signal')
    xlabel, ylabel = scan('_axes, _signal')
    plt.figure()
    plt.plot(xdata, ydata, label=nexus_file)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(scan.format('{filename}: {scan_command}'))
    plt.show()


def slider_scan(nexus_file):
    """Plot scan images with slider"""
    nxs_map = hdfmap.create_nexus_map(nexus_file)
    image_path = nxs_map.get_image_path()
    image_len = nxs_map.datasets[image_path].shape[0]
    if not image_path:
        raise Exception('File contains no image data')

    with hdfmap.load_hdf(nexus_file) as hdf:
        image = nxs_map.get_image(hdf, 0)

        fig = plt.figure()
        axes = fig.add_subplot(111)
        pcm = axes.pcolormesh(image)
        axes.invert_yaxis()
        axes.axis('image')
        axes.set_title(nexus_file)

        # Move axes for slider
        bbox = axes.get_position()
        left, bottom, width, height = bbox.x0, bbox.y0, bbox.width, bbox.height
        change_in_height = height * 0.1
        new_position = [left, bottom + 2 * change_in_height, width, height - 2 * change_in_height]
        new_axes_position = [left, bottom, width, change_in_height]

        axes.set_position(new_position, 'original')
        new_axes = axes.figure.add_axes(new_axes_position)

        sldr = plt.Slider(new_axes, label=image_path, valmin=0, valmax=image_len-1,
                          valinit=0, valfmt='%0.0f')

        def update(val):
            """Update function for pilatus image"""
            imgno = int(round(sldr.val))
            im = nxs_map.get_image(hdf, imgno)
            pcm.set_array(im.flatten())
            plt.draw()
            # fig.canvas.draw()

        sldr.on_changed(update)
        plt.show()


def slider_remap(nexus_file):
    """Plot remapped data as slider"""
    nxs_map = hdfmap.create_nexus_map(nexus_file)
    image_path = nxs_map['volume']
    if not image_path:
        raise Exception('File contains no image data')

    with hdfmap.load_hdf(nexus_file) as hdf:
        image = nxs_map.get_data(hdf, image_path, index=(slice(None), slice(None), 0))
        x_axis, y_axis, z_axis = nxs_map.eval(hdf, 'h_axis, k_axis, l_axis')
        yy, xx = np.meshgrid(y_axis, x_axis)

        fig = plt.figure()
        axes = fig.add_subplot(111)
        pcm = axes.pcolormesh(xx, yy, image)
        axes.set_title(nexus_file)
        axes.axis('image')
        axes.set_xlabel('H (r.l.u.)')
        axes.set_ylabel('K (r.l.u.)')

        # Move axes for slider
        bbox = axes.get_position()
        left, bottom, width, height = bbox.x0, bbox.y0, bbox.width, bbox.height
        change_in_height = height * 0.1
        new_position = [left, bottom + 2 * change_in_height, width, height - 2 * change_in_height]
        new_axes_position = [left, bottom, width, change_in_height]

        axes.set_position(new_position, 'original')
        new_axes = axes.figure.add_axes(new_axes_position)

        sldr = plt.Slider(new_axes, label='L', valmin=z_axis[0], valmax=z_axis[-1], valinit=z_axis[0])

        def update(val):
            """Update function for pilatus image"""
            idx = np.argmin(np.abs(sldr.val - z_axis))
            im = nxs_map.get_data(hdf, image_path, index=(slice(None), slice(None), idx))
            pcm.set_array(im.flatten())
            plt.draw()
            # fig.canvas.draw()

        sldr.on_changed(update)
        plt.show()


def plot_remap_hkl(nexus_file):
    """
    Plot Remapped file
    :param nexus_file: str
    :return:
    """

    scan = get_remap(nexus_file)

    # Get reciprocal space data from file
    h = scan('h_axis')
    k = scan('k_axis')
    l = scan('l_axis')
    vol = scan('volume')

    # print('\n'.join(scan.string(['h-axis', 'k-axis', 'l-axis', 'volume'])))

    # Plot summed cuts
    plt.figure(figsize=[18, 8], dpi=60)
    title = scan.format('{filename}\n{scan_command}')
    plt.suptitle(title, fontsize=18)

    plt.subplot(131)
    plt.plot(h, vol.sum(axis=1).sum(axis=1))
    plt.xlabel('h-axis (r.l.u.)', fontsize=16)
    plt.ylabel('sum axes [1,2]', fontsize=16)

    plt.subplot(132)
    plt.plot(k, vol.sum(axis=0).sum(axis=1))
    plt.xlabel('k-axis (r.l.u.)', fontsize=16)
    plt.ylabel('sum axes [0,2]', fontsize=16)

    plt.subplot(133)
    plt.plot(l, vol.sum(axis=0).sum(axis=0))
    plt.xlabel('l-axis (r.l.u.)', fontsize=16)
    plt.ylabel('sum axes [0,1]', fontsize=16)

    # Plot summed images
    plt.figure(figsize=[18, 8], dpi=60)
    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(wspace=0.3)

    plt.subplot(131)
    K, H = np.meshgrid(k, h)
    plt.pcolormesh(H, K, vol.sum(axis=2))
    plt.xlabel('h-axis (r.l.u.)', fontsize=16)
    plt.ylabel('k-axis (r.l.u.)', fontsize=16)
    # plt.axis('image')
    # plt.colorbar()

    plt.subplot(132)
    L, H = np.meshgrid(l, h)
    plt.pcolormesh(H, L, vol.sum(axis=1))
    plt.xlabel('h-axis (r.l.u.)', fontsize=16)
    plt.ylabel('l-axis (r.l.u.)', fontsize=16)
    # plt.axis('image')
    # plt.colorbar()

    plt.subplot(133)
    L, K = np.meshgrid(l, k)
    plt.pcolormesh(K, L, vol.sum(axis=0))
    plt.xlabel('k-axis (r.l.u.)', fontsize=16)
    plt.ylabel('l-axis (r.l.u.)', fontsize=16)
    # plt.axis('image')
    plt.colorbar()

    plt.show()


# Functions from Dans_Diffraction
def latpar2uv(a, b, c, alpha, beta, gamma):
    # From http://pymatgen.org/_modules/pymatgen/core/lattice.html
    alpha_r = np.radians(alpha)
    beta_r = np.radians(gamma)
    gamma_r = np.radians(beta)
    val = (np.cos(alpha_r) * np.cos(beta_r) - np.cos(gamma_r)) \
          / (np.sin(alpha_r) * np.sin(beta_r))
    # Sometimes rounding errors result in values slightly > 1.
    val = abs(val)
    gamma_star = np.arccos(val)
    aa = [a * np.sin(beta_r), a * np.cos(beta_r), 0.0]
    bb = [0.0, b, 0.0]
    cc = [-c * np.sin(alpha_r) * np.cos(gamma_star),
          c * np.cos(alpha_r),
          c * np.sin(alpha_r) * np.sin(gamma_star)]

    return np.round(np.array([aa, bb, cc]), 8)


def RcSp(UV):
    """
    Generate reciprocal cell from real space unit vecors
    Usage:
    UVs = RcSp(UV)
      UV = [[3x3]] matrix of vectors [a,b,c]
    """
    UVs = 2 * np.pi * np.linalg.inv(UV).T
    return UVs


def genq(h, k, l, UVstar):
    avec, bvec, cvec = UVstar
    kk, hh, ll = np.meshgrid(k, h, l)
    shape = np.shape(hh)
    q = avec * hh.reshape(-1, 1) + bvec * kk.reshape(-1, 1) + cvec * ll.reshape(-1, 1)
    qx = q[:, 0].reshape(shape)
    qy = q[:, 1].reshape(shape)
    qz = q[:, 2].reshape(shape)
    return qx, qy, qz


def cal2theta(qmag, energy_kev=17.794):
    """
    Calculate theta at particular energy in keV from |Q|
     twotheta = cal2theta(Qmag,energy_kev=17.794)
    """
    h = 6.62606868E-34  # Js  Plank consant
    c = 299792458  # m/s   Speed of light
    e = 1.6021733E-19  # C  electron charge
    energy = energy_kev * 1000.0  # energy in eV
    # Calculate 2theta angles for x-rays
    twotheta = 2 * np.arcsin(qmag * 1e10 * h * c / (energy * e * 4 * np.pi))
    # return x2T in degrees
    twotheta = twotheta * 180 / np.pi
    return twotheta


def plot_remap_q(nexus_file):
    """
    Plot cuts in Q
    :param nexus_file:
    :return:
    """

    scan = get_remap(nexus_file)

    # Get reciprocal space data from file
    h = scan('h_axis')
    k = scan('k_axis')
    l = scan('l_axis')
    vol = scan('volume')

    # Convert to Q
    a, b, c, alpha, beta, gamma = scan('unit_cell')

    astar, bstar, cstar = RcSp(latpar2uv(a, b, c, alpha, beta, gamma))

    qx, qy, qz = genq(h, k, l, [astar, bstar, cstar])
    qmag = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)

    # Plot summed images
    plt.figure(figsize=[18, 8], dpi=60)
    title = scan.format('{filename}\n{scan_command}')
    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(wspace=0.3)

    plt.subplot(131)
    plt.pcolormesh(qx.mean(axis=2), qy.mean(axis=2), vol.sum(axis=2))
    plt.xlabel('Qx [A$^{-1}$]', fontsize=16)
    plt.ylabel('Qy [A$^{-1}$]', fontsize=16)
    # plt.axis('image')
    # plt.colorbar()

    plt.subplot(132)
    plt.pcolormesh(qx.mean(axis=1), qz.mean(axis=1), vol.sum(axis=1))
    plt.xlabel('Qx [A$^{-1}$]', fontsize=16)
    plt.ylabel('Qz [A$^{-1}$]', fontsize=16)
    # plt.axis('image')
    # plt.colorbar()

    plt.subplot(133)
    plt.pcolormesh(qy.mean(axis=0), qz.mean(axis=0), vol.sum(axis=0))
    plt.xlabel('Qy [A$^{-1}$]', fontsize=16)
    plt.ylabel('Qz [A$^{-1}$]', fontsize=16)
    # plt.axis('image')
    plt.colorbar()
    plt.show()


def plot_qmag(nexus_file):
    """
    Plot two0-theta intensity and intensity vs |Q|
    :param nexus_file:
    :return:
    """

    scan = get_remap(nexus_file)

    # Get reciprocal space data from file
    h = scan('h_axis')
    k = scan('k_axis')
    l = scan('l_axis')
    vol = scan('volume')

    # Convert to Q
    a, b, c, alpha, beta, gamma = scan('unit_cell')
    energy = scan('incident_energy')

    astar, bstar, cstar = RcSp(latpar2uv(a, b, c, alpha, beta, gamma))

    qx, qy, qz = genq(h, k, l, [astar, bstar, cstar])
    qmag = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)

    qmag = qmag.reshape(-1)
    qvol = vol.reshape(-1)
    bin_cen = np.arange(qmag.min(), qmag.max(), 0.001)
    bin_edge = bin_cen + 0.005
    bin_pos = np.digitize(qmag, bin_edge) - 1
    bin_sum = [np.mean(qvol[bin_pos == n]) for n in range(len(bin_cen))]
    tth = cal2theta(bin_cen, energy)

    title2 = scan.format('{filename}\n{scan_command}')

    plt.figure(dpi=100)
    plt.plot(bin_cen, bin_sum)
    plt.title(title2, fontsize=20)
    plt.xlabel('|Q| A$^{-1}$', fontsize=20)
    plt.ylabel('Intensity', fontsize=20)

    plt.figure(dpi=100)
    plt.plot(tth, bin_sum)
    plt.title(title2, fontsize=20)
    plt.xlabel('Two-Theta [Deg]', fontsize=20)
    plt.ylabel('Intensity', fontsize=20)
    plt.show()

