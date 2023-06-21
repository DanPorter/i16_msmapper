"""
Plot completed msmapper files
"""

import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import babelscan
from babelscan.plotting_matplotlib import set_plot_defaults
set_plot_defaults()  # set custom matplotlib rcPar


def get_remap(nexus_file):
    """
    Returns babelscan Volume object
    :param nexus_file:
    :return: scan
    """
    scan = babelscan.file_loader(nexus_file)
    try:
        vol = scan['volume']
        print('File contains remapped data')
        return scan
    except Exception:
        # Find remapped files
        expdir = os.path.dirname(scan.filename)
        print('Finding remapped files in %s' % expdir)
        for ntries in range(1):
            # remapping may take some time, so keep checking until finished
            remap_files = glob.glob(expdir + '/processed/*.nxs')
            rmf = [file for file in remap_files if str(scan.scan_number) in file]
            if len(rmf) > 0:
                scan = babelscan.file_loader(rmf[0])
                try:
                    vol = scan['volume']
                    print('Remapped file loaded: %s' % scan.filename)
                    return scan
                except Exception:
                    time.sleep(10)  # wait for file to finish writing
            else:
                print('Remapped file does not exist yet, try again in 10 s')
                time.sleep(10)
        raise Exception("Scan does not exist or doesn't contain data")


def plot_scan(nexus_file):
    """Default plot scan"""
    scan = babelscan.file_loader(nexus_file)
    scan.plot()
    plt.show()


def slider_scan(nexus_file):
    """Plot scan images with slider"""
    scan = babelscan.file_loader(nexus_file)
    scan.plot.image_slider()
    plt.show()


def slider_remap(nexus_file):
    """Plot remapped data as slider"""
    scan = get_remap(nexus_file)
    vol = scan.volume('volume')
    vol.plot.image_slider()
    plt.show()


def plot_remap_hkl(nexus_file):
    """
    Plot Remapped file
    :param nexus_file: str
    :return:
    """

    scan = get_remap(nexus_file)

    # Get reciprocal space data from file
    h = scan['h-axis']
    k = scan['k-axis']
    l = scan['l-axis']
    vol = scan['volume']

    print('\n'.join(scan.string(['h-axis', 'k-axis', 'l-axis', 'volume'])))

    # Plot summed cuts
    plt.figure(figsize=[18, 8], dpi=60)
    title = scan.string_format('#{scan_number:.0f}: {scan_command}')
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
    title = scan.string_format('#{scan_number:.0f}: {scan_command}')
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
    h = scan['h-axis']
    k = scan['k-axis']
    l = scan['l-axis']
    vol = scan['volume']

    print('\n'.join(scan.string(['h-axis', 'k-axis', 'l-axis', 'volume'])))

    # Convert to Q
    a, b, c, alpha, beta, gamma = scan('unit_cell')[0]

    astar, bstar, cstar = RcSp(latpar2uv(a, b, c, alpha, beta, gamma))

    qx, qy, qz = genq(h, k, l, [astar, bstar, cstar])
    qmag = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)

    # Plot summed images
    plt.figure(figsize=[18, 8], dpi=60)
    title = scan.string_format('#{scan_number:.0f}: {scan_command}')
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
    h = scan['h-axis']
    k = scan['k-axis']
    l = scan['l-axis']
    vol = scan['volume']

    print('\n'.join(scan.string(['h-axis', 'k-axis', 'l-axis', 'volume'])))

    # Convert to Q
    a, b, c, alpha, beta, gamma = scan('unit_cell')[0]
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

    title2 = scan.string_format('#{scan_number:.0f}\n{scan_command}')

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

