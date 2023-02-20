#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 6 09:32:28 2023
@author: jlee
"""


# importing necessary modules
import numpy as np
import glob, os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from astropy.io import fits


# ----- Reading the Data ----- #

### Released (un-binned / vornoi-binned) data
wht_img = "NGC0628_PHANGS_IMAGE_white_copt_0.92asec.fits"
wht_data = fits.getdata(wht_img, ext=1, header=False)
wht_stat = fits.getdata(wht_img, ext=2, header=False)

wht_stat[wht_stat == 0] = np.nanmedian(wht_stat[wht_stat > 0.])
wht_SNR = wht_data / np.sqrt(wht_stat)

cube_img = "NGC0628_MAPS_copt_0.92asec.fits"
vbin_data = fits.getdata(cube_img, ext=1)
h0 = fits.getheader(cube_img, ext=0)
v0 = float(h0['REDSHIFT'])

### SPHEREx-rebinned data
img_rebin = fits.getdata("rebin.fits")
snr_rebin = fits.getdata("rebin_SNR.fits")


# ----- Function for reading the pPXF results ----- #
def read_all_parts(dir_run):
    if (dir_run[-1] != "/"):
        dir_run += "/"

    file_run = sorted(glob.glob(dir_run+"Results/part*.csv"))
    n_file = len(file_run)

    df = pd.read_csv(file_run[0])
    for i in range(1, n_file, 1):
        tmp = pd.read_csv(file_run[i])
        df = pd.concat([df, tmp], axis=0, ignore_index=True)
    return df

df_run1 = read_all_parts("Run1")
df_run2 = read_all_parts("Run2")
df_run3 = read_all_parts("Run3")
df_run4 = read_all_parts("Run4")


# ----- Plotting maps ----- #
def draw_map(data_frame, column, out, shape,
             xyid=True, vorbins=None):
    par_map = np.zeros(shape)
    for i in range(len(data_frame)):
        if xyid:
            ix, iy = data_frame['x'].values[i], data_frame['y'].values[i]
            par_map[iy, ix] = data_frame[column].values[i]
        else:
            vb = (vorbins == data_frame['BinID'].values[i])
            par_map[vb] = data_frame[column].values[i]
    fits.writeto(out, par_map, overwrite=True)


def plot_2Dmap(plt_Data, title, v_low, v_high, out, cmap='gray_r',
               add_cb=True, add_or=True, add_sc=True, cb_label=None, 
               x0=-2.75, y0=1.25, sign=-1, L=0.6, theta0=0.0*(np.pi/180.0),
               xN=-1.90, yN=1.25, xE=-2.95, yE=2.10,
               ang_scale=0.04751, pixel_scale=6.2):

    fig, ax = plt.subplots(1, 1, figsize=(6,5))
    plt.suptitle(title, x=0.5, ha='center', y=0.96, va='top',
                 fontsize=17.0)
    # ax.set_xlim([-3.4, 3.4])
    # ax.set_ylim([-2.45, 2.45])
    # ax.set_xticks([-3,-2,-1,0,1,2,3])
    # ax.set_yticks([-2,-1,0,1,2])
    # ax.set_xticklabels([r'$-3$',r'$-2$',r'$-1$',0,1,2,3], fontsize=15.0)
    # ax.set_yticklabels([r'$-2$',r'$-1$',0,1,2], fontsize=15.0)
    ax.set_xlabel(r'$\Delta X$ [arcsec]', fontsize=13.0) 
    ax.set_ylabel(r'$\Delta Y$ [arcsec]', fontsize=13.0)
    ax.tick_params(axis='both', direction='in', width=1.0, length=5.0, labelsize=14.0)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.0)

    im = ax.imshow(plt_Data, cmap=cmap, vmin=v_low, vmax=v_high, aspect='equal',
                   extent=[-plt_Data.shape[1]*pixel_scale/2, plt_Data.shape[1]*pixel_scale/2,
                           -plt_Data.shape[0]*pixel_scale/2, plt_Data.shape[0]*pixel_scale/2],
                   origin='lower')

    if add_cb:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(cb_label, size=12.0, labelpad=10.0)
        cb.ax.tick_params(direction='in', labelsize=12.0)

    if add_or:
        ax.arrow(x0+sign*0.025, y0, L*np.sin(theta0), L*np.cos(theta0), width=0.8,
                 head_width=5.0, head_length=5.0, fc='blueviolet', ec='blueviolet', alpha=0.9)
        ax.arrow(x0, y0+sign*0.025, -L*np.cos(theta0), L*np.sin(theta0), width=0.8,
                 head_width=5.0, head_length=5.0, fc='blueviolet', ec='blueviolet', alpha=0.9)
        ax.text(xN, yN, 'N', fontsize=11.0, fontweight='bold', color='blueviolet')
        ax.text(xE, yE, 'E', fontsize=11.0, fontweight='bold', color='blueviolet')
    
    if add_sc:
        kpc2 = 2.0 / ang_scale
        ax.arrow(75.0, -120.0, kpc2, 0., width=0.8, head_width=0., head_length=0.,
                  fc='blueviolet', ec='blueviolet', alpha=0.9)
        ax.text(80.0, -112.0, '2 kpc', fontsize=11.0, fontweight='bold', color='blueviolet')

    # plt.savefig(out+'.pdf', dpi=300)
    plt.savefig(out, dpi=300)
    plt.close()

dir_output = "Figure_maps"
if not os.path.exists(dir_output):
    os.system("mkdir "+dir_output)

cols = ['vel', 'sigma', 'Av_star', 'logAge_lw', 'Z_lw', 'logAge_mw', 'Z_mw', 'M/L']
out_suffix = ['vel_star', 'sigma_star', 'Av_star', 'logAge_lw', 'Z_lw', 'logAge_mw', 'Z_mw', 'ML']

'''
### Run 1
for i in range(len(cols)):
    outfile = str(Path(dir_output))+"/Map_run1_"+out_suffix[i]+".fits"
    draw_map(df_run1, cols[i], outfile, wht_data.shape,
             xyid=False, vorbins=vbin_data)

### Run 2
for i in range(len(cols)):
    outfile = str(Path(dir_output))+"/Map_run2_"+out_suffix[i]+".fits"
    draw_map(df_run2, cols[i], outfile, wht_data.shape,
             xyid=False, vorbins=vbin_data)

### Run 3
for i in range(len(cols)):
    outfile = str(Path(dir_output))+"/Map_run3_"+out_suffix[i]+".fits"
    draw_map(df_run3, cols[i], outfile, wht_data.shape,
             xyid=False, vorbins=vbin_data)

### Run 4
for i in range(len(cols)):
    outfile = str(Path(dir_output))+"/Map_run4_"+out_suffix[i]+".fits"
    draw_map(df_run4, cols[i], outfile, wht_data.shape,
             xyid=False, vorbins=vbin_data)
'''

# for i in range(len(cols)):
#     outfile = str(Path(dir_output))+"/Map_ppxf0_"+out_suffix[i]+".fits"
#     draw_map(df0, cols[i], outfile, img_rebin.shape)
# for i in range(len(cols)):
#     outfile = str(Path(dir_output))+"/Map_ppxf1_"+out_suffix[i]+".fits"
#     draw_map(df1, cols[i], outfile, img_rebin.shape)
# for i in range(len(cols)):
#     outfile = str(Path(dir_output))+"/Map_ppxf2_"+out_suffix[i]+".fits"
#     draw_map(df2, cols[i], outfile, img_rebin.shape)


### Original images
print("----- Unbinned -----")
vmin, vmax = np.percentile(wht_data, [2.0, 98.0])
plot_2Dmap(wht_data, "White map (NGC 628)", vmin, vmax, str(Path(dir_output))+"/Map_white.png",
           cmap='gray_r', add_cb=False, add_or=True, add_sc=True,
           x0=-80., y0=80., sign=-1, L=30., theta0=0.0*(np.pi/180.0),
           xN=-85., yN=120., xE=-130., yE=75., pixel_scale=0.2)

im = fits.getdata("NGC0628_MAPS_copt_0.92asec.fits", ext=2)
im += v0
vmin, vmax = 627.5, 672.5    #np.percentile(im[~np.isnan(im)], [50.0, 95.0])
im[(wht_SNR <= 100.) | (np.isnan(wht_SNR)) | (im == 0)] = np.nan
plot_2Dmap(im, "Radial velocity map (NGC 628)", vmin, vmax, str(Path(dir_output))+"/Map_unbinned_vel_star.png",
           cmap='viridis', cb_label=r"$v_{\rm rad}~{\rm [km~s^{-1}]}$", add_cb=True, add_or=False, add_sc=False, pixel_scale=0.2)
print(f"V_rad (star): {np.average(im[~np.isnan(im)], weights=wht_data[~np.isnan(im)]):.2f} km/s")

im = fits.getdata("NGC0628_MAPS_copt_0.92asec.fits", ext=4)
vmin, vmax = 25., 75.   #np.percentile(im[~np.isnan(im)], [50.0, 95.0])
im[(wht_SNR <= 100.) | (np.isnan(wht_SNR)) | (im == 0)] = np.nan
plot_2Dmap(im, "Velocity dispersion map (NGC 628)", vmin, vmax, str(Path(dir_output))+"/Map_unbinned_vdisp_star.png",
           cmap='viridis', cb_label=r"$\sigma_{v}({\rm star})~{\rm [km~s^{-1}]}$", add_cb=True, add_or=False, add_sc=False, pixel_scale=0.2)
print(f"V_disp (star): {np.average(im[~np.isnan(im)], weights=wht_data[~np.isnan(im)]):.2f} km/s")


### Unbinned images
run_name = ["run1", "run2", "run3", "run4"]
col_name = ["ML", "vel_star", "sigma_star",
            "logAge_lw", "logAge_mw", "Z_lw", "Z_mw",
            "Av_star"]
map_name = [r"$M/L_{r}$", "Radial velocity", "Velocity dispersion",
            "Age (LW)", "Age (MW)", r"[$Z/Z_{\odot}$] (LW)", r"[$Z/Z_{\odot}$] (MW)",
            r"$A_{V,\ast}$"]
colorbar = [r"$M/L_{r}$", r"$v_{\rm rad}~{\rm [km~s^{-1}]}$", r"$\sigma_{v}({\rm star})~{\rm [km~s^{-1}]}$",
            "log Age [yr]", "log Age [yr]", r"$[Z/Z_{\odot}]$", r"$[Z/Z_{\odot}]$",
            r"$A_{V,\ast}$ [mag]"]
vmins = [0.5, 627.5, 25.,  8.9,  8.9, -1.25, -1.25, 0.0]
vmaxs = [2.5, 672.5, 75., 10.1, 10.1,  0.25,  0.25, 1.0]
for ppxf_run in run_name:
    print("\n----- "+ppxf_run+" -----")
    for i in range(len(col_name)):
        im = fits.getdata(str(Path(dir_output))+"/Map_"+ppxf_run+"_"+col_name[i]+".fits")
        im[(wht_SNR <= 10.) | (np.isnan(wht_SNR)) | (im == 0)] = np.nan
        plot_2Dmap(im, map_name[i]+" map (NGC 628)", vmins[i], vmaxs[i],
                   str(Path(dir_output))+"/Map_"+ppxf_run+"_"+col_name[i]+".png",
                   cmap='viridis', cb_label=colorbar[i],
                   add_cb=True, add_or=False, add_sc=False, pixel_scale=0.2)
        print(col_name[i]+f": {np.average(im[~np.isnan(im)], weights=wht_data[~np.isnan(im)]):.3f}")




    
