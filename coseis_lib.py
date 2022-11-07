#!/usr/bin/env python3
'''
## coseis_lib.py

Library of python functions to be used with the coseismic_practical.

'''

# Packages
import numpy as np
import matplotlib.pyplot as plt
from okada_wrapper import dc3dwrapper

#-------------------------------------------------------------------------------

def disloc3d3(m, x, e1, e2):
    '''

    Assumes a single fault plan

    '''

    nx = x.shape[1]
    
    # apply shift in fault location
    x[0,:] = x[0,:] - m[0]
    x[1,:] = x[1,:] - m[1]
    
    # Calc alpha
    alpha = (e1 + e2) / (e1 + 2*e2)

    # Convert into dc3dwrapper inputs
    flt_x = np.ones(nx)
    flt_y = np.ones(nx)
    strike = m[2]
    dip = m[3]
    rake = m[4]
    slip = m[5]
    length = m[6]
    hmin = m[7]
    hmax = m[8]

    sstrike = np.radians(strike+90)
    ct = np.cos(sstrike);
    st = np.sin(sstrike);
    rrake = np.radians(rake+90)
    sindip = np.sin(np.radians(dip))
    w = (hmax - hmin) / sindip
    ud = np.ones(nx) * slip * np.cos(rrake)
    us = np.ones(nx) * -slip * np.sin(rrake)
    halflen = length / 2
    al2 = np.ones(nx) * halflen
    al1 = -al2
    aw1 = np.ones(nx) * hmin / sindip
    aw2 = np.ones(nx) * hmax / sindip

    # Check that fault isn't above surface
    assert hmin > 0

    # Loop over points
    X = ct * (-flt_x + x[0,:]) - st * (-flt_y + x[1,:])
    Y = ct * (-flt_y + x[1,:]) + st * (-flt_x + x[0,:])

    U = np.zeros((3,nx))
    u_xyz = np.zeros((3,nx))

    for ii in range(nx):

        # Run okada model for point
        success, u, grad_u = dc3dwrapper(alpha, [X[ii], Y[ii], 0], hmin, -dip,
            [al1[ii], al2[ii]], [aw1[ii], aw2[ii]], [us[ii], ud[ii], 0])

        u_xyz[0,ii] = u[0]
        u_xyz[1,ii] = u[1]
        u_xyz[2,ii] = u[2]

    # Format output
    U[0,:] = ct*u_xyz[0,:] + st*u_xyz[1,:]
    U[1,:] = -st*u_xyz[0,:] + ct*u_xyz[1,:]
    U[2,:] = u_xyz[2,:]

    return U

#-------------------------------------------------------------------------------

def plot_enu(U, model, x, y):
    '''
    Plot East, North, and Up displacements from disloc3d3.
    '''
    
    # convert to km for better plotting
    x = x / 1000
    y = y / 1000
    
    # convert to mm
    U = U * 1000
    
    # coord grids
    xx, yy = np.meshgrid(x, y)
    
    # Regrid displacements
    xgrid = np.reshape(U[0,:],xx.shape)
    ygrid = np.reshape(U[1,:],xx.shape)
    zgrid = np.reshape(U[2,:],xx.shape)
    
    # Fault outline for plotting
    end1x, end2x, end1y, end2y, c1x, c2x, c3x, c4x, c1y, c2y, c3y, c4y = fault_for_plotting(model)
    
    # Setup plot
    fig, ax = plt.subplots(2, 2, figsize=(20, 18))
    extent = (x[0], x[-1], y[0], y[-1])
    
    # Plot East
    im_e = ax[0,0].imshow(xgrid, extent=extent, origin='lower')
    ax[0,0].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='White')
    ax[0,0].scatter(end1x/1000, end1y/1000, color='white')
    ax[0,0].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='White')
    fig.colorbar(im_e, ax=ax[0,0])
    ax[0,0].set_xlabel('Easting (km)')
    ax[0,0].set_ylabel('Northing (km)')
    ax[0,0].set_title('East displacement (mm)')
    
    # Plot North
    im_n = ax[0,1].imshow(ygrid, extent=extent, origin='lower')
    ax[0,1].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='White')
    ax[0,1].scatter(end1x/1000, end1y/1000, color='white')
    ax[0,1].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='White')
    fig.colorbar(im_n, ax=ax[0,1])
    ax[0,1].set_xlabel('Easting (km)')
    ax[0,1].set_ylabel('Northing (km)')
    ax[0,1].set_title('North displacement (mm)')
    
    # Plot Up
    im_u = ax[1,0].imshow(zgrid, extent=extent, origin='lower')
    ax[1,0].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='White')
    ax[1,0].scatter(end1x/1000, end1y/1000, color='white')
    ax[1,0].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='White')
    fig.colorbar(im_u, ax=ax[1,0])
    ax[1,0].set_xlabel('Easting (km)')
    ax[1,0].set_ylabel('Northing (km)')
    ax[1,0].set_title('Vertical displacement (mm)')
    
    # Plot 3D deformation
    im_3d = ax[1,1].imshow(zgrid, extent=extent, origin='lower')
    ax[1,1].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='White')
    ax[1,1].scatter(end1x/1000, end1y/1000, color='white')
    ax[1,1].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='White')
    fig.colorbar(im_3d, ax=ax[1,1], label='Vertical displacement (mm)')
    ax[1,1].quiver(xx[24::25, 24::25], yy[24::25, 24::25], xgrid[24::25, 24::25]/1000, ygrid[24::25, 24::25]/1000, scale=1, color='White')
    ax[1,1].set_xlabel('Easting (km)')
    ax[1,1].set_ylabel('Northing (km)')
    ax[1,1].set_title('3D displacement (mm)')
    

#-------------------------------------------------------------------------------

def plot_los(U, model, x, y, e2los, n2los, u2los):
    '''
    Plot line-of-sight displacements from East, North, and Up displacements.
    '''
    
    # convert to km for better plotting
    x = x / 1000
    y = y / 1000
    
    # coord grids
    xx, yy = np.meshgrid(x, y)
    
    # Regrid displacements
    xgrid = np.reshape(U[0,:],xx.shape)
    ygrid = np.reshape(U[1,:],xx.shape)
    zgrid = np.reshape(U[2,:],xx.shape)
    
    # Convert to LOS
    los_grid = (xgrid * e2los) + (ygrid * n2los) + (zgrid * u2los)
    los_grid_wrap = np.mod(los_grid + 10000, 0.028333)
    
    # Fault outline for plotting
    end1x, end2x, end1y, end2y, c1x, c2x, c3x, c4x, c1y, c2y, c3y, c4y = fault_for_plotting(model)
    
    # Setup plot
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    extent = (x[0], x[-1], y[0], y[-1])
    
    # Plot Unwrapped
    im_u = ax[0].imshow(los_grid*1000, extent=extent, origin='lower')
    ax[0].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='White')
    ax[0].scatter(end1x/1000, end1y/1000, color='white')
    ax[0].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='White')
    fig.colorbar(im_u, ax=ax[0])
    ax[0].set_xlabel('Easting (km)')
    ax[0].set_ylabel('Northing (km)')
    ax[0].set_title('Unwrapped LOS displacement (mm)')
    
    # Plot Wrapped
    im_w = ax[1].imshow(los_grid_wrap/0.028333*2*np.pi-np.pi, extent=extent, origin='lower')
    ax[1].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='White')
    ax[1].scatter(end1x/1000, end1y/1000, color='white')
    ax[1].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='White')
    fig.colorbar(im_w, ax=ax[1])
    ax[1].set_xlabel('Easting (km)')
    ax[1].set_ylabel('Northing (km)')
    ax[1].set_title('Wrapped LOS displacement (radians)')
    
    
#-------------------------------------------------------------------------------

def plot_data_model(x, y, U, model, data_unw, e2los, n2los, u2los):
    '''
    Compare modelled LOS displacements with wrapped and unwrapped intererograms.
    '''
    
    # convert to km for better plotting
    x = x / 1000
    y = y / 1000
    
    # coord grids
    xx, yy = np.meshgrid(x, y)
    
    # Regrid displacements
    xgrid = np.reshape(U[0,:],xx.shape)
    ygrid = np.reshape(U[1,:],xx.shape)
    zgrid = np.reshape(U[2,:],xx.shape)
    
    # Convert to LOS
    los_grid = (xgrid * e2los) + (ygrid * n2los) + (zgrid * u2los)
    los_grid_wrap = np.mod(los_grid + 10000, 0.028333)
    
    # Rewrap the original
    data_diff = np.mod(data_unw + 10000, 0.028333)
    
    # Calculate residual
    resid = data_unw - los_grid
    resid_wrapped = np.mod(resid + 10000, 0.028333)
    
    # Fault outline for plotting
    end1x, end2x, end1y, end2y, c1x, c2x, c3x, c4x, c1y, c2y, c3y, c4y = fault_for_plotting(model)
    
    # Setup plot
    fig, ax = plt.subplots(3, 2, figsize=(23, 30))
    extent = (x[0], x[-1], y[0], y[-1])
    
    # Plot unwrapped data
    im_u = ax[0,0].imshow(data_unw*1000, extent=extent, origin='lower')
    ax[0,0].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='black')
    ax[0,0].scatter(end1x/1000, end1y/1000, color='black')
    ax[0,0].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='black')
    fig.colorbar(im_u, ax=ax[0,0])
    ax[0,0].set_xlabel('Easting (km)')
    ax[0,0].set_ylabel('Northing (km)')
    ax[0,0].set_title('Unwrapped interferogram (mm)')
    
    # Plot unwrapped model
    im_u = ax[1,0].imshow(los_grid*1000, extent=extent, origin='lower')
    ax[1,0].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='black')
    ax[1,0].scatter(end1x/1000, end1y/1000, color='black')
    ax[1,0].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='black')
    fig.colorbar(im_u, ax=ax[1,0])
    ax[1,0].set_xlabel('Easting (km)')
    ax[1,0].set_ylabel('Northing (km)')
    ax[1,0].set_title('Unwrapped model (mm)')
    
    # Plot unwrapped residual
    im_u = ax[2,0].imshow(resid*1000, extent=extent, origin='lower')
    ax[2,0].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='black')
    ax[2,0].scatter(end1x/1000, end1y/1000, color='black')
    ax[2,0].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='black')
    fig.colorbar(im_u, ax=ax[2,0])
    ax[2,0].set_xlabel('Easting (km)')
    ax[2,0].set_ylabel('Northing (km)')
    ax[2,0].set_title('Unwrapped residual (mm)')
    
    # Plot wrapped data
    im_u = ax[0,1].imshow(data_diff/0.028333*2*np.pi-np.pi, extent=extent, origin='lower')
    ax[0,1].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='black')
    ax[0,1].scatter(end1x/1000, end1y/1000, color='black')
    ax[0,1].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='black')
    fig.colorbar(im_u, ax=ax[0,1])
    ax[0,1].set_xlabel('Easting (km)')
    ax[0,1].set_ylabel('Northing (km)')
    ax[0,1].set_title('Wrapped interferogram (mm)')
    
    # Plot wrapped model
    im_u = ax[1,1].imshow(los_grid_wrap/0.028333*2*np.pi-np.pi, extent=extent, origin='lower')
    ax[1,1].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='black')
    ax[1,1].scatter(end1x/1000, end1y/1000, color='black')
    ax[1,1].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='black')
    fig.colorbar(im_u, ax=ax[1,1])
    ax[1,1].set_xlabel('Easting (km)')
    ax[1,1].set_ylabel('Northing (km)')
    ax[1,1].set_title('Wrapped model (mm)')
    
    # Plot wrapped residual
    im_u = ax[2,1].imshow(resid_wrapped/0.028333*2*np.pi-np.pi, extent=extent, origin='lower')
    ax[2,1].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='black')
    ax[2,1].scatter(end1x/1000, end1y/1000, color='black')
    ax[2,1].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='black')
    fig.colorbar(im_u, ax=ax[2,1])
    ax[2,1].set_xlabel('Easting (km)')
    ax[2,1].set_ylabel('Northing (km)')
    ax[2,1].set_title('Wrapped residual (mm)')
    

#-------------------------------------------------------------------------------

def fault_for_plotting(model):
    '''
    Get trace and projected corners of fault for plotting.
    '''
    
    end1x = model[0] + np.sin(np.deg2rad(model[2])) * model[6]/2
    end2x = model[0] - np.sin(np.deg2rad(model[2])) * model[6]/2
    end1y = model[1] + np.cos(np.deg2rad(model[2])) * model[6]/2
    end2y = model[1] - np.cos(np.deg2rad(model[2])) * model[6]/2
    
    c1x = end1x + np.sin(np.deg2rad(model[2]+90)) * (model[7] / np.tan(np.deg2rad(model[3])))
    c2x = end1x + np.sin(np.deg2rad(model[2]+90)) * (model[8] / np.tan(np.deg2rad(model[3])))
    c3x = end2x + np.sin(np.deg2rad(model[2]+90)) * (model[8] / np.tan(np.deg2rad(model[3])))
    c4x = end2x + np.sin(np.deg2rad(model[2]+90)) * (model[7] / np.tan(np.deg2rad(model[3])))
    c1y = end1y + np.cos(np.deg2rad(model[2]+90)) * (model[7] / np.tan(np.deg2rad(model[3])))
    c2y = end1y + np.cos(np.deg2rad(model[2]+90)) * (model[8] / np.tan(np.deg2rad(model[3])))
    c3y = end2y + np.cos(np.deg2rad(model[2]+90)) * (model[8] / np.tan(np.deg2rad(model[3])))
    c4y = end2y + np.cos(np.deg2rad(model[2]+90)) * (model[7] / np.tan(np.deg2rad(model[3])))
    
    return end1x, end2x, end1y, end2y, c1x, c2x, c3x, c4x, c1y, c2y, c3y, c4y

#-------------------------------------------------------------------------------

def get_par(par_file,par_name):
    '''
    Returns the value of the requested parameter in the parameter file.
    
    INPUTS
        par_file = name of param file (str)
        par_name = name of desired par (str)
    OUTPUTS
        par_val = value of param for par file
    '''
    
    with open(par_file, 'r') as f:
        for line in f.readlines():
            if par_name in line:
                par_val = line.split()[1].strip()
    
    return par_val