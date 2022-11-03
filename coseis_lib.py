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

def run_okada(model, LOS_vector):

    # Set up grid
    x = np.arange(-25000, 25000, 100)
    y = np.arange(-25000, 25000, 100)
    xx, yy = np.meshgrid(x, y)
    xx_vec = np.reshape(xx, -1)
    yy_vec = np.reshape(yy, -1)
    coords = np.vstack((xx_vec, yy_vec))

    # Elastic params
    e1 = 3.2e10     # lambda
    e2 = 3.2e10     # mu

    # Calcualte displacements
    U = disloc3d3(model, coords, e1, e2)

    # Regrid
    xgrid = np.reshape(U[0,:],xx.shape)
    ygrid = np.reshape(U[1,:],xx.shape)
    zgrid = np.reshape(U[2,:],xx.shape)

    # Apply LOS
    los_grid = (xgrid * LOS_vector[0]) + (ygrid * LOS_vector[1]) + (zgrid * LOS_vector[2]) # source is 1 1 3 ??
    los_grid_wrap = np.mod(los_grid + 10000, 0.028333)

    # Fault parameters for plotting
    end1x = 1 + np.sin(np.radians(model[0])) * model[4]/2
    end2x = 1 - np.sin(np.radians(model[0])) * model[4]/2
    end1y = 1 + np.cos(np.radians(model[0])) * model[4]/2
    end2y = 1 - np.cos(np.radians(model[0])) * model[4]/2
    c1x = end1x + np.sin(np.radians(model[0]+90)) * np.cos(np.radians(model[0])) * model[5]
    c2x = end1x + np.sin(np.radians(model[0]+90)) * np.cos(np.radians(model[0])) * model[6]
    c3x = end2x + np.sin(np.radians(model[0]+90)) * np.cos(np.radians(model[0])) * model[6]
    c4x = end2x + np.sin(np.radians(model[0]+90)) * np.cos(np.radians(model[0])) * model[5]
    c1y = end1y + np.cos(np.radians(model[0]+90)) * np.cos(np.radians(model[0])) * model[5]
    c2y = end1y + np.cos(np.radians(model[0]+90)) * np.cos(np.radians(model[0])) * model[6]
    c3y = end2y + np.cos(np.radians(model[0]+90)) * np.cos(np.radians(model[0])) * model[6]
    c4y = end2y + np.cos(np.radians(model[0]+90)) * np.cos(np.radians(model[0])) * model[5]

    # Plot unwrapped
    fig, ax = plt.subplots(2, 2)
    extent = (-25, 25, -25, 25)
    im_unw = ax[0, 0].imshow(los_grid, extent=extent, origin='lower')
    ax[0, 0].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='White')
    ax[0, 0].scatter(end1x/1000, end1y/1000, color='white')
    ax[0, 0].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='White')
    fig.colorbar(im_unw, ax=ax[0, 0], label='metres')
    ax[0, 0].set_xlabel('Easting (km)')
    ax[0, 0].set_ylabel('Northing (km)')

    # Plot wrapped
    im_w = ax[0, 1].imshow(los_grid_wrap/0.028333*2*np.pi-np.pi, extent=extent, origin='lower')
    ax[0, 1].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='White')
    ax[0, 1].scatter(end1x/1000, end1y/1000, color='white')
    ax[0, 1].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='White')
    fig.colorbar(im_w, ax=ax[0, 1], label='radians')
    ax[0, 1].set_xlabel('Easting (km)')
    ax[0, 1].set_ylabel('Northing (km)')

    # Plot 3D deformation
    im_z = ax[1, 0].imshow(zgrid, extent=extent, origin='lower')
    ax[1, 0].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='White')
    ax[1, 0].scatter(end1x/1000, end1y/1000, color='white')
    ax[1, 0].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='White')
    fig.colorbar(im_z, ax=ax[1, 0], label='Vertical deformation (m)')
    ax[1, 0].quiver(xx[24::25, 24::25]/1000, yy[24::25, 24::25]/1000, xgrid[24::25, 24::25], ygrid[24::25, 24::25], scale=2, color='White')
    ax[1, 0].set_xlabel('Easting (km)')
    ax[1, 0].set_ylabel('Northing (km)')

    # Plot profile through y=0
    ax[1, 1].plot(x/1000, zgrid[250, :], label='Vertical')
    ax[1, 1].plot(x/1000, xgrid[250, :], label='East')
    ax[1, 1].plot(x/1000, ygrid[250, :], label='North')
    ax[1, 1].plot(x/1000, los_grid[250, :], label='LOS')
    ax[1, 1].legend()
    ax[1, 0].set_xlabel('Easting (km)')
    ax[1, 0].set_ylabel('Displacement (m)')

    plt.show(fig)

    return #success, u, grad_u

#-------------------------------------------------------------------------------

def disloc3d3(m, x, e1, e2):
    '''

    Assumes a single fault plan

    '''

    nx = x.shape[1]

    # Calc alpha
    alpha = (e1 + e2) / (e1 + 2*e2)

    # Convert into dc3dwrapper inputs
    flt_x = np.ones(nx)
    flt_y = np.ones(nx)
    strike = m[0]
    dip = m[1]
    rake = m[2]
    slip = m[3]
    length = m[4]
    hmin = m[5]
    hmax = m[6]

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
