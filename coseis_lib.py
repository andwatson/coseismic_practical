#!/usr/bin/env python3
'''
## coseis_lib.py

Library of python functions to be used with the coseismic_practical.

'''

# Packages
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm

eps = 1e-14

#-------------------------------------------------------------------------------

def disloc3d3(x, y, xoff=0, yoff=0,
            depth=5e3, length=1e3, width=1e3, 
            slip=0.0, opening=10.0, 
            strike=0.0, dip=0.0, rake=0.0,
            nu=0.25):
    '''
    Calculate surface displacements for Okada85 dislocation model
    
    Original version at "https://github.com/scottyhq/roipy"
    
    %--------------------------------------------------------------
    OKADA85 Surface deformation due to a finite rectangular source.
    [uE,uN,uZ,uZE,uZN,uNN,uNE,uEN,uEE] = OKADA85(...
       E,N,DEPTH,STRIKE,DIP,LENGTH,WIDTH,RAKE,SLIP,OPEN)
    computes displacements, tilts and strains at the surface of an elastic
    half-space, due to a dislocation defined by RAKE, SLIP, and OPEN on a
    rectangular fault defined by orientation STRIKE and DIP, and size LENGTH and
    WIDTH. The fault centroid is located (0,0,-DEPTH).

       E,N    : coordinates of observation points in a geographic referential
                (East,North,Up) relative to fault centroid (units are described below)
       DEPTH  : depth of the fault centroid (DEPTH > 0)
       STRIKE : fault trace direction (0 to 360 relative to North), defined so
                that the fault dips to the right side of the trace
       DIP    : angle between the fault and a horizontal plane (0 to 90)
       LENGTH : fault length in the STRIKE direction (LENGTH > 0)
       WIDTH  : fault width in the DIP direction (WIDTH > 0)
       RAKE   : direction the hanging wall moves during rupture, measured relative
                to the fault STRIKE (-180 to 180).
       SLIP   : dislocation in RAKE direction (length unit)
       OPEN   : dislocation in tensile component (same unit as SLIP)

    returns the following variables (same matrix size as E and N):
       uN,uE,uZ        : displacements (unit of SLIP and OPEN)
    Orginal matlab function from:
    http://www.mathworks.com/matlabcentral/fileexchange/25982-okada--surface-deformation-due-to-a-finite-rectangular-source/content/okada85.m
    
    '''
    
    # check that top of fault isn't above 0 m surface
    top_depth = depth - (width/2) * np.sin(np.deg2rad(dip))
    assert top_depth >= np.float(0), "Fault breaches 0 m surface, please change either centroid depth or fault width."        
    
    x = x - xoff
    y = y - yoff
    e = x
    n = y
    
    strike = np.deg2rad(strike)
    dip = np.deg2rad(dip)
    rake = np.deg2rad(rake)
    
    L = length
    W = width

    U1 = np.cos(rake) * slip
    U2 = np.sin(rake) * slip
    U3 = opening
    
    d = depth + np.sin(dip) * W / 2
    ec = e + np.cos(strike) * np.cos(dip) * W / 2
    nc = n - np.sin(strike) * np.cos(dip) * W / 2
    x = np.cos(strike) * nc + np.sin(strike) * ec + L / 2
    y = np.sin(strike) * nc - np.cos(strike) * ec + np.cos(dip) * W
    p = y * np.cos(dip) + d * np.sin(dip)
    q = y * np.sin(dip) - d * np.cos(dip)

    ux = - U1 / (2 * np.pi) * chinnery(ux_ss, x, p, L, W, q, dip, nu) - \
           U2 / (2 * np.pi) * chinnery(ux_ds, x, p, L, W, q, dip, nu) + \
           U3 / (2 * np.pi) * chinnery(ux_tf, x, p, L, W, q, dip, nu)
    uy = - U1 / (2 * np.pi) * chinnery(uy_ss, x, p, L, W, q, dip, nu) - \
           U2 / (2 * np.pi) * chinnery(uy_ds, x, p, L, W, q, dip, nu) + \
           U3 / (2 * np.pi) * chinnery(uy_tf, x, p, L, W, q, dip, nu)
    uz = - U1 / (2 * np.pi) * chinnery(uz_ss, x, p, L, W, q, dip, nu) - \
           U2 / (2 * np.pi) * chinnery(uz_ds, x, p, L, W, q, dip, nu) + \
           U3 / (2 * np.pi) * chinnery(uz_tf, x, p, L, W, q, dip, nu)
    ue = np.sin(strike) * ux - np.cos(strike) * uy
    un = np.cos(strike) * ux + np.sin(strike) * uy

    return np.vstack((ue, un, uz))

'''
Notes for I... and K... subfunctions:
    1. original formulas use Lame's parameters as mu/(mu+lambda) which
       depends only on the Poisson's ratio = 1 - 2*nu
    2. tests for cos(dip) == 0 are made with "cos(dip) > eps"
       because cos(90*np.pi/180) is not zero but = 6.1232e-17 (!)
       NOTE: don't use cosd and sind because of incompatibility
       with Matlab v6 and earlier...
'''

def chinnery(f, x, p, L, W, q, dip, nu):
    ''' % Chinnery's notation [equation (24) p. 1143]'''
    u = ( f(x, p, q, dip, nu) -
          f(x, p - W, q, dip, nu) -
          f(x - L, p, q, dip, nu) +
          f(x - L, p - W, q, dip, nu) )
    return u

'''
Displacement subfunctions
strike-slip displacement subfunctions [equation (25) p. 1144]
'''

def ux_ss(xi, eta, q, dip, nu):

    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    u = xi * q / (R * (R + eta)) + \
        I1(xi, eta, q, dip, nu, R) * np.sin(dip)
    k = (q != 0)
    #u[k] = u[k] + np.arctan2( xi[k] * (eta[k]) , (q[k] * (R[k])))
    u[k] = u[k] + np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]) )
    return u


def uy_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    u = (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + eta)) + \
        q * np.cos(dip) / (R + eta) + \
        I2(eta, q, dip, nu, R) * np.sin(dip)
    return u


def uz_ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    u = (eta * np.sin(dip) - q * np.cos(dip)) * q / (R * (R + eta)) + \
        q * np.sin(dip) / (R + eta) + \
        I4(db, eta, q, dip, nu, R) * np.sin(dip)
    return u


def ux_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    u = q / R - \
        I3(eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip)
    return u


def uy_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    u = ( (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + xi)) -
           I1(xi, eta, q, dip, nu, R) * np.sin(dip) * np.cos(dip) )
    k = (q != 0)
    u[k] = u[k] + np.cos(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]))
    return u


def uz_ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    u = ( db * q / (R * (R + xi)) -
          I5(xi, eta, q, dip, nu, R, db) * np.sin(dip) * np.cos(dip) )
    k = (q != 0)
    #u[k] = u[k] + np.sin(dip) * np.arctan2(xi[k] * eta[k] , q[k] * R[k])
    u[k] = u[k] + np.sin(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]))
    return u


def ux_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    u = q ** 2 / (R * (R + eta)) - \
        I3(eta, q, dip, nu, R) * (np.sin(dip) ** 2)
    return u


def uy_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
    u = - (eta * np.sin(dip) - q * np.cos(dip)) * q / (R * (R + xi)) - \
        np.sin(dip) * xi * q / (R * (R + eta)) - \
        I1(xi, eta, q, dip, nu, R) * (np.sin(dip) ** 2)
    k = (q != 0)
    #u[k] = u[k] + np.sin(dip) * np.arctan2(xi[k] * eta[k] , q[k] * R[k])
    u[k] = u[k] + np.sin(dip) * np.arctan( (xi[k] * eta[k]) , (q[k] * R[k]) )
    return u


def uz_tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    db = eta * np.sin(dip) - q * np.cos(dip)
    u = (eta * np.cos(dip) + q * np.sin(dip)) * q / (R * (R + xi)) + \
         np.cos(dip) * xi * q / (R * (R + eta)) - \
         I5(xi, eta, q, dip, nu, R, db) * np.sin(dip)**2
    k = (q != 0) #not at depth=0?
    u[k] = u[k] - np.cos(dip) * np.arctan( (xi[k] * eta[k]) / (q[k] * R[k]) )
    return u


def I1(xi, eta, q, dip, nu, R):
    db = eta * np.sin(dip) - q * np.cos(dip)
    if np.cos(dip) > eps:
        I = (1 - 2 * nu) * (- xi / (np.cos(dip) * (R + db))) - \
            np.sin(dip) / np.cos(dip) * \
            I5(xi, eta, q, dip, nu, R, db)
    else:
        I = -(1 - 2 * nu) / 2 * xi * q / (R + db) ** 2
    return I


def I2(eta, q, dip, nu, R):
    I = (1 - 2 * nu) * (-np.log(R + eta)) - \
        I3(eta, q, dip, nu, R)
    return I


def I3(eta, q, dip, nu, R):
    yb = eta * np.cos(dip) + q * np.sin(dip)
    db = eta * np.sin(dip) - q * np.cos(dip)
    if np.cos(dip) > eps:
        I = (1 - 2 * nu) * (yb / (np.cos(dip) * (R + db)) - np.log(R + eta)) + \
            np.sin(dip) / np.cos(dip) * \
            I4(db, eta, q, dip, nu, R)
    else:
        I = (1 - 2 * nu) / 2 * (eta / (R + db) + yb * q / (R + db) ** 2 - np.log(R + eta))
    return I


def I4(db, eta, q, dip, nu, R):
    if np.cos(dip) > eps:
        I = (1 - 2 * nu) * 1.0 / np.cos(dip) * \
            (np.log(R + db) - np.sin(dip) * np.log(R + eta))
    else:
        I = - (1 - 2 * nu) * q / (R + db)
    return I


def I5(xi, eta, q, dip, nu, R, db):
    X = np.sqrt(xi**2 + q**2)
    if np.cos(dip) > eps:
        with np.errstate(divide='ignore'):
            I = (1 - 2 * nu) * 2 / np.cos(dip) * \
                 np.arctan( (eta * (X + q*np.cos(dip)) + X*(R + X) * np.sin(dip)) /
                            (xi*(R + X) * np.cos(dip)) ) 
        I[xi == 0] = 0
    else:
        I = -(1 - 2 * nu) * xi * np.sin(dip) / (R + db)
    return I

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
    im_e = ax[0,0].imshow(xgrid, extent=extent, origin='lower', cmap=cm.batlow)
    ax[0,0].contour(xx, yy, xgrid, linestyles='dashed', colors='white')
    ax[0,0].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='Black')
    ax[0,0].scatter(end1x/1000, end1y/1000, color='Black')
    ax[0,0].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='Black')
    fig.colorbar(im_e, ax=ax[0,0])
    ax[0,0].set_xlabel('Easting (km)')
    ax[0,0].set_ylabel('Northing (km)')
    ax[0,0].set_title('East displacement (mm)')
    
    # Plot North
    im_n = ax[0,1].imshow(ygrid, extent=extent, origin='lower', cmap=cm.batlow)
    ax[0,1].contour(xx, yy, ygrid, linestyles='dashed', colors='white')
    ax[0,1].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='Black')
    ax[0,1].scatter(end1x/1000, end1y/1000, color='Black')
    ax[0,1].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='Black')
    fig.colorbar(im_n, ax=ax[0,1])
    ax[0,1].set_xlabel('Easting (km)')
    ax[0,1].set_ylabel('Northing (km)')
    ax[0,1].set_title('North displacement (mm)')
    
    # Plot Up
    im_u = ax[1,0].imshow(zgrid, extent=extent, origin='lower', cmap=cm.batlow)
    ax[1,0].contour(xx, yy, zgrid, linestyles='dashed', colors='white')
    ax[1,0].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='Black')
    ax[1,0].scatter(end1x/1000, end1y/1000, color='Black')
    ax[1,0].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='Black')
    fig.colorbar(im_u, ax=ax[1,0])
    ax[1,0].set_xlabel('Easting (km)')
    ax[1,0].set_ylabel('Northing (km)')
    ax[1,0].set_title('Vertical displacement (mm)')
    
    # Plot 3D deformation
    im_3d = ax[1,1].imshow(zgrid, extent=extent, origin='lower', cmap=cm.batlow)
    ax[1,1].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='Black')
    ax[1,1].scatter(end1x/1000, end1y/1000, color='Black')
    ax[1,1].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='Black')
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
    im_u = ax[0].imshow(los_grid*1000, extent=extent, origin='lower', cmap=cm.batlow)
    ax[0].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='White')
    ax[0].scatter(end1x/1000, end1y/1000, color='white')
    ax[0].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='White')
    fig.colorbar(im_u, ax=ax[0])
    ax[0].set_xlabel('Easting (km)')
    ax[0].set_ylabel('Northing (km)')
    ax[0].set_title('Unwrapped LOS displacement (mm)')
    
    # Plot Wrapped
    im_w = ax[1].imshow(los_grid_wrap/0.028333*2*np.pi-np.pi, extent=extent, origin='lower', cmap=cm.batlowK)
    ax[1].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='White')
    ax[1].scatter(end1x/1000, end1y/1000, color='white')
    ax[1].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='White')
    fig.colorbar(im_w, ax=ax[1])
    ax[1].set_xlabel('Easting (km)')
    ax[1].set_ylabel('Northing (km)')
    ax[1].set_title('Wrapped LOS displacement (radians)')
    
    
#-------------------------------------------------------------------------------

def plot_data_model(x, y, U, model, data_unw, e2los, n2los, u2los, show_grid=False):
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
    los_grid_wrap = np.mod(los_grid+10000, 0.028333)
    
    # Rewrap the original
    with np.errstate(all='ignore'):
        data_diff = np.mod(data_unw+10000, 0.028333)
    
    # Calculate residual
    resid = data_unw - los_grid
    with np.errstate(all='ignore'):
        resid_wrapped = np.mod(resid+10000, 0.028333)
    
    # Fault outline for plotting
    end1x, end2x, end1y, end2y, c1x, c2x, c3x, c4x, c1y, c2y, c3y, c4y = fault_for_plotting(model)
    
    # Calculate and print seismic moment
    seis_moment = 3e10 * model[5] * (model[6] * model[8]) # in Nm
    moment_mag = (2/3) * np.log10(seis_moment/1e-7) - 10.7
    
    print('Estimated seismic moment = {} Nm'.format(seis_moment))
    print('Estimated moment magnitude = {}'.format(round(moment_mag,2)))
    print('RMS misfit between data and model = {} mm'.format(round(rms_misfit(data_unw,los_grid)*1000,2)))
    
    # Setup plot
    fig, ax = plt.subplots(3, 2, figsize=(23, 30))
    extent = (x[0], x[-1], y[0], y[-1])
    
    # Plot unwrapped data
    im_u = ax[0,0].imshow(data_unw*1000, extent=extent, origin='lower', cmap=cm.batlow)
    ax[0,0].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='black')
    ax[0,0].scatter(end1x/1000, end1y/1000, color='black')
    if show_grid:
        ax[0,0].grid()
    ax[0,0].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='black')
    fig.colorbar(im_u, ax=ax[0,0])
    ax[0,0].set_xlabel('Easting (km)')
    ax[0,0].set_ylabel('Northing (km)')
    ax[0,0].set_title('Unwrapped interferogram (mm)')
    
    # Plot unwrapped model
    im_u = ax[1,0].imshow(los_grid*1000, extent=extent, origin='lower', cmap=cm.batlow)
    ax[1,0].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='black')
    ax[1,0].scatter(end1x/1000, end1y/1000, color='black')
    if show_grid:
        ax[1,0].grid()
    ax[1,0].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='black')
    fig.colorbar(im_u, ax=ax[1,0])
    ax[1,0].set_xlabel('Easting (km)')
    ax[1,0].set_ylabel('Northing (km)')
    ax[1,0].set_title('Unwrapped model (mm)')
    
    # Plot unwrapped residual
    im_u = ax[2,0].imshow(resid*1000, extent=extent, origin='lower', cmap=cm.batlow)
    ax[2,0].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='black')
    ax[2,0].scatter(end1x/1000, end1y/1000, color='black')
    if show_grid:
        ax[2,0].grid()
    ax[2,0].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='black')
    fig.colorbar(im_u, ax=ax[2,0])
    ax[2,0].set_xlabel('Easting (km)')
    ax[2,0].set_ylabel('Northing (km)')
    ax[2,0].set_title('Unwrapped residual (mm)')
    
    # Plot wrapped data
    im_u = ax[0,1].imshow(data_diff/0.028333*2*np.pi-np.pi, extent=extent, origin='lower', vmin=-3.14, vmax=3.14, cmap=cm.batlowK)
    ax[0,1].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='black')
    ax[0,1].scatter(end1x/1000, end1y/1000, color='black')
    if show_grid:
        ax[0,1].grid()
    ax[0,1].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='black')
    fig.colorbar(im_u, ax=ax[0,1])
    ax[0,1].set_xlabel('Easting (km)')
    ax[0,1].set_ylabel('Northing (km)')
    ax[0,1].set_title('Wrapped interferogram (mm)')
    
    # Plot wrapped model
    im_u = ax[1,1].imshow(los_grid_wrap/0.028333*2*np.pi-np.pi, extent=extent, origin='lower', vmin=-3.14, vmax=3.14, cmap=cm.batlowK)
    ax[1,1].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='black')
    ax[1,1].scatter(end1x/1000, end1y/1000, color='black')
    if show_grid:
        ax[1,1].grid()
    ax[1,1].plot(np.array([c1x, c2x, c3x, c4x, c1x])/1000, np.array([c1y, c2y, c3y, c4y, c1y])/1000, color='black')
    fig.colorbar(im_u, ax=ax[1,1])
    ax[1,1].set_xlabel('Easting (km)')
    ax[1,1].set_ylabel('Northing (km)')
    ax[1,1].set_title('Wrapped model (mm)')
    
    # Plot wrapped residual
    im_u = ax[2,1].imshow(resid_wrapped/0.028333*2*np.pi-np.pi, extent=extent, origin='lower', vmin=-3.14, vmax=3.14, cmap=cm.batlowK)
    ax[2,1].plot(np.array([end1x, end2x])/1000, np.array([end1y, end2y])/1000, color='black')
    ax[2,1].scatter(end1x/1000, end1y/1000, color='black')
    if show_grid:
        ax[2,1].grid()
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
    
    cen_offset = model[7]/np.tan(np.deg2rad(model[3]))
    
    trace_cen_x = model[0] - (cen_offset * np.cos(np.deg2rad(model[2])))
    trace_cen_y = model[1] + (cen_offset * np.sin(np.deg2rad(model[2])))
    
    top_depth = model[7] - (model[8]/2)*np.sin(np.deg2rad(model[3]))
    bottom_depth = model[7] + (model[8]/2)*np.sin(np.deg2rad(model[3]))
    
    end1x = trace_cen_x + np.sin(np.deg2rad(model[2])) * model[6]/2
    end2x = trace_cen_x - np.sin(np.deg2rad(model[2])) * model[6]/2
    end1y = trace_cen_y + np.cos(np.deg2rad(model[2])) * model[6]/2
    end2y = trace_cen_y - np.cos(np.deg2rad(model[2])) * model[6]/2
    
    c1x = end1x + np.sin(np.deg2rad(model[2]+90)) * (top_depth / np.tan(np.deg2rad(model[3])))
    c2x = end1x + np.sin(np.deg2rad(model[2]+90)) * (bottom_depth / np.tan(np.deg2rad(model[3])))
    c3x = end2x + np.sin(np.deg2rad(model[2]+90)) * (bottom_depth / np.tan(np.deg2rad(model[3])))
    c4x = end2x + np.sin(np.deg2rad(model[2]+90)) * (top_depth / np.tan(np.deg2rad(model[3])))
    c1y = end1y + np.cos(np.deg2rad(model[2]+90)) * (top_depth / np.tan(np.deg2rad(model[3])))
    c2y = end1y + np.cos(np.deg2rad(model[2]+90)) * (bottom_depth / np.tan(np.deg2rad(model[3])))
    c3y = end2y + np.cos(np.deg2rad(model[2]+90)) * (bottom_depth / np.tan(np.deg2rad(model[3])))
    c4y = end2y + np.cos(np.deg2rad(model[2]+90)) * (top_depth / np.tan(np.deg2rad(model[3])))
    
    return end1x, end2x, end1y, end2y, c1x, c2x, c3x, c4x, c1y, c2y, c3y, c4y

#-------------------------------------------------------------------------------

def comp2los(azimuth_angle, incidence_angle):
    '''
    Convert azimuth and incidence angle to unit vector components.
    '''
    
    e2los = np.cos(np.deg2rad(azimuth_angle)) * np.sin(np.deg2rad(incidence_angle))
    n2los = -np.sin(np.deg2rad(azimuth_angle)) * np.sin(np.deg2rad(incidence_angle))
    u2los = -np.cos(np.deg2rad(incidence_angle))
    
    return e2los, n2los, u2los

#-------------------------------------------------------------------------------

def load_ifgs(example_name, downsamp=False):
    '''
    Load of wrapped and unwrapped interferograms for a given example.
    Options are "iran", "greece", and "afghanistan".
    '''
    
    # set input file paths
    if example_name == 'iran':
        unw_file = 'data/iran/sarpol.unw'
        diff_file = 'data/iran/sarpol.diff'
        param_file = 'data/iran/sarpol.par'
            
    elif example_name == 'greece':
        unw_file = 'data/greece/greece.unw'
        diff_file = 'data/greece/greece.diff'
        param_file = 'data/greece/greece.par'
                
    elif example_name == 'afghanistan':
        unw_file = 'data/afghanistan/afghanistan.unw'
        diff_file = 'data/afghanistan/afghanistan.diff'
        param_file = 'data/afghanistan/afghanistan.par'
        
    else:
        print('Please provide name of example to load')
            
    
    # Read height and width in pixels from parameter file
    ifg_length = int(get_par(param_file,'length'))
    ifg_width = int(get_par(param_file,'width'))

    # Read coordinates of bottom corner
    corner_x = float(get_par(param_file,'corner_x'))
    corner_y = float(get_par(param_file,'corner_y'))

    # Read spacing of coordinates
    x_spacing = float(get_par(param_file,'x_spacing'))
    y_spacing = float(get_par(param_file,'y_spacing'))

    # Generate coordinate grids
    x = corner_x + x_spacing*np.arange(1,ifg_width+1) - x_spacing/2
    y = corner_y + y_spacing*np.arange(1,ifg_length+1) - y_spacing/2

    # Load the interferogram
    unw = np.fromfile(unw_file, dtype='float32').reshape((ifg_length, ifg_width))
    diff = np.fromfile(diff_file, dtype='float32').reshape((ifg_length, ifg_width))
    
    # Downsample Sarpol by selecting every other point, so as to increase run speed on binder.
    # Doing so here so that it's easy to change back.
    if downsamp:
        x = x[0::2]
        y = y[0::2]
        unw = unw[0::2,0::2]
        diff = diff[0::2,0::2]
        
    # The Afghanistan earthquake has a low signal-to-noise ratio, so the reference point 
    # noticeably impacts the residual rms. Shifting it so that the noise is roughly zero centred.
    if example_name == 'afghanistan':
        unw = unw - np.nanmean(unw)
    
    return x, y, unw, diff

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

#-------------------------------------------------------------------------------

def rms_misfit(a,b):
    '''
    Calculate the root-mean-square misfit between 'a' and 'b'.
    
    INPUTS
        a,b = two arrays of same length
    OUTPUTS
        rms = rms misfit between a and b (a-b)
    '''
    
    rms = np.sqrt(np.nanmean((a-b)**2))
    
    return rms

#-------------------------------------------------------------------------------

def print_results():
    '''
    Print desired model parameter values for the three examples
    '''
    
    output = """
    Well done for completing the practical.
    Printed below are 'good fit' model parameters for the three examples, based on the provided source
    and with a bit of adjusting from John Elliott.
    
    ========================================
    Greece
    ----------------------------------------
              xcen = 150
              ycen = -150
            strike = 315
               dip = 36
              rake = -100
              slip = 1.15
    centroid depth = 4500
             width = 9400
            length = 9900
            
    Source: https://pubs.geoscienceworld.org/ssa/srl/article/93/5/2584/614110/Coseismic-Surface-Deformation-Fault-Modeling-and
    
    ========================================
    Afghanistan
    ----------------------------------------
              xcen = 1000
              ycen = 1000
            strike = 210
               dip = 78
              rake = 15
              slip = 1.4
    centroid depth = 2800
             width = 4000
            length = 6000
            
    Source: https://earthquake.usgs.gov/earthquakes/eventpage/us7000hj3u/executive
            
    ========================================
    Iran
    ----------------------------------------
              xcen = -20000
              ycen = -10000
            strike = 353.7
               dip = 136.8
              rake = 3.05
              slip = 3.05
    centroid depth = 14800
             width = 21200
            length = 40100
            
    Source: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JB016221
    
    """
    
    print(output)
    
    
