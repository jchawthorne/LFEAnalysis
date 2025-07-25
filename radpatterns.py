"""
This file contains some functions to
calculate the coefficients of the P, SH, and SV waves
given a fault plane and slip direction along with
an azimuth and takeoff angle.

The equations for these calculations are documented in a variety of
seismology textbooks and papers, including

Madariaga, R. (1989). Seismic source: Theory . In: Geophysics. Encyclopedia of Earth Science. Springer, Boston, MA. https://doi.org/10.1007/0-387-30752-4_137.

Ou, G. B., 2008: Seismological studies for tensile faults. Terr. Atmos. Ocean. Sci., 19, 463-471, doi: 10.3319/TAO.2008.19.5.463(T).



Two functions are normally used directly.

create_moment_tensor :
    to create a moment tensor with a specified strike, dip, rake, and opening
    by default, the moment tensor is in an E-N-up coordinate system

calc_radiation_pattern :
    calculates the radiation coefficients for the P, SH, and SV waves
    given a moment tensor, a takeoff angle, and an azimuth




These two functions call two helper functions, which are also accessible.

fault_projected_moment_tensor :
    creates a moment tensor that is either shear or opening,
    with axes aligned along the fault plane

calc_unit_vectors :
    calculates vectors for the spherical coordinate system used in calculation
    these vectors are specified in the E-N-up coordinate system and
    are also the directions of positive particle motion for the P, SH, and SV waves

"""


import numpy as np


def create_moment_tensor(strike=0,dip=20,rake=90,opening_angle=0.,
                         coordinates='xyz',poissons_ratio=0.25):
    """
    Parameters
    ----------
    strike :
        strike of the fault
    dip :
        dip of the fault
    rake :
        rake of shear slip on the fault
          0: left-lateral, 90: thrust, 180: right-lateral, -90: normal
    opening_angle :
        direction of slip relative to the fault
          0: simple shear (default), 90: opening, -90: closing
    coordinates :
        whether you want the moment tensor in the xyz (aka ENZ) system or
          in an 'fault' system, with [updip,along strike,up fault-perpendicular] axes
    poissons_ratio :
        Poisson's ratio (default: 0.25)

    Returns
    -------
    M :
        a moment tensor 
    """

    # start with slip in updip-along_strike-fault_perpendicular/up reference frame
    Mshear=fault_projected_moment_tensor(slip_type='shear',rake=rake)
    Mopen=fault_projected_moment_tensor(slip_type='opening')

    fraction_shear=np.cos(opening_angle*np.pi/180)
    fraction_opening=np.sin(opening_angle*np.pi/180)
    
    # sum them
    M=fraction_shear*Mshear+fraction_opening*Mopen


    # rotate to an ENZ grid
    if coordinates in ['xyz','XYZ','ENZ','enz']:
        dip,strike=dip*np.pi/180,strike*np.pi/180

        # first project to move the dip to the right orientation
        # started with [downdip,along-strike,fault-perp up]
        # move to [horizontal downdip,along-strike, up]
        R=np.zeros([3,3],dtype=float)
        R[1,1]=1.
        R[0,0]=np.cos(dip)
        R[2,0]=-np.sin(dip)
        R[0,2]=np.sin(dip)
        R[2,2]=np.cos(dip)

        #print('R for dip\n',np.round(R,1))
        
        M=np.tensordot(R,np.tensordot(M,R.T,axes=[1,0]),axes=[1,0])

        # now the vertical is set
        # rotate along-strike to the desired azimuth, downdip horizontal to 90% more clockwise
        R=np.zeros([3,3],dtype=float)
        R[2,2]=1.
        R[0,0]=np.cos(strike)
        R[0,1]=np.sin(strike)
        R[1,1]=np.cos(strike)
        R[1,0]=-np.sin(strike)

        #print('R for strike',np.round(R,1))

        M=np.tensordot(R,np.tensordot(M,R.T,axes=[1,0]),axes=[1,0])
    
    return M
    



def fault_projected_moment_tensor(slip_type='shear',rake=0,poissons_ratio=0.25):
    """
    Parameters
    ----------
    slip_type :
        which sort of slip you want
         'shear' (default), 'opening', or 'explosion'
    rake :
        rake of shear slip on the fault
          0: left-lateral, 90: thrust, 180: right-lateral, -90: normal
    poissons_ratio :
        Poisson's ratio (default: 0.25)


    Returns
    -------
    M :
        the moment tensor as proejcted along the fault
            x1=down-dip, x2=along strike x3=fault perpendicular, up
    """

    # ratio of Lame parameter to shear modulus
    lambda2shmod=2*poissons_ratio/(1-2*poissons_ratio)

    # initialise as zeros
    M=np.zeros([3,3],dtype=float)

    if slip_type in ['shear']:
        # slip in along-strike direction
        slip_strike=np.cos(rake*np.pi/180)
        M[1,2]=slip_strike
        M[2,1]=slip_strike
        
        # slip in down-dip direction
        slip_downdip=-np.sin(rake*np.pi/180)
        M[0,2]=slip_downdip
        M[2,0]=slip_downdip
        
    elif slip_type in ['explosion']:
        M[0,0]=(2./3)**0.5
        M[1,1]=(2./3)**0.5
        M[2,2]=(2./3)**0.5
        
    elif slip_type in ['opening']:
        M[0,0]=lambda2shmod
        M[1,1]=lambda2shmod
        M[2,2]=lambda2shmod+2.
        
    return M


def calc_unit_vectors(takeoff_angle,azimuth):
    """
    calculate the unit vectors relative to the *wave travelling from the source#

    
    Parameters
    ----------
    takeoff_angle :
       the takeoff angle, the angle between the downward direction and the takeoff direction
          in degrees
    azimuth :
       the azimuth of the seismic wave,
          measured in degrees clockwise from north

    Returns
    -------
    R_vec :
       unit vector in the takeoff direction
    theta_vec :
       unit vector in the theta direction
    phi_vec :
       unit vector in the phi direction
    """

    # convert to radians
    theta=takeoff_angle*np.pi/180.
    phi=azimuth*np.pi/180

    # in the R directions
    R_vec=np.array([np.sin(theta)*np.sin(phi),
                    np.sin(theta)*np.cos(phi),
                    -np.cos(theta)])

    # in the theta direction
    theta_vec=np.array([np.cos(theta)*np.cos(phi),
                        np.cos(theta)*np.sin(phi),
                        -np.sin(theta)])
    theta_vec=np.array([np.cos(theta)*np.sin(phi),
                        np.cos(theta)*np.cos(phi),
                        np.sin(theta)])

    # in the phi direction
    phi_vec=np.array([-np.sin(phi),
                      np.cos(phi),
                      0.])
    phi_vec=np.array([np.cos(phi),
                      -np.sin(phi),
                      0.])

    return R_vec,theta_vec,phi_vec




def calc_radiation_pattern(M,takeoff_angle,azimuth):
    """
    Parameters
    ----------
    M :
        the moment tensor, chosen such that the axes
            X=E, x2=N, x3=up
    takeoff_angle :
       the takeoff angle, the angle between the downward direction and the takeoff direction
          in degrees
    azimuth :
       the azimuth of the seismic wave,
          measured in degrees clockwise from north
    
    Returns
    -------
    rd_p : 
        p wave radiation coefficient
          positive is in the direction of travel, away from the earthquake
    rd_sh : 
        SH wave radiation coefficient
          positive is to the right when facing the direction of travel
    rd_sv : 
        SV wave radiation coefficient
          positive is raypath perpendicular, partly up, when facing the direction of travel
    """
    
    # calculate the unit vectors
    R_vec,theta_vec,phi_vec=calc_unit_vectors(takeoff_angle,azimuth)

    # SH
    rd_sh=np.dot(phi_vec,np.dot(M,R_vec))
    
    # SV
    rd_sv=np.dot(theta_vec,np.dot(M,R_vec))
    
    # P
    rd_p=np.dot(R_vec,np.dot(M,R_vec))
        
    return rd_p,rd_sh,rd_sv
