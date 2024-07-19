# some codes to read particular LFE catalogues
import numpy as np
import os,glob
import datetime

#-----BEGIN FRANK 2014 GUERRERO CATALOGUE----------------

def read_frank2014(data_directory=None):
    """
    Parameters
    ----------
    data_directory : 
          the overall data directory, containing the folder 'Catalogues'

    Returns: 
    tms: 
          LFE times
    loc: 
          LFE locations (lon,lat,depth)
    dct: 
          dictionary with additional values
    """

    # read Farge et al LFE catalog from Guerrero
    fname = os.path.join(data_directory,'Catalogues',
                         'Frank2014','frank_jgr_2014_lfe_catalog.txt')

    # read headers
    hdr=['Year','Month','Day','Hour','Minute','Second',
         'Template_ID','Latitude','Longitude','Depth']

    # read values
    vls=np.loadtxt(fname,skiprows=1,dtype=float)

    # make a dictionary
    dct={}
    for k in range(0,len(hdr)):
        hdri=hdr[k]
        if hdri in ['Year','Month','Day','Hour','Minute']:
            dct[hdri]=vls[:,k].astype(int)
        else:
            dct[hdri]=vls[:,k].astype(float)

    # times
    vls=vls[:,0:5].astype(int)
    tms=[datetime.datetime(vls[k,0],vls[k,1],vls[k,2],vls[k,3],vls[k,4])+\
         datetime.timedelta(seconds=dct['Second'][k])
         for k in range(0,vls.shape[0])]
    tms=np.array(tms)

    # locations
    loc=np.vstack([dct['Longitude'],dct['Latitude'],-dct['Depth']]).T

    return tms,loc,dct

#-----END FRANK 2014 GUERRERO CATALOGUE------------------

#----BEGIN BOSTOCK ET AL CASCADIA CATALOGUE--------------------------

def read_bostock_cascadia(trange=None,data_directory=''):
    """
    read LFE info for Michael's Cascadia catalogue
    
    Parameters
    ----------
    trange: 
          time range to consider
    data_directory : 
          the overall data directory, containing the folder 'Catalogues'

    Returns
    -------
    tms: 
           LFE times
    loc:  
           locations
    mags:  
           magnitudes
    iev: 
           family number
    """



    # LFE locations
    datdir=os.path.join(data_directory,'Catalogues','Bostock')
    evn,snm,loc = readTRbostockloc(data_directory=datdir)

    # LFE times
    iev,mags,nm,tms = readTRbostock(data_directory=datdir)
    # change mapping
    ievh = np.array(ismember(iev,snm))

    # subset
    ix = ievh!=0
    iev,mags,nm,tms,ievh=iev[ix],mags[ix],nm[ix],tms[ix],ievh

    # only some time range?
    if trange is not None:
        ix=np.logical_and(tms>=trange[0],tms<=trange[1])
        iev,mags,nm,tms=iev[ix],mags[ix],nm[ix],tms[ix]

    # extract per-event location
    loc=loc[:,ievh].T

    return tms,loc,mags,iev


def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, 0) for itm in a]


def readTRbostock(data_directory=''):
    """
    Parameters
    ----------
    data_directory : 
          the data directory with the actual files

    Returns
    -------
    iev: 
           family number
    mags:  
           magnitudes
    nm:  
           ??
    tms: 
           LFE times
    """
    
    fdir = os.path.join(data_directory)
    fname = 'total_mag_detect_0000_cull.txt'
    fl = open(os.path.join(fdir,fname),'r')

    # event index
    iev = [];

    # times
    tms = []

    # magnitudes
    mags = [];

    # some number
    nm = []

    for line in fl:
        # read
        vl = line.split()

        yr=int(vl[1][0:2])
        if yr < 90:
            yr = yr + 2000
        else:
            yr = yr + 1900

        sc = vl[3].split('.')
        ms = int(sc[1])
        sc = int(sc[0])
        hr=int(vl[2]) - 1

        dt=datetime.datetime(year=yr,month=int(vl[1][2:4]),
                             day=int(vl[1][4:6]),hour=hr)
        dt=dt+datetime.timedelta(minutes=0,seconds=sc,milliseconds=ms)

        # add to list
        nm.append(int(vl[5]))
        mags.append(float(vl[4]))
        tms.append(dt)
        iev.append(int(vl[0]))

    tms = np.array(tms)
    mags = np.array(mags)
    iev = np.array(iev)
    nm = np.array(nm)

    fl.close()

    return iev,mags,nm,tms


def readTRbostockloc(data_directory=''):
    """
    Parameters
    ----------
    data_directory : 
          the data directory with the actual files

    Returns
    -------
    evn: 
           LFE numbers
    svn:  
           ??
    loc:  
           locations
    """

    fdir = os.path.join(data_directory)
    fname = 'svi_sta.dat'
    fname = os.path.join(fdir,fname)
    vl = np.genfromtxt(fname,dtype=None,encoding=None)

    # event numbers
    evn=np.array([x[0] for x in vl])
    snm=np.array([int(x[1][1:]) for x in vl])
    lat=np.array([x[2] for x in vl])
    lon=np.array([x[3] for x in vl])
    dep=np.array([x[4] for x in vl])
    
    evn=np.array([x[0] for x in vl])
    loc=np.array([lon,lat,dep])

    return evn,snm,loc 

def read_bostock_template(fnum,data_directory=''):
    """
    read the template waveforms

    Parameters
    ----------
    fnum :
        the family number to read
    data_directory : 
          the overall data directory, containing the folder 'Catalogues'
    
    Returns
    -------
    st : 
        the templates as obspy streams
    """

    from scipy.io import loadmat
    import obspy
    
    # file of interest
    fdir=os.path.join(data_directory,'Catalogues','Bostock','WFPVM')
    fname='wfpvm_{:3.0f}.mat'.format(fnum).replace(' ','0')
    fname=os.path.join(fdir,fname)

    # read
    data=loadmat(fname)

    # create an obspy stream and add to it
    st=obspy.Stream()

    # and a trace to modify
    tr=obspy.Trace()
    tr.data=np.random.rand(300)
    tr.stats.delta=float(data['ndt'])

    for k in range(0,len(data['ordlst'])):
        # set station
        tr.stats.station=data['ordlst'][k]

        # get E component
        tr.data=data['ecomp'][k,:]
        tr.stats.channel='E'
        st.append(tr.copy())

        # get N component
        tr.data=data['ncomp'][k,:]
        tr.stats.channel='N'
        st.append(tr.copy())

        # get vertical component
        tr.data=data['zcomp'][k,:]
        tr.stats.channel='Z'
        st.append(tr.copy())

    return st




#----END BOSTOCK ET AL CASCADIA CATALOGUE--------------------------
