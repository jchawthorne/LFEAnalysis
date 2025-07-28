import numpy as np
import obspy
import io,sys
import scipy
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import gridspec

def readsac_portion(fname,starttime=None,endtime=None,byteorder=None):
    """
    :param          fname: file name
    :param      starttime: start time
    :param        endtime: end time
    :param      byteorder: byte order, if needed
    """

    # mostly copied from obspy's read_sac function

    if starttime is None and endtime is None:
        st=obspy.read(fname)
    else:
        # the headonly already ignores the rest, so just use that
        st=obspy.read(fname,headonly=True)
        tr=st[0]
        
        if endtime is None:
            endtime=tr.stats.endtime
        if starttime is None:
            starttime=tr.stats.starttime
        
        if tr.stats.starttime>endtime or tr.stats.endtime<starttime:
            # if there's no time overlap, stop now
            st=obspy.Stream()

        else:
            # number of points in the seismogram
            npts=tr.stats.npts
            
            # indices to start and end
            startn=int((starttime-tr.stats.starttime)/tr.stats.delta)
            endn=int((endtime-tr.stats.starttime)/tr.stats.delta)
            startn=np.maximum(startn,0)
            endn=np.minimum(endn,npts-1)
            nread=endn-startn+1
            
            # open file
            try:
                f = open(fname, 'rb')
                is_file_name = True
            except TypeError:
                # source is already a file-like object
                f = fname
                is_file_name = False
            
            # deal with byte order
            is_byteorder_specified = byteorder is not None
            if not is_byteorder_specified:
                byteorder = sys.byteorder
            if byteorder == 'little':
                endian_str = '<'
            elif byteorder == 'big':
                endian_str = '>'
            else:
                raise ValueError("Unrecognized byteorder. Use {'little', 'big'}")
        
            from obspy.core.compatibility import from_buffer
            
            # seek to end of header, plus offset
            f.seek(632+4*startn,0)
            
            # reset start time
            tr.stats.starttime=tr.stats.starttime+startn*tr.stats.delta
            
            # read data
            tr.data = from_buffer(f.read(int(nread)*4),
                                  dtype=endian_str+'f4')
            
            # close file
            if is_file_name:
                f.close()
        
    return st
        

        
def copypicks(st1,st2,pks):
    """
    :param     st1:  first set of waveforms
    :param     st2:  second set of waveforms
    :param     pks:  picks to copy
    """

    # set up as list and streams
    if isinstance(pks,str):
        pks = [pks]
    if isinstance(st1,obspy.Trace):
        st1=obspy.Stream(st1)
    if isinstance(st2,obspy.Trace):
        st2=obspy.Stream(st2)

    for tr1 in st1:
        st2i = st1.select(id=tr1.get_id())
        for tr2 in st2i:
            for pk in pks:
                # pick set to the same absolute time
                try:
                    tr2.stats[pk]=tr1.stats.starttime-tr2.stats.starttime+\
                        tr1.stats[pk]
                except:
                    pass


def plotstations(st,inkm=True):
    """
    simple lon-lat station plot from files with sac stlo and stla set
    """
    

    lon=dict((tr.stats.station,tr.stats.sac.stlo) for tr in st)
    lat=dict((tr.stats.station,tr.stats.sac.stla) for tr in st)

    if inkm:
        lonref = np.median(lon.values())
        latref = np.median(lat.values())
        rt = np.cos(latref*np.pi/180.)

        for ky in lon.keys():
            lon[ky]=(lon[ky]-lonref)*rt*111.
        for ky in lat.keys():
            lat[ky]=(lat[ky]-latref)*111.


    f = plt.figure()
    p = plt.axes()

    for stn in lon.keys():
        p.plot(lon[stn],lat[stn],linestyle='none',marker='^',color='k')
        p.text(lon[stn],lat[stn],stn)

    if inkm:
        p.set_xlabel('distance E of '+str(round(lonref,2))+' (km)')
        p.set_ylabel('distance N of '+str(round(latref,2))+' (km)')

def delsame(st,nrow=5):
    """
    :param       st:  waveforms or trace
    :param     nrow:  number of values the same in a row
    """

    if isinstance(st,obspy.Trace):
        # the initial mask
        if isinstance(st.data,np.ma.masked_array):
            st.data=np.ma.masked_array(st.data,mask=False)

        # differences
        df=np.diff(st.data)==0.
        #dfi=np.logical_and(st.data.mask[:-1],st.data.mask[1:])
        #df=np.logical_or(df,dfi)
        sm=np.cumsum(df)
        sm[~df]=0

    elif isinstance(st,obspy.Stream):
        for tr in st:
            delsame(tr,nrow=nrow)
    

def addfiltmask(st,msk):
    """
    :param        st: waveforms or trace
    :param       msk: a set of waveforms with masks
    """
    
    if isinstance(st,obspy.Trace):
        # the mask
        #ms = msk.data.astype(bool)
        ms = msk.data > 0.1
        try:
            # if there's already a mask, combine them
            st.data.mask=np.logical_or(st.data.mask,ms)
        except:
            st.data=np.ma.masked_array(st.data,mask=ms)

    elif isinstance(st,obspy.Stream):

        for tr in st:
            # select mask
            ms = msk.select(id=tr.id)[0]

            # add filter
            addfiltmask(tr,ms)

    

def prepfiltmask(st,tmask=3.):
    """
    :param        st: waveforms or trace
    :param     tmask: time window to mask within some interval or endpoint, 
                         in seconds
    :return      msk: a set of waveforms with masks
    """

    if isinstance(st,obspy.Trace):

        # allowable values
        try:
            ms=np.logical_or(st.data.mask,np.isnan(st.data.data))
            data=st.data.data
        except:
            ms = np.isnan(st.data)
            data=st.data

        # times
        tm=np.arange(0.,data.size)

        # interpolate
        if np.sum(~ms):
            data[ms]=np.interp(tm[ms],tm[~ms],data[~ms])
        else:
            data[:]=0.
        
        # and copy data
        st.data = data    

        # to mask
        nwin = int(tmask/st.stats.delta)
        nwin = np.maximum(nwin,1)
        #win = scipy.signal.boxcar(nwin*2+1)
        win = np.ones(nwin*2+1,dtype=float)
        ms = ms.astype(float)
        ms = scipy.signal.convolve(ms,win,mode='same')

        # also the beginning and end
        if tmask != 0.:
            ms[0:nwin+1]=1.
            ms[-nwin:]=1.

        # place in trace
        ms = np.minimum(ms,1.)
        msk = st.copy()
        msk.data = ms

    elif isinstance(st,obspy.Stream):
        msk = obspy.Stream()

        for tr in st:
            mski = prepfiltmask(tr,tmask=tmask)
            msk.append(mski)

    return msk


def bufferwaveforms(st,tbf=0.,taf=0.):
    """
    add time on either side of a waveform
    :param    st: waveforms to use
    :param   tbf: time beforehand in seconds (default: 0)
    :param   taf: time after in seconds (default: 0)
    :return   st: original waveforms, modified in place
    """

    if isinstance(st,obspy.Stream):
        # for all traces
        for tr in st:
            tr = bufferwaveforms(tr,tbf=tbf,taf=taf)
    elif isinstance(st,obspy.Trace):
        tst = st.stats.starttime-tbf
        tnd = st.stats.endtime+taf
        st = trimandshift(st,starttime=tst,endtime=tnd,
                          pad=True,fill_value=0.)

    return st

def trimandshift(st,*args,**kwargs):
    """
    calls obspy.trim, but also shifts the times to match
    :param    st: waveforms
    :param     ?: remaining values go directly to trim
    :return   st: waveforms, modified in place
    """

    if isinstance(st,obspy.Stream):
        # for all traces
        for tr in st:
            tr = trimandshift(tr,*args,**kwargs)
    elif isinstance(st,obspy.Trace):
        # relative to start time
        tst = st.stats.starttime
        
        # trim
        st = st.trim(*args,**kwargs)

        # time shift to move t0, etc
        tst = tst - st.stats.starttime

        # to check
        for k in range(0,50):
            tch = 't'+str(k)
            if tch in st.stats:
                st.stats[tch] = float(st.stats[tch]) + tst


    return st
    
    
    
    
#----------MANIPULATING WAVEFORM HEADERS------------------
    

def copytosacheader(tr):
    """
    :param      tr:  trace or waveform
    """
    
    if isinstance(tr,obspy.Trace):
        # start time and year
        starttime = tr.stats.starttime
        tr.stats.sac['b']=0.
        tr.stats.sac['e']=(tr.stats.npts-1)*tr.stats.delta
        tr.stats.sac['nzyear']=starttime.year
        tr.stats.sac['nzjday']=starttime.julday
        tr.stats.sac['nzhour']=starttime.hour
        tr.stats.sac['nzmin']=starttime.minute
        tr.stats.sac['nzsec']=starttime.second
        tr.stats.sac['nzmsec']=int(round(starttime.microsecond/1.e3))

        # also the station info
        tr.stats.sac['knetwk']=tr.stats.network
        tr.stats.sac['kstnm']=tr.stats.station
        tr.stats.sac['kcmpnm']=tr.stats.channel

        # and the timing
        tr.stats.sac['delta']=tr.stats.delta

        # and the time picks
        for k in range(0,10):
            ky = 't'+str(k)
            if ky in tr.stats.keys():
                tr.stats.sac['kt'+str(k)]=tr.stats[ky]

    elif isinstance(tr,obspy.Stream):
        # go through each trace
        for tri in tr:
            copytosacheader(tri)



def copyfromsacheader(tr):
    """
    :param      tr:  trace or waveform
    """
    
    if isinstance(tr,obspy.Trace):
        # just the time picks
        for k in range(0,10):
            ky = 't'+str(k)
            kys = 'kt'+str(k)
            if kys in tr.stats.sac.keys():
                tr.stats[ky]=float(tr.stats.sac[kys])

        # start time
        starttime=obspy.UTCDateTime(year=tr.stats.sac['nzyear'],
                                    julday=tr.stats.sac['nzjday'],
                                    hour=tr.stats.sac['nzhour'],
                                    minute=tr.stats.sac['nzmin'])
        starttime=starttime+tr.stats.sac['nzsec']+\
            tr.stats.sac['nzmsec']*1.e-3
        starttime=starttime+tr.stats.sac['b']
        tr.stats.starttime=starttime

    elif isinstance(tr,obspy.Stream):
        # go through each trace
        for tri in tr:
            copyfromsacheader(tri)


def sactokml(st):
    """
    :param    st:  a set of waveforms
    :return  kml:  setup for a kml file
    write to file with kml.save('filename')
    """

    import simplekml

    # kml file
    kml = simplekml.Kml()

    # lon, lat
    lon=dict((tr.stats.network+'.'+tr.stats.station,
              tr.stats.sac.stlo) for tr in st)
    lat=dict((tr.stats.network+'.'+tr.stats.station,
              tr.stats.sac.stla) for tr in st)

    # names
    nms=np.array([tr.stats.network+'.'+tr.stats.station 
                  for tr in st])
    nms=np.unique(nms)

    # add points to kml file
    for nm in nms:
        kml.newpoint(name=nm,coords=[(lon[nm],lat[nm])])


    return kml




def plottogpicks(st,pk='t0',tlm=None,tscl=None,rscl=True,fname=None):
    # INPUT
    # 
    # st          a list of waveform sets
    # rscl        rescale the amplitudes

    if isinstance(st,obspy.Stream):
        st=[st]

    if tlm is None:
        tlm = np.array([-1.,3])

    if tscl is None:
        tscl = tlm

    # set of stations and components
    lbls,lblss=[],[]
    for sti in st:
        for tr in sti:
            wid=tr.stats.network+'.'+tr.stats.station+'.'+\
                tr.stats.channel
            lbls.append(wid.strip())
            #wid=tr.stats.station
            lblss.append(wid.strip())

    lbls,ix=np.unique(np.array(lbls),return_index=True)
    lblss=np.array(lblss)[ix]
    
    # number of stations
    N = len(lbls)

    # initialize plots
    plt.close()
    f = plt.figure(figsize=(8,12))
    gs,p=gridspec.GridSpec(N,1),[]
    for k in range(0,N):
        p.append(plt.subplot(gs[k]))

    cls = ['blue','red','green','black','yellow','orange']
    M = len(st)

    for k in range(0,N):
        for m in range(0,M):
            nw,stn,chn=lbls[k].split('.')
            tr = st[m].select(network=nw,station=stn,channel=chn)
            
            #wid=tr.stats.network+'.'+tr.stats.station+'.'+\
            #    tr.stats.channel

            # find the relevant values
            #            tr = st[m].select(id=lbls[k])

            if tr:
                tr = tr[0]
                # the reference time
                tsec,tpk=resolvepick(tr,pk=pk,mkset=None)

                # for scaling
                if rscl:
                    scl = tr.copy().trim(starttime=tpk+tscl[0],
                                         endtime=tpk+tscl[1])
                    if scl.data.any():
                        scl = np.max(np.abs(scl.data))
                    else:
                        scl = float('nan')
                else:
                    scl = 1.

                # for plotting
                data = tr.copy().trim(starttime=tpk+tlm[0],
                                      endtime=tpk+tlm[1])
                tm = data.times()+(data.stats.starttime-tpk)
                data = data.data / scl

                # plot
                p[k].plot(tm,data,color=cls[m])

        p[k].set_ylabel(lblss[k])
        p[k].set_xlim(tlm)
        if rscl:
            p[k].set_ylim(np.array([-1.,1.])*1.1)
        if k<N-1:
            p[k].set_xticklabels([])

    if fname:
        graphical.printfigure(fname,f)
    else:
        plt.show()

    return p


def resolvepick(tr,pk='t0',mkset=None):
    # INPUT
    #
    # tr          trace
    # pk          reference time for tr
    #                either a string for a marker, a time in seconds from the beginning,
    #                or a time
    # mkset       string of a marker to set (default: None, not set)
    #
    # OUTPUT
    #  
    # tsec        pick time in seconds relative to the start time
    # tdat        pick time as a date

    if isinstance(pk,str):
        # time from marker
        tsec = tr.stats[pk]
        tdat = tr.stats.starttime+tsec
    elif isinstance(pk,float):
        # if it's a time from the beginning
        tsec = pk
        tdat = tr.stats.starttime+tsec
    elif isinstance(pk,datetime.datetime):
        # if it's just a time
        tdat = obspy.UTCDateTime(pk)
        tsec = tdat - tr.stats.starttime
    elif isinstance(pk,obspy.UTCDateTime):
        # if it's just a time
        tdat = pk
        tsec = tdat - tr.stats.starttime

    if mkset:
        # set marker
        tr.stats[mkset]=tsec

    return tsec,tdat

