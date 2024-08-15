import numpy as np
import obspy
import os,glob
import matplotlib.pyplot as plt
import seisproc
import pickle
from matplotlib import gridspec
from mpl_toolkits.basemap import Basemap
from scipy import signal
from scipy import interpolate
import read_catalogues

class lfanalyse:
    # class for LFE analysis, for one LFE family

    #-----BEGIN INITIATION INFO------------------------

    def __init__(self,fnum=74,lfi=None,region='Cascadia',catalog=None,
                 data_directory=None):
        """
        Parameters
        ----------
        fnum :
             family number
        lfi : 
             a previous object, to copy all info from
        region : 
             the region we plan to work with (default: 'Cascadia')
        catalog :
             which LFE catalogue to use
        data_directory : 
             the computer directory that will contain the data
                 (default: $DATA or the current working directory)
        """

        # set the data directory
        self.set_data_directory()

        # just note family number
        self.fnum=int(fnum)

        # and select some parameters
        # window length
        self.tlen=80

        # window for scaling computation
        self.wlen=np.array([0.,4])

        # some groups of events
        self.groups={}

        # some reference distances
        self.ref_distances=np.array([],dtype=float)

        # time splits
        self.tsplits=np.array([0.,0.75,3])

        # region and LFE catalogue
        self.region=region
        self.catalog=catalog
        
        # copy information from another object
        if lfi is not None:
            for ky in lfi.__dict__.keys():
                self.__setattr__(ky,lfi.__getattribute__(ky))

        # initialize anything that's region-dependent
        self.initiate_region()

        
    def initiate_region(self,region=None):
        """
        intialize functions that could vary between regions

        Parameters
        ----------
        region :
              the region, if it is to be changed 
               (default: None, use self.region)
        """

        # choose a slip direction and fault dip
        self.slipdir=54+180
        self.faultdip=20.

        if region is not None:
            self.region = region

        if self.region in ['Cascadia']:
            # choose a slip direction and fault dip
            self.slipdir=54+180
            self.faultdip=20.

            if self.catalog is None:
                self.catalog='Bostock'
            
            # note how to read the detections
            if self.catalog in ['Bostock']:
                self.read_detection_info=self.read_detection_info_bostock

            # note rough areas for data and plotting
            self.latlim=np.array([46,53])
            self.lonlim=np.array([-132,-117])

            # sampling rate
            self.sampling_rate=40.

        elif self.region in ['Guerrero']:
            if self.catalog is None:
                self.catalog='Frank2014'
                
            # note how to read the detections
            if self.catalog in ['Frank2014']:
                self.read_detection_info=self.read_detection_info_frank2014

            # note rough areas for data and plotting
            self.latlim=np.array([15,24])
            self.lonlim=np.array([-101,-97.6])

            # sampling rate
            self.sampling_rate=100.

        elif self.region in ['Parkfield']:
            if self.catalog is None:
                self.catalog='Shelly2017'

            
            

            
    #-----END INITIATION INFO------------------------


    #-----BEGIN SUMMARY PROCESSING SCRIPTS---------------

    def load_prep_data(self,flm=[1,5],single_norm=False):
        """
        Parameters 
        ----------
        flm : 
            frequency band to filter to
        single_norm :
            just use one normalization per station, not per interval
             (default: False)
        """
        
        # read detection info and templates
        self.read_detection_info()
        # may need to delete this line
        self.pick_templates()

        # read, normalize, and filter the data
        print('Reading data')
        self.read_data()
        print('Normalizing data')
        self.normalize_data(single_norm=single_norm)
        print('Filtering data')
        self.filter_data(flm=flm)

        # compute LFE rates
        self.identify_sse_times()
        self.compute_lfe_rates(bin_duration=5*3600)
        
        # bin the LFEs into groups by time in the SSE
        self.select_start_times()
        self.group_events_bytime()


    #-----END SUMMARY PROCESSING SCRIPTS----------------

    #-----BEGIN READ TEMPLATE AND DETECTION INFO---------

    
    def read_detection_info_frank2014(self):
        """
        read Frank et al (2014)'s detection times and templates
        """

        # read some LFE times from William Frank's catalog
        tms,loc,dct=read_catalogues.read_frank2014(data_directory=self.data_directory)
        iev=dct['Template_ID'].astype(int)

        # grab some info about the families
        fnums,ix=np.unique(iev,return_index=True)
        flocs=loc[ix,:]

        # select a family 
        ii=np.where(fnums==self.fnum)[0]
        fnum=fnums[ii]
        self.floc=flocs[ii,:].flatten()

        # keep a copy of all the detections
        self.atms=np.array([obspy.UTCDateTime(tm) for tm in tms])
        
        # the times of interest
        ii=iev==fnum
        self.tms=self.atms[ii]

        # the frank et al catalog doesn't have magnitudes, so
        # let's set the magnitudes to nan for now
        self.mags=np.ndarray(self.tms.size,dtype=float)*float('nan')

        
        
    def read_detection_info_bostock(self):
        """
        read Bostock et al's detection times and templates
        """

        # read some LFE times for Michael Bostock's catalog
        tms,loc,mags,iev=\
            read_catalogues.read_bostock_cascadia(data_directory=self.data_directory)

        # grab some info about the families
        fnums,ix=np.unique(iev,return_index=True)
        flocs=loc[ix,:]

        # select a family 
        ii=np.where(fnums==self.fnum)[0]
        fnum=fnums[ii]
        self.floc=flocs[ii,:].flatten()

        # keep a copy of all the detections
        self.atms=np.array([obspy.UTCDateTime(tm) for tm in tms])
        
        # the times of interest
        ii=iev==fnum
        self.tms=self.atms[ii]
        self.mags=mags[ii]

        # read the templates for this family
        self.sttemp=read_catalogues.read_bostock_template(self.fnum,
                         data_directory=self.data_directory)
        for tr in self.sttemp:
            tr.stats.station=tr.stats.station.strip()
            tr.stats.channel=tr.stats.channel.strip()
    


    #-----END READ TEMPLATE AND DETECTION INFO---------

    #-----BEGIN EVENT GROUPING AND LFE RATE ANALYSIS----

    def select_start_times(self,start_rate=0.5):
        """
        select the start time for each event as the first time that the rate reaches half the maximum

        Parameters
        ----------
        start_rate :
            rate to start event, as fraction of the maximum
        """

        # note max rates
        self.max_rates=np.ndarray(self.Nsse,dtype=float)*float('nan')
        self.start_times=[]
    
        for iev in range(0,self.Nsse):
            # times for this event
            ii=np.where(np.logical_and(self.tcent>=self.sse_tlms[iev,0],
                                       self.tcent<=self.sse_tlms[iev,1]))[0]
            rt=self.lferate[ii]

            # max rate
            self.max_rates[iev]=np.max(rt)
            ix=np.where(rt>=start_rate*self.max_rates[iev])[0]

            # find the start times
            if ix.size:
                self.start_times.append(self.tcent[ii[ix[0]]])

        # move start time to beginning of bin
        self.start_times=np.array(self.start_times)-self.bin_duration/2.
            
    def identify_sse_times(self,n_events=10):
        """
        Identify the time windows with large slow slip events.
        Just divides the events by the largest time gaps between LFEs.

        Parameters
        ----------
        n_events : 
             number of events anticipated
        """

        # sorted set of times
        iatms=self.atms-self.atms[0]
        ix=np.argsort(iatms)
        atms=self.atms[ix]
        
        # time between LFEs
        dtim=np.diff(atms)

        # find the biggest time gaps
        ix=np.argsort(dtim)[-(n_events-1):]
        ix.sort()

        # start and stop times
        tstart=np.append([0],ix+1)
        tstop=np.append([ix],atms.size-1)
        tstart=atms[tstart]
        tstop=atms[tstop]

        # just the dates rather than the times, buffered
        tbuffer=2*3600
        tstart=np.array([obspy.UTCDateTime((tm-tbuffer).date)
                         for tm in tstart])
        tstop=np.array([obspy.UTCDateTime((tm+tbuffer+86400).date)
                         for tm in tstop])

        # combine and save
        self.sse_tlms=np.vstack([tstart,tstop]).T

        # note the number of SSEs
        self.Nsse=n_events

    def compute_lfe_rates(self,bin_duration=7200,bin_spacing=None):
        """
        compute the LFE rate and median magnitude in various time windows
        --saves bin centre times as self.tcent
        --saves LFE rate in detections / hour as self.lferate

        Parameters
        ----------
        bin_duration :
            time window for binning, in seconds (default: 7200)
        bin_spacing :
            lag between window starts, in seconds (default: bin_duration/4)
            usually chosen to be a fraction of bin_duration, so that
            there's significant overlap
        """

        # bin spacing
        if bin_spacing is None:
            bin_spacing=bin_duration/4.
        
        # determine center times
        tcent=np.ndarray(0)
        for k in range(0,self.Nsse):
            tcenti=np.arange(self.sse_tlms[k,0]+bin_duration/2,
                             self.sse_tlms[k,1]-bin_duration/2,
                             bin_spacing)
            tcent=np.append(tcent,tcenti)
        self.tcent=tcent

        # and how many LFEs in each range
        i1=np.searchsorted(self.tms,self.tcent-bin_duration/2)
        i2=np.searchsorted(self.tms,self.tcent+bin_duration/2)
        self.lferate=(i2-i1)

        # also want mean magnitude in each bin
        mg=np.append([0.],np.cumsum(self.mags))
        self.meanmag=np.ndarray(self.lferate.size,dtype=float)*float('nan')
        ii=self.lferate>0
        self.meanmag[ii]=np.divide(mg[i2[ii]]-mg[i1[ii]],self.lferate[ii])
        
        # convert number of LFEs to rate: number per hour
        self.lferate=self.lferate*(3600/bin_duration)
        
        # save binning info
        self.bin_duration=float(bin_duration)
        self.bin_spacing=float(bin_spacing)

    def group_events_bytime(self,tsplits=[0,0.75,3]):
        """
        split the events into groups
        
        Parameters
        ----------
        tsplits :
            time splits relative to start time, in days
        """

        # note splits and labels
        self.tsplits=np.atleast_1d(tsplits)
        if self.tsplits.size==3:
            lbls=['early','late']
        elif self.tsplits.size==4:
            lbls=['early','middle','late']
        else:
            lbls=['group{:d}'.format(k)
                  for k in range(0,self.tsplits.size)]

        # initialize times
        self.groups={}

        # just iterate over events and splits
        for k in range(0,self.tsplits.size-1):
            # limits of timing
            i1=np.searchsorted(self.tms,self.start_times+self.tsplits[k]*86400)
            i2=np.searchsorted(self.tms,self.start_times+self.tsplits[k+1]*86400)

            # in range?
            ii=np.bincount(np.append(i1,i2),
                           np.append(np.ones(self.Nsse),-1*np.ones(self.Nsse)),
                           minlength=self.tms.size+1)
            ii=np.cumsum(ii).astype(bool)[0:-1]
            self.groups[lbls[k]]=np.where(ii)[0]

    def plot_rate(self,iev=0,plot_vred=False,tplot=[-2,7],plot_portions=False):
        """
        plot the event rates

        Parameters
        ----------
        iev : 
            index of event to consider, or year
        plot_vred : 
            also plot the preferred velocity reduction
        tplot : 
            time window in days (default: [-2,7])
        """

        # get the event index if needed
        if iev>self.Nsse:
            yrs=np.array([tm.year for tm in self.sse_tlms[:,0]])
            iev=np.argmin(np.abs(yrs-iev))

        if plot_vred:
            Np=3
        else:
            Np=2

        f=plt.figure(figsize=(10,4*Np))
        gs,p=gridspec.GridSpec(Np,1),[]
        gs.update(left=0.1,right=0.97,bottom=0.2,top=0.92)
        gs.update(hspace=0.05,wspace=0.25)
        p=[plt.subplot(gs[0])]
        for k in range(1,Np):
            p.append(plt.subplot(gs[k],sharex=p[0]))
        p=np.array(p)
        pm=p.reshape([p.size,1])

        # time window
        tlm=self.sse_tlms[iev,:]

        # the relevant LFEs
        ix=np.logical_and(self.tms>=tlm[0],self.tms<=tlm[1])

        # the relevant bins
        ib=np.where(np.logical_and(self.tcent>=tlm[0],self.tcent<=tlm[1]))[0]
        tcent=self.tcent[ib]
        rt=self.lferate[ib]

        # choose a reference time
        tref=self.start_times[iev]

        # magnitudes through time
        p[0].plot((self.tms[ix]-tref)/86400,self.mags[ix],linestyle='none',marker='x',
                  color='k')
        p[0].plot((tcent-tref)/86400,self.meanmag[ib])

        # event rate through time
        p[1].plot((tcent-tref)/86400,rt)

        if 'late' in list(self.groups.keys()):
            ix=self.groups['late']
            p[0].plot((self.tms[ix]-tref)/86400,self.mags[ix],linestyle='none',marker='x',
                      color='r')

        if 'early' in list(self.groups.keys()):
            ix=self.groups['early']
            p[0].plot((self.tms[ix]-tref)/86400,self.mags[ix],linestyle='none',marker='x',
                      color='b')

        p[0].set_ylabel('magnitude')
        p[1].set_ylabel('LFE rate (events per hour)')
        p[Np-1].set_xlabel('time (days)')


        if plot_vred:
            tmsw=self.running_velocity_windows[:,0]
            ii=np.logical_and(tmsw>=tlm[0],tmsw<=tlm[1])
            p[2].plot((tmsw[ii]-tref)/86400,self.running_velocity_values[ii],
                      marker='x')
            p[2].set_ylabel('fractional velocity reduction')
        
        for ph in p:
            for ts in self.tsplits:
                ph.axvline(ts,linestyle=':',color='k')
            ph.set_xlim(tplot)

        if plot_portions:
            portions=self.event_portion[ib]
            ix=portions==1
            p[1].plot((tcent[ix]-tref)/86400,rt[ix],label='beginning',color='red',linestyle='none',marker='*')
            ix=portions==2
            p[1].plot((tcent[ix]-tref)/86400,rt[ix],label='middle',color='goldenrod',linestyle='none',marker='s')
            ix=portions==3
            p[1].plot((tcent[ix]-tref)/86400,rt[ix],label='end',color='blue',linestyle='none',marker='o')

            
    
    #-----END EVENT GROUPING AND LFE RATE ANALYSIS-----
    
    #-----BEGIN ARRIVAL TIME AND SNR ANALYSIS----------

    def set_picks(self,st):
        """
        set the picks to be those in self.stackpicks

        Parameters
        ----------
        st : 
            the traces with unset picks
        """

        for tr in st:
            tr.stats.t0=self.stackpicks[tr.stats.station]
    
    def pick_stacks(self,wlen=None,minsnr=10):
        """
        pick arrival times on the template and compute snr
        
        Parameters
        ----------
        wlen :
             window length to use for computation, in seconds
        minsnr : 
             minimum signal to noise ratio to accept stacks
                    in power (default: 0)
        """

        # save for later
        if wlen is not None:
            self.wlen=np.atleast_1d(wlen)
        else:
            self.tminsnr=minsnr
        
        # select picks on templates and compute snr
        self.stack_snrs,self.totstk,self.stns_stack =\
            compute_snrs_and_picks(self.totstk,minsnr=minsnr,
                                   wlen=np.diff(self.wlen)[0])

        # save the individual picks as well
        self.stackpicks={}
        for tr in self.totstk:
            self.stackpicks[tr.stats.station]=tr.stats.t0

    
    def pick_templates(self,snr_wlen=5,minsnr=20):
        """
        pick arrival times on the template and compute snr
        
        Parameters
        ----------
        snrwlen :
             window length to use for computation, in seconds
                    (default: 5)
        minsnr : 
             minimum signal to noise ratio to accept stacks
                    in power (default: 20)
        """

        # select picks on templates and compute snr
        self.temp_snrs,self.sttemp,self.stns =\
            compute_snrs_and_picks(self.sttemp,minsnr=minsnr,
                                   wlen=snr_wlen)

        # save for later
        self.tsnr_wlen=snr_wlen
        self.tminsnr=minsnr

    #-----END ARRIVAL TIME AND SNR ANALYSIS------------

    #-----BEGIN STACKING-------------------------------

    def plot_stacks(self,stype='template',Ns=3,tlm=[-15,15],stn=None,normalize=True,
                    cmps=['E','N','Z'],topick=True,color='black',morestacks=[]):
        """
        plot some of the stacks

        Parameters
        ----------
        stype :
            which stack (default: 'template','all', some group's name,
                          or subtracted groups)
        Ns :
            max number of stations to plot (default: 3)
        tlm :
            time window to plot, relative to pick 
                (default: [-15,15])
        stn :
            station or stations to plot, if known
        normalize :
            normalize each trace? (default: True)
        cmps : 
            components to plot (default: ['E','N','Z'])
        topick :
            use times relative to the pick? (default: True)
        color :
            what color to plot with
        morestacks : 
            any additional stacks to plot, eg ['early','late']
        """

        if isinstance(stype,obspy.Stream):
            st=stype
            stype='input'
            wlen=[0,self.tsnr_wlen]
        elif stype=='template':
            st=self.sttemp
            wlen=[0,self.tsnr_wlen]
        elif stype=='all':
            st=self.totstk
            wlen=[0,self.tsnr_wlen]
        elif 'template' in stype:
            stype=stype.split('template-')[1]
            st=self.modstack[stype]
            wlen=[0,self.tsnr_wlen]
        elif stype in self.grpstk.keys():
            st=self.grpstk[stype]
            wlen=[0,self.tsnr_wlen]
            self.set_picks(st)
        elif stype in self.diffstk.keys():
            st=self.diffstk[stype]
            wlen=[0,self.tsnr_wlen]
            self.set_picks(st)

        # additional stacks
        if isinstance(morestacks,str):
            morestacks=[morestacks]
        morestacks_string=np.atleast_1d(morestacks).astype(str)
        for k in range(0,len(morestacks)):
            morestacks[k]=self.grpstk[morestacks[k]]
        morecolors=['red','blue','green']
            
        # select some stations
        if stn is None:
            stn=np.unique([tr.stats.station for tr in st])
            Ns=np.minimum(stn.size,Ns)
            stn=np.random.choice(stn,Ns,replace=False)
        else:
            if isinstance(stn,str):
                stn=[stn]
            stn=np.atleast_1d(stn)
            Ns=stn.size

        # sort picks by arrival time
        pks=[st.select(station=stni)[0].stats.t0
             for stni in stn]
        ix=np.argsort(pks)
        stn=stn[ix]
            
        # components
        if isinstance(cmps,str):
            cmps=[cmps]
        cmps=np.atleast_1d(cmps)
        Nc=cmps.size
            
        
        xtext=tlm[0]+np.diff(tlm)[0]*0.01

        # figure
        f=plt.figure(figsize=(8,Nc*1.5/3*Ns+4))

        p=[]
        scl=1.
        tref=0.
        for ks in range(0,Ns):
            sti=st.select(station=stn[ks])
            stm=[stmi.select(station=stn[ks]) for stmi in morestacks]

            # if we need to rotate
            if 'R' in cmps or 'T' in cmps:
                azm=self.stataz[stn[ks]]
                sti=self.project_waveforms(sti,stns=stn[ks],y_azimuths=azm)
                for tr in sti.select(channel='X*'):
                    tr.stats.channel='T'
                for tr in sti.select(channel='Y*'):
                    tr.stats.channel='R'

                for k in range(0,len(stm)):
                    stm[k]=self.project_waveforms(stm[k],stns=stn,y_azimuths=azm)
                    for tr in stm[k].select(channel='X*'):
                        tr.stats.channel='T'
                    for tr in stm[k].select(channel='Y*'):
                        tr.stats.channel='R'

            
            # each component
            ymax,ymaxp=0.,0.
            for kc in range(0,Nc):
                pe=plt.subplot(Ns*Nc,1,ks*Nc+1+kc)
                tr=sti.select(channel=cmps[kc])[0]
                if normalize:
                    scl=np.max(np.abs(tr.data))
                    ymax=1.
                else:
                    ymax=np.maximum(ymax,np.max(np.abs(tr.data)))

                if topick:
                    tref=tr.stats.t0
                hr,=pe.plot(tr.times()-tref,tr.data/scl,color=color)
                ymaxp=np.maximum(ymaxp,np.max(np.abs(tr.data))/scl)
                
                pe.text(0.01,0.95,'.'.join([tr.stats.station,tr.stats.channel]),
                        verticalalignment='top',transform=pe.transAxes)
                pe.set_ylim(np.array([-1,1])*1.05)

                hm=[]
                for km in range(0,len(morestacks)):
                    tr=stm[km].select(channel=cmps[kc])[0]
                    hh,=pe.plot(tr.times()-tref,tr.data/scl,color=morecolors[km])
                    ymaxp=np.maximum(ymaxp,np.max(np.abs(tr.data))/scl)
                    hm.append(hh)

                p=p+[pe]
            for ph in p[-Nc:]:
                ph.set_ylim(np.array([-1,1])*1.05*ymaxp)
    
        for ph in p:
            if wlen is not None:
                pass
                #ph.axvspan(wlen[0],wlen[1],zorder=0,alpha=0.3,color='red')
            ph.set_xlim(tlm)
        for ph in p[0:-1]:
            ph.set_xticklabels('')
        p[-1].set_xlabel('time (s)')
        imid=int(len(p)/2)
        p[imid].set_ylabel('normalized ground velocity')
        
        #p[0].set_title(stype)
        if len(morestacks):
            lg=p[0].legend([hr]+hm,np.append([stype],morestacks_string),
                           loc='upper right',fontsize='large')
    
    def stack_by_group(self,minevents=100,Nboot=10,renormalize=False,
                       amp_range=None,weighting='even',group_weights=None):
        """
        stacks the seismograms

        Parameters
        ----------
        minevents :
            minimum number of events to use
        Nboot :
            number of times to bootstrap the data
        renormalize :
            renormalize the data component by component (default: False)
        amp_range : 
            only include LFEs with amplitudes in this range
        weighting :
            how to weight the traces
                'even': each trace evenly (default)
                'over_max': divide by max value
        group_weights :
            any additional weighting to use in the groups only, per event (default: all ones)
        """

        # note minimum number of events
        self.minevents=minevents
        self.normalize_by_component=bool(renormalize)
        self.Nboot=Nboot

        # note that stacks are unfiltered
        self.stack_flm=np.array([0,float('inf')])

        # initialize
        tr=obspy.Trace()
        nvl=int(self.tlen*self.sampling_rate)
        tr.data=np.ndarray(nvl,dtype=float)
        tr.stats.sampling_rate=self.sampling_rate
        self.totstk=obspy.Stream()
        self.grpstk=dict([(ky,obspy.Stream()) for ky in self.groups.keys()])
        self.totstkb={}
        self.grpstkb=dict([(ky,{}) for ky in self.groups.keys()])

        # events with desired amplitudes
        if amp_range is not None:
            evok=np.logical_and(self.evamp>=amp_range[0],
                                self.evamp<=amp_range[1])
            evok=np.where(evok)[0]
        else:
            evok=np.arange(0,self.tms.size)


        # choose timing for bootstrapping
        Nsamp=int(evok.size*0.8)
        iboot=np.random.choice(evok,Nsamp*Nboot,replace=True)
        iboot=iboot.reshape([Nsamp,Nboot])

        if group_weights is None:
            group_weights=np.ones(evok.size,dtype=float)
        
        for idi in self.data.keys():
            stn,chn=idi.split('.')
            if self.evperchan[idi]>=minevents:

                # note the events
                evi=self.ev[idi]
                
                # normalize each seismogram
                datai=self.data[idi]
                if renormalize:
                    datai=np.divide(datai[:,evok],np.max(np.abs(datai[:,evok]),axis=0,keepdims=True))

                # choose weighting of each trace
                if 'even' in weighting:
                    wgts=np.ones(datai.shape[1],dtype=float)
                elif 'over_max' in weighting:
                    wgts=np.divide(self.data_norm[stn],self.data_max[stn])
                    
                # maybe also weight by amplitude
                if 'with_amp' in weighting:
                    amps=self.evamp[evi]
                    amps[np.isnan(amps)]=0.
                    wgts=np.multiply(wgts,amps)

                # normalize weights
                wgts=wgts/np.sum(wgts)
                
                # and average total
                stk=np.dot(datai,wgts)
                tr.data=stk
                tr.stats.station=stn
                tr.stats.channel=chn
                
                # add to set
                self.totstk.append(tr.copy())

                # now that the total stack is done, add weights for the groups
                addweight=group_weights[evi]
                wgts=np.multiply(wgts,addweight)
                wgts=wgts/np.sum(wgts)

                # also for the groups
                for ky in self.groups.keys():
                    # note the events we want to analyse
                    evdes=np.intersect1d(self.groups[ky],evok)
                    # find the events with available data
                    evj,ia,ib=np.intersect1d(evi,evdes,return_indices=True)
                    if ia.size:
                        # save total stack
                        wgtsh=wgts[ia]/np.sum(wgts[ia])
                        tr.data=np.dot(datai[:,ia],wgtsh)
                        self.grpstk[ky].append(tr.copy())

                        # and initialize bootstrap for this station/group
                        self.grpstkb[ky][idi]=np.ndarray([nvl,Nboot])*float('nan')
                        
                # initialize bootstrap for this station
                self.totstkb[idi]=np.ndarray([nvl,Nboot])*float('nan')

                evii=np.append(evi,self.tms.size)
                for kb in range(0,Nboot):
                    # stack for all these bootstrap samples
                    ia=np.searchsorted(evi,iboot[:,kb],side='left')
                    ia=ia[(evii[ia]-iboot[:,kb])==0]
                    if ia.size:
                        wgtsh=wgts[ia]/np.sum(wgts[ia])
                        self.totstkb[idi][:,kb]=np.dot(datai[:,ia],wgtsh)

                    # and for each group
                    for ky in self.groups.keys():
                        # check if bootstrap times are in desired group
                        ibooti=iboot[:,kb]
                        ibooti=ibooti[np.array([ib in self.groups[ky] for ib in ibooti])]
                        # and then look for them in the data
                        ia=np.searchsorted(evi,ibooti,side='left')
                        ia=ia[(evii[ia]-ibooti)==0]
                        if ia.size:
                            wgtsh=wgts[ia]/np.sum(wgts[ia])
                            self.grpstkb[ky][idi][:,kb]=np.dot(datai[:,ia],wgtsh)

        # replace channel names with just E, N, Z
        self.replace_channels(self.totstk)
        self.replace_channels(self.totstkb)
        for st in self.grpstk.values():
            self.replace_channels(st)
        for dct in self.grpstkb.values():
            self.replace_channels(dct)

    
    #-----END STACKING---------------------------------
    

    #-----BEGIN DATA LOADING----------------------------

    def set_data_directory(self,directory_name=None):
        """
        Parameters
        ----------
        directory_name :
              the directory where all the data will be stored
                (default: the environmental variable $DATA/LFEAnalysis, if present
                          the current working directory, otherwise)

              This directory will contain some catalogues as well
               as a subdirectory 'LFEData', which includes the seismograms
        """

        if directory_name is None:
            try:
                directory_name=os.path.join(os.environ['DATA'],'LFEAnalysis')
            except:
                directory_name=os.get_cwd()

        self.data_directory=directory_name
                
        # check that the directory exists
        if not os.path.exists(self.data_directory):
            print('Data directory {:s} does not exist'.format(self.data_directory))

                
        
    def directory(self):
        """
        Returns
        -------
        fdir :
           directory for storing information
        """

        # directory
        fdir='Family{:d}'.format(self.fnum)
        fdir=os.path.join(self.data_directory,self.region,fdir)

        if not os.path.exists(fdir):
            os.makedirs(fdir)

        return fdir
    
    def read_data(self):
        """
        read in the seismic data
        """

        # relevant files
        fdir=self.directory()
        fname=os.path.join(self.directory(),'data_*-*')
        fls=glob.glob(fname)
        
        if len(fls):
            # first file
            fname=fls[0]
            with open(fname, "rb") as fl:
                data=pickle.load(fl)
            fname=fname.replace('data','events')
            with open(fname, "rb") as fl:
                ev=pickle.load(fl)
                
            # and add
            for fnamei in fls[1:]:
                with open(fnamei, "rb") as fl:
                    datai=pickle.load(fl)
                fname=fnamei.replace('data','events')
                with open(fname, "rb") as fl:
                    evi=pickle.load(fl)
            
                for idi in data.keys():
                    data[idi]=np.append(data[idi],datai[idi],axis=1)
                    ev[idi]=np.append(ev[idi],evi[idi])

        # save info
        self.data=data
        self.ev=ev

        # note how many per channel
        self.evperchan=dict([(ky,self.data[ky].shape[1]) for ky in self.data.keys()])


        
    #-----END DATA LOADING------------------------------

    #-----BEGIN SAVING AND RELOADING--------------------

    def collect_energies(self,fnums=[41]):
        """
        collect the energies computed for several families
        """

        # note family numbers
        fnums=np.atleast_1d(fnums)
        flocs=np.ndarray([0,3])

        # diffen,diffenb,toten,totenb,statxy,statxym,takeoff_angles
        simpadd=['toten','totenb','statxy','statxym','takeoff_angles','statloc',
                 'arrival_times','totpol','totpolb']
        simpadd=['statxy','statxym','takeoff_angles','statloc',
                 'arrival_times']

        # try adding 'scalings' to the list above
        

        
        #subadd=['diffen','diffenb','diffpol','diffpolb','scalingsc','scalingscb']
        subadd=['scalingsc','scalingscb','scalings']
        listadd=['stacked_velocity_reduction']
        listadd=[]
        
        # initialize an object
        lfi=lfanalyse(lfi=self)
        
        for k in range(0,len(fnums)):

            
            fnum=fnums[k]
            print('Family {:d}'.format(fnum))
            
            # read the results for this family
            lfi.fnum=fnum
            lfi.read_results()

            # adjust the station names
            lfi.add_number_to_station()

            # station locations
            lfi.relative_station_locations()
            lfi.find_takeoff_angles(refdepth=30.)
            
            # add locations
            flocs=np.append(flocs,lfi.floc.reshape([1,3]),axis=0)
            
            if k==0:
                # copy over if this is the first
                for ky in lfi.__dict__.keys():
                    self.__setattr__(ky,lfi.__getattribute__(ky))
                for ky in listadd:
                    dct2=lfi.__getattribute__(ky)
                    self.__setattr__(ky,dict.fromkeys(dct2.keys(),{}))
                    dct1=self.__getattribute__(ky)
                    for ky in dct2.keys():
                        dct1[ky]={}
                        dct1[ky][lfi.fnum]=dct2[ky]
                        
            else:
                # append if it's not
                for dname in simpadd:
                    dct1=self.__getattribute__(dname)
                    dct2=lfi.__getattribute__(dname)
                    dct1.update(dct2)
                for dname in subadd:
                    dct1=self.__getattribute__(dname)
                    dct2=lfi.__getattribute__(dname)
                    for grp in dct1.keys():
                        dct1[grp].update(dct2[grp])
                for dname in listadd:
                    dct1=self.__getattribute__(dname)
                    dct2=lfi.__getattribute__(dname)
                    for ky in dct2.keys():
                        dct1[ky][lfi.fnum]=dct2[ky]
                
        self.fnums=fnums
        self.flocs=flocs
        self.floc=np.mean(self.flocs,axis=0)
            
    def add_number_to_station(self):
        """
        add the family number to the station names
        """

        # dictionaries to modify
        # energies

        #dcts=[self.toten,self.totenb,self.totpol,self.totpolb]
        dcts=[]
        #dcts=dcts+list(self.diffen.values())+list(self.diffenb.values())
        #dcts=dcts+list(self.diffpol.values())+list(self.diffpolb.values())
        dcts=dcts+list(self.scalingsc.values())+list(self.scalingscb.values())
        
        # station info
        print(self.statxy)
        print(self.scalings.values())
        dcts=dcts+[self.statxy,self.statloc]+list(self.scalings.values())
        if 'statxym' in self.__dict__.keys():
            dcts=dcts+[self.statxym,self.takeoff_angles,self.arrival_times]

        # what to add to the keys
        kyadd='{:d}-'.format(self.fnum)
        
        for dct in dcts:
            # for each key, replace in new location
            kys=list(dct.keys())
            for ky in kys:
                dct[kyadd+ky]=dct.pop(ky)
            
    
    def read_results(self):
        """
        read the results 
        """

        # where to save
        fname=os.path.join(self.directory(),'Results')

        # save
        with open(fname, "rb") as fl:
            lfi=pickle.load(fl)
        
        # copy information from the saved object
        if lfi is not None:
            for ky in lfi.__dict__.keys():
                self.__setattr__(ky,lfi.__getattribute__(ky))


    def save_results(self):
        """
        save all the processed information except for the original seismograms
        """

        # make a copy?
        lfi=lfanalyse(lfi=self)

        # remove the data from the copy
        if 'data' in lfi.__dict__.keys():
            lfi.__delattr__('data')

        # where to save
        fname=os.path.join(self.directory(),'Results')

        # save
        with open(fname, "wb") as fl:
            pickle.dump(lfi, fl)

    
    #-----END SAVING AND RELOADING----------------------

    #-----BEGIN DATA NAMING CONVENTIONS-----------------

    
    def observation_channels(self,st,remove_unknown=True):
        """
        replace the channel name with that written on the station,
        eg 'EH1','BHE',etc

        Parameters
        ----------
        st :
             the obspy stream to edit
        remove_unknown : 
             remove traces where the channel is unknown
              (default: True)
        """

        # mapping
        # we'll just drop the first two letters of the channel
        dmap={'1':'E','2':'N','Z':'Z',
              'E':'E','N':'N'}

        # create a map from 'E','N','Z' to instrument channels
        chanmap={}
        for stn in self.channels.keys():
            chanmap[stn]={}
            for chan in self.channels[stn]:
                nky=dmap.get(chan[-1])
                chanmap[stn][nky]=chan
        self.chanmap=chanmap

        # and edit the streams
        for tr in st:
            if tr.stats.station in chanmap:
                tr.stats.channel=\
                    chanmap[tr.stats.station].get(tr.stats.channel,tr.stats.channel)
            elif remove_unknown:
                st.remove(tr)

    def add_networks(self,st,remove_unknown=True):
        """
        add the network names to the obspy traces

        Parameters
        ----------
        st :
             the obspy stream to edit
        remove_unknown : 
             remove traces where the channel is unknown (default: True)
        """

        for tr in st:
            if tr.stats.station in self.networks:
                tr.stats.network=self.networks.get(tr.stats.station,tr.stats.network)
            elif remove_unknown:
                st.remove(tr)
            
    def replace_channels(self,st):
        """
        replace the channel name with 'E','N', or 'Z'

        Parameters
        ----------
        st :
             the obspy stream to edit
        """

        # mapping
        # we'll just drop the first two letters of the channel
        dmap={'1':'E','2':'N','Z':'Z',
              'E':'E','N':'N'}

        # make it a stream
        if isinstance(st,obspy.Trace):
            st=obspy.Stream(st)

        if isinstance(st,obspy.Stream):
            # replace each one
            for tr in st:
                tr.stats.channel=dmap.get(tr.stats.channel[-1],tr.stats.channel[-1])
        elif isinstance(st,dict):
            kys=list(st.keys())
            for oky in kys:
                # new key
                stn,chn=oky.split('.')
                chn=dmap.get(chn[-1])
                nky='.'.join([stn,chn])
                # replace
                st[nky]=st.pop(oky)
                
    #-----END DATA NAMING CONVENTIONS-------------------

    
    #-----BEGIN DATA FILTERING AND NORMALIZATION--------

    def filter_data(self,flm=[1,8]):
        """
        filter the input data
        
        Parameters
        ----------
        flm : 
            frequency band to filter to, in Hz
                (default: [1,8])
        """

        # frequency band and sampling rate
        self.flm=np.atleast_1d(flm)
        fs=self.sampling_rate

        if self.flm[0]>0 or self.flm[1]<float('inf'):
            # construct filter
            if self.flm[0]>0 and self.flm[1]<float('inf'):
                # bandpass filter
                sos=signal.butter(4,self.flm,btype='bandpass',fs=fs,output='sos')
            elif self.flm[0]>0:
                # highpass filter
                sos=signal.butter(4,self.flm[0],btype='highpass',fs=fs,output='sos')
            elif self.flm[1]<float('inf'):
                # lowpass filter
                sos=signal.butter(4,self.flm[1],btype='lowpass',fs=fs,output='sos')

            # apply to each trace
            for ky in self.data:
                data=self.data[ky]
                if data.size:
                    # detrend first
                    signal.detrend(data,axis=0,type='linear',overwrite_data=True)
                    # and filter
                    self.data[ky]=signal.sosfiltfilt(sos,data,axis=0,padtype='constant')


    def discard_data(self,max_value=30):
        """
        discard data with values that are too big

        Parameters
        ----------
        max_value : 
             max value allowed relative to median max value
        """

        kys=np.array(list(self.data.keys()))
        stns=np.array([ky.split('.')[0] for ky in kys])
        astns=np.unique(stns)

        for stn in astns:
            # which keys are relevant here
            ix=stns==stn
            kysh=kys[ix]
            if len(kysh)==3 and self.data_max[stn].size:

                # max values relative to the normalizations
                mx=np.divide(self.data_max[stn],self.data_norm[stn])
                iok=mx/np.nanmedian(mx)<max_value

                # change the data and indexing
                for ky in kysh:
                    self.data[ky]=self.data[ky][:,iok]
                    self.ev[ky]=self.ev[ky][iok]

                # and change the noted max values
                self.data_norm[stn]=self.data_norm[stn][iok]
                self.data_max[stn]=self.data_max[stn][iok]

        
                
    def normalize_data(self,single_norm=False):
        """
        normalize the seismograms by the max amplitude on each component

        Parameters
        ----------
        single_norm :
               just use a single norm per station, not per interval
                (default: False)
        """

        # stations and channels of interest
        kys=np.array(list(self.data.keys()))
        stns=np.array([ky.split('.')[0] for ky in kys])
        chans=np.array([ky.split('.')[1] for ky in kys])
        astns=np.unique(stns)

        
        # normalizations to save
        self.data_norm=dict([(stn,[]) for stn in astns])
        self.data_max=dict([(stn,[]) for stn in astns])

        for stn in astns:
            ii=stns==stn
            if np.sum(ii)==3:
                # grab the relevant data and events
                iky=kys[ii]
                data=[self.data[ky] for ky in iky]
                ev=[self.ev[ky] for ky in iky]
                
                # events in all of them
                iev=np.intersect1d(ev[0],ev[1])
                iev=np.intersect1d(iev,ev[2])
                iev.sort()

                mx=np.zeros([1,iev.size])
                for k in range(0,3):
                    # grab the relevant data from each
                    ieva,ia,ib=np.intersect1d(iev,ev[k],return_indices=True)
                    datai=data[k][:,ib]

                    # compute the maxima
                    mx=np.maximum(mx,np.max(np.abs(datai),axis=0))

                # note the max values
                self.data_max[stn]=mx.flatten()
                    
                # if we want just one norm per station,
                # replace the values per interval with the same values
                if mx.size>0 and single_norm:
                    mx=np.ones(mx.shape)*np.median(mx.flatten())
                    
                # save the maxima
                self.data_norm[stn]=mx.flatten()

                for k in range(0,3):
                    # grab relevant data again
                    ky=iky[k]
                    ieva,ia,ib=np.intersect1d(iev,ev[k],return_indices=True)

                    # normalize and resave
                    self.data[ky]=np.divide(data[k][:,ib],mx)
                    self.ev[ky]=ieva


    #-----END DATA FILTERING AND NORMALIZATION----------


    #-----BEGIN DATA DOWNLOAD AND SAVING----------------


    def identify_channels(self,cdes=['E','B','H']):
        """
        identify the available components and stations to download

        Parameters
        ----------
        cdes :
              the channel types to record in order of priority
              (default: ['E','B','H']
                  first look for EH?, then for BH?, then for HH?)
        """
        
        # rough location
        latlim=self.latlim
        lonlim=self.lonlim

        # open the client
        from obspy.clients.fdsn.client import Client
        clnt=Client('IRIS')

        # find the relevant stations and channels
        inv=obspy.Inventory()
        
        # channels in order
        cdes=np.atleast_1d(cdes)
        
        # channels to retrieve
        channels={}
        networks={}

        # create the station inventory
        stnlist=','.join(self.stns)
        inva=clnt.get_stations(station=stnlist,minlatitude=latlim[0],maxlatitude=latlim[1],
                               minlongitude=lonlim[0],maxlongitude=lonlim[1],
                               channel='?H?',level='channel',network='*')
        
        # find the stations available
        for stn in self.stns:
            found_channel=False
            k=0
            while not found_channel and k<len(cdes):
                invi=inva.select(station=stn,channel=cdes[k]+'H?')
                if len(invi):
                    networks[stn]=invi[0].code
                    channels[stn]=np.unique([invj.code for invj in invi[0][0].channels])
                    if len(channels[stn])>=3:
                        found_channel=True
                k=k+1
                inv=inv+invi


    
        # save the stations with 3 channels
        stns=np.array(list(channels.keys()))
        iok=np.array([len(channels[stn])==3 for stn in stns])
        stns=stns[iok]
        channels=dict([(stn,channels[stn]) for stn in stns])
        networks=dict([(stn,networks[stn]) for stn in stns])

        print('Identified channels for {:d} of {:d} stations'.format(stns.size,self.stns.size))
        
        # save this info
        self.stns=stns
        self.channels=channels
        self.networks=networks
        
        # note the inventory
        self.inventory=inv

    def make_data_list(self,tm,tbuf=10):
        """
        Parameters
        ----------
        tm : 
            time of interest
        tbuf : 
            buffer to add at each end, in s (default: 10)
        
        Returns
        -------
        bulk :
            a list of waveform intervals to submit to the fdsn server
        """
    
        # time window
        tlen=self.tlen
        t1,t2=tm-tbuf-tlen/4,tm+tlen*3/4+tbuf
        
        # create a list of waveforms to get
        bulk=[(self.networks[stn],stn,'*',','.join(self.channels[stn]),t1,t2)
              for stn in self.stns]
    
        return bulk



    def download_data(self,max_events=float('inf')):
        """
        retrieve data from IRIS via an FDSN downloader

        Parameters
        ----------
        max_events :
            the maximum number of data intervals to download, 
              usually used if testing a new area
              (default: float('inf')---all of them)

        """

        # open the client
        from obspy.clients.fdsn.client import Client
        clnt=Client('IRIS')
        
        # length to download and buffer
        tlen=self.tlen
        tbuf=10

        # choose a sampling rate
        srate = self.sampling_rate

        # buffer needs to have integer multiple of timestep
        tbuf=np.round(tbuf*srate)/srate

        # initialize output
        nvl=int(self.tlen*srate)
        ids=np.array([['.'.join([stn,cmp]) for cmp in self.channels[stn]] 
                      for stn in self.channels.keys()]).flatten()
        data=dict([(idi,np.ndarray([nvl,0])) for idi in ids])
        ev=dict([(idi,np.ndarray(0,dtype=int)) for idi in ids])

        # times
        tm=self.tms
        nev=int(np.minimum(tm.size,max_events))

        if nev<tm.size:
            print('ONLY DOWNLOADING THE FIRST {:d} EVENTS'.format(nev))
        
        i1=0
        # go through and download data
        for ctr in range(0,nev):
            print('Downloading for event {:d} of {:d}'.format(ctr+1,tm.size))
            tmi=tm[ctr]
            t1,t2=tmi-tbuf-tlen/4,tmi+tlen*3/4+tbuf
            bulk=self.make_data_list(tm=tmi,tbuf=tbuf)
            
            # and download
            try:
                st=clnt.get_waveforms_bulk(bulk,attach_response=True)
            except:
                print('No data?')
                st=obspy.Stream()
                
            # trim and pad
            st.trim(starttime=t1,endtime=t2,pad=True)
        
            # remove anything with gaps
            for tr in st:
                if isinstance(tr.data,np.ma.masked_array):
                    if np.sum(tr.data.mask)>0.01*tr.stats.npts:
                        st.remove(tr)
                    else:
                        msk=seisproc.prepfiltmask(tr)
        
            # correct for response and bandpass filter
            for tr in st:
                try:
                    tr.remove_response(output='VEL',pre_filt=(1/15, 1/5, 15, 20),
                                       water_level=100)
                except:
                    st.remove(tr)
    
            # resample
            st.interpolate(sampling_rate=srate)
            
            # trim again
            st.trim(starttime=tmi-tlen/4,endtime=tmi+tlen*3/4)
    
            # and add to set
            for tr in st:
                idi='.'.join([tr.stats.station,tr.stats.channel])
                data[idi]=np.append(data[idi],tr.data[0:nvl].reshape([nvl,1]),axis=1)
                ev[idi]=np.append(ev[idi],ctr)
                
            if (ctr>i1 and (ctr+1)%100==0) or ctr==nev-1:
                print('Writing to file and continuing')
                fdir=self.directory()
                fname='data_{:d}-{:d}'.format(i1,ctr)
                with open(os.path.join(fdir,fname), "wb") as fl:
                    pickle.dump(data, fl)
                fname=fname.replace('data','events')
                with open(os.path.join(fdir,fname), "wb") as fl:
                    pickle.dump(ev, fl)
                i1=ctr+1
                  
                # reset data to save
                data=dict([(idi,np.ndarray([nvl,0])) for idi in ids])
                ev=dict([(idi,np.ndarray(0,dtype=int)) for idi in ids])

                # restart the client
                clnt=Client('IRIS')

    

    
    #-----END DATA DOWNLOAD AND SAVING------------------

    #-----BEGIN STATION INFORMATION AND ORIENTATION-----

    def read_station_locations(self):
        """
        read the station locations from a file
        """

        # first get an inventory
        fdir=self.directory()
        fname=os.path.join(fdir,'station_inventory.xml')
        self.inventory=obspy.read_inventory(fname,format='STATIONXML')

        # available channels
        chans=self.inventory.get_contents()['channels']

        # add to dictionary
        self.statloc={}
        self.station_local_depth={}
        for chan in chans:
            # this station
            nw,stn,lc,chn=chan.split('.')
            # location
            xy=self.inventory.get_coordinates(chan)
            self.statloc[stn]=\
                np.array([xy['longitude'],xy['latitude'],xy['elevation']])
            
            # and station depth?
            self.station_local_depth[stn]=xy['local_depth']

    
    def relative_station_locations(self):
        """
        computes LFE-station distances, in km
        and azimuths from the LFEs to the stations
        """
        
        # read station locations if needed:
        if not 'statloc' in self.__dict__.keys():
            self.read_station_locations()

        # for each station: azimuth from earthquake to station
        self.stataz={}
        self.statdst={}
        self.statxy={}
        for stn in self.statloc.keys():
            dst,az1,az2=obspy.geodetics.base.gps2dist_azimuth(\
                 self.floc[1],self.floc[0],
                 self.statloc[stn][1],self.statloc[stn][0])
            self.stataz[stn]=az1
            self.statdst[stn]=dst/1000
            self.statxy[stn]=dst/1000*np.array([np.sin(az1*np.pi/180),
                                                np.cos(az1*np.pi/180)])


    def project_waveforms(self,st,stns=None,y_azimuths=[0.,30.,60.]):
        """
        Parameters
        ----------
        st : 
           set of waveforms
        stns : 
           stations to consider
        y_azimuths : 
           azimuths to project to, in degrees

        Returns
        -------
        strot :
           set of rotated waveforms
        """

        # stations
        if stns is None:
            stns=np.unique([tr.stats.station for tr in st])
        elif isinstance(stns,str):
            stns=[stns]
        stns=np.atleast_1d(stns)

        # azimuths
        y_azimuths=np.atleast_1d(y_azimuths).astype(float)

        # outputs
        strot=obspy.Stream()
        
        for stn in stns:
            sti=st.select(station=stn)
            for azm in y_azimuths:
                # the relevant data
                tre=sti.select(channel='E')[0]
                trn=sti.select(channel='N')[0]

                # angle in radians
                thet=np.pi/180*azm

                # new y/N
                tr_y=trn.copy()
                tr_y.data=tre.data*np.sin(thet)+trn.data*np.cos(thet)
                tr_y.stats.channel='Y_{:0.0f}'.format(azm)

                # new x/E
                tr_x=tre.copy()
                tr_x.data=tre.data*np.cos(thet)-trn.data*np.sin(thet)
                tr_x.stats.channel='X_{:0.0f}'.format(azm)

                # add to set
                strot.append(tr_x)
                strot.append(tr_y)

        return strot


    

    def relative_to_fault(self):
        """
        """

        # convert takeoff angle and azimuth to fault coordinates
        # phi: angle from the fault perpendicular
        # theta: the angle from the slip direction to the takeoff,
        #            measured within the fault plane



        if 'statxym' in self.__dict__.keys():
            statxy=self.statxym
        else:
            statxy=self.statxy
            
        # get the station azimuths clockwise from
        # the horizontal slip direction
        kys=list(statxy.keys())
        xy=np.array([statxy[ky] for ky in kys])
        azm=np.angle(xy[:,1]+1j*xy[:,0])*180/np.pi
        azm=(azm-self.slipdir) % 360.

        # also grab the takeoff angles
        tkg=np.array([self.takeoff_angles[ky][0] for ky in kys])

        # compute components in the horizontal slip direction (x),
        # the horizontal direction perpendicular to slip (y=x2)
        # and the vertical (z, positive up)
        x=np.multiply(np.sin(tkg*np.pi/180),
                      np.cos(azm*np.pi/180))
        x2=np.multiply(np.sin(tkg*np.pi/180),
                       -np.sin(azm*np.pi/180))
        z=-np.cos(tkg*np.pi/180)

        # and project from x-z to x1-x3
        x1=np.multiply(x,np.cos(self.faultdip*np.pi/180))+\
            np.multiply(z,-np.sin(self.faultdip*np.pi/180))
        x3=np.multiply(x,-np.sin(self.faultdip*np.pi/180))+\
            np.multiply(z,np.cos(self.faultdip*np.pi/180))

        # note the phi and theta values
        # as defined by Stein and Wysession
        phi=np.angle(x1+1j*x2)*180/np.pi
        theta=np.power(np.power(x1,2)+np.power(x2,2),0.5)
        theta=np.angle(x3+1j*theta)*180/np.pi
        
        # create a dictionary with the orientations
        drs=np.stack([x1,x2,x3],axis=1)
        self.takeoff_from_fault=dict([(kys[k],drs[k,:])
                                      for k in range(0,len(kys))])
        self.theta_from_fault=dict([(kys[k],theta[k])
                                    for k in range(0,len(kys))])
        self.phi_from_fault=dict([(kys[k],phi[k])
                                  for k in range(0,len(kys))])

        # create unit vectors in the R,theta,phi directions
        # in the x1,x2,x3 coordinate system
        phivec=[-np.sin(phi*np.pi/180),np.cos(phi*np.pi/180),
                np.zeros(phi.size,dtype=float)]
        phivec=np.stack(phivec,axis=1)
        
        self.unit_vec_R=dict([(kys[k],drs[k,:])
                              for k in range(0,len(kys))])
        self.unit_vec_phi=dict([(kys[k],phivec[k,:])
                                for k in range(0,len(kys))])
        self.unit_vec_theta=dict([(kys[k],np.cross(phivec[k,:],drs[k,:]))
                                  for k in range(0,len(kys))])

    def grid_takeoff_angles(self,refdepth=30.,plot=True):
        """
        compute the takeoff angles for these stations
        and map the station locations to what you'd get 
        for an LFE at a reference depth

        Parameters
        ----------
        refdepth :
            reference depth in km (default: 30.)
        plot :
            plot the results
        """

        # note reference detph
        self.refdepth=float(refdepth)
        
        # distances in km
        dst=np.arange(0,200,1)

        # and degrees
        dstd=obspy.geodetics.base.kilometers2degrees(1)*dst

        # initialize list
        tklist_bigS=[]
        tklist_litS=[]
        
        # initialize velocity model
        from obspy import taup
        mdl=taup.tau.TauPyModel(model='iasp91')

        for k in range(0,len(dstd)):
            # find arrivals
            arvl=mdl.get_travel_times(source_depth_in_km=self.refdepth,
                                      distance_in_degree=dstd[k],
                                      phase_list=['s'])
            tklist_litS.append([arv.takeoff_angle for arv in arvl])

            arvl=mdl.get_travel_times(source_depth_in_km=self.refdepth,
                                      distance_in_degree=dstd[k],
                                      phase_list=['S'])
            tklist_bigS.append([arv.takeoff_angle for arv in arvl])

        self.tkang_bigS=tklist_bigS
        self.tkang_litS=tklist_litS
        self.dst_grid=dst

        # note some reference distances

        # the first location with a downgoing S
        ndown=np.array([len(vl) for vl in self.tkang_bigS])
        rdown=np.append(self.dst_grid[np.where(ndown>0)[0]],500)
        rdown=np.min(rdown)
        tdown=np.append(np.array(self.tkang_litS).flatten()[np.where(ndown>0)[0]],-500)
        tdown=np.max(tdown)
        # when the upgoing s gets to 45 degrees
        mnang=np.array([np.min(np.append(vl,180)) for vl in self.tkang_litS])
        r45=np.append(self.dst_grid[np.where(mnang<135)[0]],500)
        r45=np.min(r45)
        self.ref_distances=np.array([rdown,r45])
        self.ref_takeoff=np.array([tdown,45])
        
        if plot:
            f=plt.figure()
            p=plt.axes()

            for k in range(0,len(self.dst_grid)):
                x=np.ones(len(self.tkang_bigS[k]))*self.dst_grid[k]
                p.plot(x,self.tkang_bigS[k],marker='x',linestyle='none',
                       color='navy')
                x=np.ones(len(self.tkang_litS[k]))*self.dst_grid[k]
                p.plot(x,self.tkang_litS[k],marker='x',linestyle='none',
                       color='firebrick')

            p.set_xlim([np.min(self.dst_grid),np.max(self.dst_grid)])
            p.axvline(rdown,linestyle='--')
            p.axvline(r45,linestyle='-.')
            p.axhline(135,linestyle='-.')
            p.set_xlabel('distance to station (km)')
            p.set_ylabel('takeoff angle (degrees from down)')
            p.set_ylim([0,180])
            p.set_yticks(np.arange(0,181,45))

    def find_takeoff_angles(self,refdepth=30.):
        """
        compute the takeoff angles for these stations
        and map the station locations to what you'd get 
        for an LFE at a reference depth

        Parameters
        ----------
        refdepth :
            reference depth in km (default: 30.)
        """

        # initialize velocity model
        from obspy import taup
        mdl=taup.tau.TauPyModel(model='iasp91')

        # save arrival info
        self.arrivals={}
        self.takeoff_angles={}
        self.arrival_times={}
        self.phases={}
        self.statdstm={}
        self.statxym={}

        # for mapping, grab first little s arrival
        self.grid_takeoff_angles(refdepth=refdepth,plot=False)
        grid_tkang=np.array([vl[0] for vl in self.tkang_litS])
        ix=np.argsort(grid_tkang)
        
        for stn in self.statloc.keys():
            # distance in degrees for this station
            dst=obspy.geodetics.base.kilometers2degrees(self.statdst[stn])

            # find arrivals
            arvl=mdl.get_travel_times(source_depth_in_km=self.floc[2],
                                      distance_in_degree=dst,
                                      phase_list=['s','S'])
            self.arrivals[stn]=arvl
            self.takeoff_angles[stn]=np.array([arv.takeoff_angle for arv in arvl])
            self.arrival_times[stn]=np.array([arv.time for arv in arvl])
            self.phases[stn]=np.array([arv.name for arv in arvl])

            # find upward-going arrivals
            arvl=mdl.get_travel_times(source_depth_in_km=self.floc[2],
                                      distance_in_degree=dst,
                                      phase_list=['s'])

            # and the distance that gives this angle
            dnew=np.interp(arvl[0].takeoff_angle,grid_tkang[ix],self.dst_grid[ix])
            self.statdstm[stn]=dnew
            self.statxym[stn]=self.statxy[stn]*(dnew/self.statdst[stn])

        
    
    #-----END STATION INFORMATION AND ORIENTATION-------

    #-----BEGIN SCALING---------------------------------

    def compute_scalings(self,wlen=None):
        """
        compute scalings between the various stacks and the total stack
        
        Parameters
        ----------
        wlen :
           window to consider in seconds relative to pick
            (default: self.wlen)
        """

        # window to consider
        if wlen is not None:
            self.wlen=np.atleast_1d(wlen)
        self.scaling_wlen=self.wlen.copy()

        # identify each station with a stack
        stns=np.unique([tr.stats.station for tr in self.totstk])

        # intialize scalings
        grps=self.groups.keys()
        scl=dict([(ky,{}) for ky in grps])
        sclb=dict([(ky,{}) for ky in grps])
        sclc=dict([(ky,{}) for ky in grps])
        sclcb=dict([(ky,{}) for ky in grps])

        for stn in stns:
            # for each station with E, N, Z
            stt_pos=self.totstk.select(station=stn)
            stt=stt_pos.select(channel='E')
            stt=stt+stt_pos.select(channel='N')
            stt=stt+stt_pos.select(channel='Z')
            if len(stt)==3:
                # also rotate to radial and transverse
                # here Y is radial, X is transverse
                stt_rot=self.project_waveforms(stt,stns=[stn],
                                               y_azimuths=self.stataz[stn])
                
                # find portion of the data to extract
                tr=stt[0]
                tms=tr.times()-tr.stats.t0
                ii=np.logical_and(tms>=self.wlen[0],tms<self.wlen[1])

                for ky in grps:
                    # seismograms for this group
                    stg=self.grpstk[ky].select(station=stn)
                    stg_rot=self.project_waveforms(stg,stns=[stn],
                                                   y_azimuths=self.stataz[stn])

                    # keep track of radial and transverse value
                    t_radb,t_transb=0.,0.
                    g_radb,g_transb=0.,0.
                    cs=np.cos(self.stataz[stn]*np.pi/180),
                    sn=np.sin(self.stataz[stn]*np.pi/180)
                    
                    if len(stg)==3:
                        # have the data, so initialize computation
                        sm,nml=0.,0.
                        smb,nmlb=np.zeros(self.Nboot,dtype=float),np.zeros(self.Nboot,dtype=float)
                        sclc[ky][stn]={}
                        sclcb[ky][stn]={}
                        
                        for trt in stt:
                            # find data for the same component
                            datat=trt.data[ii]
                            trg=stg.select(channel=trt.stats.channel)[0]
                            datag=trg.data[ii]

                            # the bootstrapped total stack
                            idi='.'.join([stn,trt.stats.channel])
                            datatb=self.totstkb[idi][ii,:]
                            datagb=self.grpstkb[ky][idi][ii,:]
                            
                            # unnormalized x-c
                            smi=np.dot(datat,datag)
                            sm=sm+smi
                            smbi=np.sum(np.multiply(datatb,datagb),axis=0)
                            smb=smb+smbi

                            # normalize by total in template
                            nmli=np.dot(datat,datat)
                            nmlbi=np.sum(np.multiply(datatb,datatb),axis=0)
                            nml=nml+nmli
                            nmlb=nmlb+nmlbi

                            # save per-component scaling
                            sclc[ky][stn][trt.stats.channel]=smi/nmli
                            sclcb[ky][stn][trt.stats.channel]=\
                                np.divide(smbi,nmlbi)

                            # add any contributions to radial and transverse
                            # transverse positive to the right along path
                            if trt.stats.channel=='E':
                                t_radb,t_transb=t_radb+sn*datatb,t_transb+cs*datatb
                                g_radb,g_transb=g_radb+sn*datagb,g_transb+cs*datagb
                            elif trt.stats.channel=='N':
                                t_radb,t_transb=t_radb+cs*datatb,t_transb-sn*datatb
                                g_radb,g_transb=g_radb+cs*datagb,g_transb-sn*datagb

                        # save this scaling
                        scl[ky][stn]=sm/nml
                        sclb[ky][stn]=np.divide(smb,nmlb)

                        # also compute the scalings for rotated components
                        t_rad=stt_rot.select(channel='Y*')[0][ii]
                        t_trans=stt_rot.select(channel='X*')[0][ii]
                        g_rad=stg_rot.select(channel='Y*')[0][ii]
                        g_trans=stg_rot.select(channel='X*')[0][ii]

                        #---for radial---------------------------
                        datat,datag,datatb,datagb=t_rad,g_rad,t_radb,g_radb
                        chn='R'

                        # unnormalized x-c
                        smi=np.dot(datat,datag)
                        smbi=np.sum(np.multiply(datatb,datagb),axis=0)
                        
                        # normalize by total in template
                        nmli=np.dot(datat,datat)
                        nmlbi=np.sum(np.multiply(datatb,datatb),axis=0)
                        
                        # save per-component scaling
                        sclc[ky][stn][chn]=smi/nmli
                        sclcb[ky][stn][chn]=\
                            np.divide(smbi,nmlbi)

                        #---for transverse---------------------------
                        datat,datag,datatb,datagb=t_trans,g_trans,t_transb,g_transb
                        chn='T'

                        # unnormalized x-c
                        smi=np.dot(datat,datag)
                        smbi=np.sum(np.multiply(datatb,datagb),axis=0)
                        
                        # normalize by total in template
                        nmli=np.dot(datat,datat)
                        nmlbi=np.sum(np.multiply(datatb,datatb),axis=0)
                        
                        # save per-component scaling
                        sclc[ky][stn][chn]=smi/nmli
                        sclcb[ky][stn][chn]=\
                            np.divide(smbi,nmlbi)


                        # divide by the average scaling
                        for chn in sclc[ky][stn].keys():
                            sclc[ky][stn][chn]=sclc[ky][stn][chn] / \
                                scl[ky][stn]
                            sclcb[ky][stn][chn]=np.divide(sclcb[ky][stn][chn], \
                                                          sclb[ky][stn])

                        

        self.scalings=scl
        self.scalingsb=sclb
        self.scalingsc=sclc
        self.scalingscb=sclcb



    def normalize_radtrans_scaling(self):
        """
        normalize the scalings on the radial and transverse so 
        that the total power is 2
        """

        scl=self.scalingsc
        for ky in scl.keys():
            for stn in scl[ky].keys():
                # grab values
                R=scl[ky][stn]['R']
                T=scl[ky][stn]['T']

                # normalize
                mlt=np.sqrt(2/(np.power(R,2)+np.power(T,2)))
                R,T=R*mlt,T*mlt

                # reset
                scl[ky][stn]['R']=R
                scl[ky][stn]['T']=T

        scl=self.scalingscb
        for ky in scl.keys():
            for stn in scl[ky].keys():
                # grab values
                R=scl[ky][stn]['R']
                T=scl[ky][stn]['T']

                # normalize
                mlt=np.power(0.5*(np.power(R,2)+np.power(T,2)),-0.5)
                R,T=np.multiply(R,mlt),np.multiply(T,mlt)

                # reset
                scl[ky][stn]['R']=R
                scl[ky][stn]['T']=T

    def plot_scaling_by_location(self,cplot=['E','N','Z'],adjusted_distances=False,
                                 plot_background=False,grps=['late - early'],
                                 minstat=3,width=None):
        """
        plot scalings from averaged stacks to differenced stacks
        
        Parameters
        ----------
        cplot :
            which component or components to plot: from 'E','N','Z','T'
        adjusted_distances :
            adjust the distances to a reference depth? 
               (default: False, just use normal distances)
        plot_background : 
            plot the spatial average in the background 
               (default: True)
        minstat : 
            minimum number of stations to average over?
        width :
            width of plot, in km
        """

        # groups
        if isinstance(grps,str):
            grps=[grps]
        grps=np.atleast_1d(grps)
        Ng=len(grps)

        acmps=['E','N','Z','R','T']
        cplot=np.atleast_1d(cplot)
        Np=cplot.shape[0]
        
        f=plt.figure(figsize=(Np*3+1,Ng*3+1))
        gs,p=gridspec.GridSpec(Ng,Np),[]
        gs.update(left=0.1,right=0.85,bottom=0.2,top=0.92)
        gs.update(hspace=0.1,wspace=0.1)
        for k in range(0,Np*Ng):
            p.append(plt.subplot(gs[k]))
        p=np.array(p)
        pm=p.reshape([Ng,Np]).T

        gs2,p2=gridspec.GridSpec(Ng,1),[]
        gs2.update(left=0.87,right=0.9,bottom=0.2,top=0.9)
        for k in range(0,Ng):
            p2.append(plt.subplot(gs2[k]))
        
        # percentages
        prc=np.array([0.15,0.85])
        iprc=(prc*self.Nboot).astype(int)

        if plot_background:
            mksize=5
        else:
            mksize=35
            
        for kg in range(0,len(grps)):
            grp=grps[kg]
            if grp in self.scalingsc.keys():
                data1=self.scalingsc[grp]
                data1b=self.scalingscb[grp]
                data2,data2b=0.,0.
                usediff=0.
            else:
                usediff=1.
                grp1=grp.split(' - ')
                grp1,grp2=grp1[0],grp1[1]
                data1=self.scalingsc[grp1]
                data1b=self.scalingscb[grp1]
                data2=self.scalingsc[grp2]
                data2b=self.scalingscb[grp2]
            
            # find all stations
            kys=np.array(list(data1.keys()))
            stns=np.unique([ky.split('.')[0] for ky in kys])

            # create a matrix with all the energies
            enmat=np.ndarray([len(stns),len(acmps)],dtype=float)
            enmatb=np.ndarray([len(stns),len(acmps),self.Nboot],dtype=float)

            # note locations
            if adjusted_distances:
                locs=np.vstack([self.statxym[stn] for stn in stns])
            else:
                locs=np.vstack([self.statxy[stn] for stn in stns])
            
            for ks in range(0,len(stns)):
                ky=stns[ks]
                vli=np.array([data1[ky][chn] for chn in acmps])
                vlbi=np.vstack([data1b[ky][chn] for chn in acmps])
                if data2:
                    vli=vli-np.array([data2[ky][chn] for chn in acmps])
                    vlbi=vlbi-np.vstack([data2b[ky][chn] for chn in acmps])
                enmat[ks,:]=vli
                enmatb[ks,:,:]=vlbi.reshape([1,5,self.Nboot])

            # extract mean of horizontal
            mn=np.mean(enmat[:,0:2],axis=1,keepdims=True)
            enmat=np.append(enmat,mn,axis=1)
            mn=np.mean(enmatb[:,0:2,:],axis=1,keepdims=True)
            enmatb=np.append(enmatb,mn,axis=1)
            bcmps=np.append(acmps,'H')

            # sort bootstraps and extract percentiles
            enmatb.sort(axis=2)
            enmat1=enmatb[:,:,iprc[0]]
            enmat2=enmatb[:,:,iprc[1]]

            # location bounds
            if width is None:
                xd=np.max(np.abs(locs))*1.1
            else:
                xd=width/2.

            # limits on colorbar
            bd=np.median(np.abs(enmat-(1-usediff)))*2

            # use a local median?
            usemedian=False
            
            # note the components
            for k in range(0,Np):
                i1=np.where(bcmps==cplot[k])[0][0]

                #uncertainties
                xerr=np.vstack([enmat[:,i1]-enmat1[:,i1],
                                enmat2[:,i1]-enmat[:,i1]])
                xerr=np.maximum(xerr,0)
                
                # make plots
                if cplot[k]=='A':
                    bdi,cmap,clbl=90.,'twilight','azimuth'
                else:
                    lmap={'R':'radial','T':'transverse'}
                    lmap=lmap.get(bcmps[i1],bcmps[i1])
                    bdi,cmap,clbl=bd,'RdBu_r','change in {:s} component'
                    clbl=clbl.format(lmap)
                    clbl=clbl+'\n'+r'$\leftarrow$less later'.ljust(30)
                    clbl=clbl+r'more later$\rightarrow$'.rjust(30)

                # compute median values?
                if usemedian:
                    vl=local_medians(locs[:,0:2],enmat[:,i1],N=usemedian)
                else:
                    vl=enmat[:,i1]
                    
                h=pm[k,kg].scatter(x=locs[:,0],y=locs[:,1],c=vl,
                                   marker='o',vmin=(1-usediff)-bdi,
                                   vmax=(1-usediff)+bdi,cmap=cmap,
                                   edgecolor='gray',s=mksize)

                if plot_background and kg<len(grps):
                    ibk=np.where(self.spatscmps==cplot[k])[0][0]
                    enmap=self.spatscl[grp][:,:,ibk]
                    enmap[self.spat_Nstat<minstat]=float('nan')
                    x,y=np.unique(self.spat_grid[0]),np.unique(self.spat_grid[1])
                    pm[k,kg].pcolormesh(x,y,enmap,zorder=0,cmap=cmap,vmin=(1-usediff)-bdi,
                                        vmax=(1-usediff)+bdi)

                if kg==len(grps)-1:
                    pm[k,kg].set_xlabel('distance E (km)')
                else:
                    pm[k,kg].set_xticklabels('')
                if k==0:
                    pm[k,kg].set_ylabel('distance N (km)')
                else:
                    pm[k,kg].set_yticklabels('')
                pm[k,kg].set_aspect('equal')
                pm[k,kg].set_xlim(np.array([-1,1])*xd)
                pm[k,kg].set_ylim(np.array([-1,1])*xd)
                pm[k,kg].axvline(0,color='k',linestyle='--',zorder=0)
                pm[k,kg].axhline(0,color='k',linestyle='--',zorder=0)
                if kg==0:
                    pm[k,kg].set_title(bcmps[i1])

            cb=plt.colorbar(h,cax=p2[kg])
            cb.set_label(clbl)

            
        # plot reference distances
        yslip=np.array([-1,1])*xd*2
        xslip=yslip*np.tan(self.slipdir*np.pi/180)
        ystrike=np.array([-1,1])*xd*2
        xstrike=yslip*np.tan((90+self.slipdir)*np.pi/180)

        thet=np.linspace(0,np.pi*2,1000)
        x,y=np.cos(thet),np.sin(thet)
        for ph in pm.flatten():
            ph.plot(xslip,yslip,color='k',linestyle='--',linewidth=0.5)
            ph.plot(xstrike,ystrike,color='k',linestyle='--',linewidth=0.5)

            for rd in self.ref_distances:
                ph.plot(x*rd,y*rd,color='k',linewidth=0.5,linestyle='--')


    def plot_binned_coefficients(self,chn='R',avetype='median',group1='early',group2='late'):
        """
        Parameters
        ----------
        chn : 
             which channel to plot (default: 'R')
        avetype : 
             how to average: 'median' (default) or 'mean'
        group1 :
             name of the 1st/reference group (default: 'early')
        group2 : 
             name of the 2nd group (default: 'late')
        """

        lms=np.array([[135,180],[112,135],[80,112]])

        Np=lms.shape[0]
        f=plt.figure(figsize=(Np*3+1,2*3+1))
        gs,p=gridspec.GridSpec(Np,2),[]
        gs.update(left=0.1,right=0.9,bottom=0.2,top=0.92)
        gs.update(hspace=0.1,wspace=0.05)
        for k in range(0,Np*2):
            p.append(plt.subplot(gs[k]))
        p=np.array(p)
        pm=p.reshape([Np,2])
        fs='large'

        # find usable takeoff angles
        kys=np.array(list(self.scalingsc[group1].keys()))
        tkg=np.array([self.takeoff_angles[ky][0] for ky in kys])

        bns=np.linspace(-.3,.3,30)
        bbns=np.linspace(-.03,.03,30)

        if avetype=='mean':
            avefun=np.mean
        elif avetype=='median':
            avefun=np.median
            
        lmap={'R':'radial','T':'transverse'}
        lmap=lmap.get(chn,chn)
        for k in range(0,Np):
            # which values to use
            ix=np.logical_and(tkg>=lms[k,0],tkg<lms[k,1])
            stns=kys[ix]
            
            # scalings
            scl=np.array([self.scalingsc[group2][stn][chn]-
                          self.scalingsc[group1][stn][chn]
                          for stn in stns])
            sclb=[self.scalingscb[group2][stn][chn]-
                  self.scalingscb[group1][stn][chn]
                  for stn in stns]
            scls=np.array([np.std(vl) for vl in sclb])
            scl=scl[scls<0.05]

            # median
            mdn=avefun(scl)
            median_by_station=True
            if median_by_station:
                bmdn=[avefun(scl[np.random.choice(scl.size,scl.size,replace=True)])
                      for k in range(0,1000)]
                bmdn=np.array(bmdn)
            else:
                bmdn=avefun(np.array(sclb),axis=0)
            
            # how many are negative
            frcneg=np.sum(bmdn<0)/np.sum(bmdn<float('inf'))
            neglbl='{:0.0f}% < 0'.format(frcneg*100)

            # mostly positive or negative
            pm[k,0].hist(scl,bins=bns)
            pm[k,1].hist(bmdn,bins=bbns)
            pm[k,0].set_xlim([bns[0],bns[-1]])
            pm[k,1].set_xlim([bbns[0],bbns[-1]])
            if k<Np-1:
                pm[k,1].set_xticklabels('')
                pm[k,0].set_xticklabels('')

            lbl=r'takeoff: {:0.0f} - {:0.0f} degrees'.format(lms[k,0],lms[k,1])
            pm[k,1].text(1,1,lbl,transform=pm[k,0].transAxes,verticalalignment='top',
                         horizontalalignment='center',backgroundcolor='w',zorder=10,
                         fontsize=fs)
            pm[k,1].text(0.95,0.95,neglbl,horizontalalignment='right',
                         backgroundcolor='w',fontsize=fs,
                         transform=pm[k,1].transAxes,verticalalignment='top')
            pm[k,0].xaxis.grid('on',linestyle='--',zorder=0)
            pm[k,1].xaxis.grid('on',linestyle='--',zorder=0)

            pm[k,1].axvline(mdn,color='k',linestyle='-')
            pm[k,0].axvline(mdn,color='k',linestyle='-')
            
        imid=int(Np/2)
        pm[imid,0].set_ylabel('number of observations',fontsize=fs)
        pm[imid,1].set_ylabel('number of bootstrapped '+avetype+'s',fontsize=fs)

        clbl='change in {:s} component'.format(lmap)
        clbl=clbl+'\n'+r'$\leftarrow$less later'.ljust(40)
        clbl=clbl+r'more later$\rightarrow$'.rjust(40)
        pm[-1,0].set_xlabel(clbl,fontsize=fs)
        pm[-1,1].set_xlabel(clbl,fontsize=fs)
        for ph in pm[:,1]:
            ph.yaxis.tick_right()
            ph.yaxis.set_label_position('right')


                
        
    #-----END SCALING-----------------------------------


    #-----BEGIN X-C FOR MOMENT ESTIMATION---------------


    def xc_with_stack(self,stack=None,xcwin=[0,4],max_shift=0.1,
                      olddur=0.5,newdur=None,stns=None):
        """
        cross correlate recorded data with a stack 
        to determine time shifts, amplitude, or duration
        
        Parameters
        ----------
        stack : 
             the stack to use (default: self.totstk)
        xcwin : 
             time window to use for x-c (default: [0,4])
        max_shift :
             maximum time shift to allow, in s (default: 0.1)
        olddur : 
             an old template duration to consider
        newdur : 
             a new template duration, if of interest
             (default: olddur, does nothing)
        stns : 
             which stations to consider (default: all of them)
        """

        # defaults
        if stack is None:
            stack=self.totstk
        xcwin=np.atleast_1d(xcwin)
        self.xcwin=xcwin
        if newdur is None:
            newdur = olddur
        
        # how many events
        Nev=np.max(np.hstack(list(self.ev.values())))+1

        # the time shifts in grid points
        nshf=int(np.round(max_shift*self.sampling_rate))
        ishf=np.arange(-nshf,nshf+1)
        Nshf=ishf.size

        # the stations
        if stns is None:
            stns=np.unique([tr.stats.station for tr in stack])
        else:
            stns=np.atleast_1d(stns)
        self.xc_stns=stns
        Nstat=stns.size

        # and components
        cmps=np.unique([tr.stats.channel for tr in stack])
        cmps=np.array(['E','N','Z'])
        self.xc_cmps=cmps
        Ncmps=cmps.size
        
        # create a grid of results
        self.xc=np.ndarray([Nshf,Nev,Nstat,Ncmps],dtype=float)*float('nan')
        self.nml2=np.ndarray([Nshf,Nev,Nstat,Ncmps],dtype=float)*float('nan')
        self.nml1=np.ndarray([1,1,Nstat,Ncmps],dtype=float)*float('nan')

        # add a buffer
        tbuf=2/self.sampling_rate

        # number of points in the grid
        Nt=int(np.round(np.diff(xcwin)[0]*self.sampling_rate))

        # map keys from data
        dmap={'1':'E','2':'N','Z':'Z',
              'E':'E','N':'N'}
        kymap={}
        for ky in self.data.keys():
            stn,chn=ky.split('.')
            kymap['.'.join([stn,dmap.get(chn[-1])])]=ky
        
        for ks in range(0,len(stns)):
            stn=stns[ks]
            # normalization already applied to this station
            data_norm=self.data_norm[stn]
            
            for kc in range(0,len(cmps)):
                cmp=cmps[kc]
                ky=kymap.get('.'.join([stn,cmp]))
                tr=stack.select(station=stn,channel=cmp)
                
                if len(tr) and ky in self.data.keys():
                    # grab the relevant template
                    tr=tr[0].copy()
                    i1=np.argmin(np.abs(tr.times()-(tr.stats.t0+xcwin[0])))
                    i2=i1+Nt
                    if olddur==newdur:
                        # just grab the data
                        data1=tr.data[i1:i2]
                    else:
                        # buffer first
                        nbuf=int(1*self.sampling_rate)
                        data1=tr.data[(i1-nbuf):(i2+nbuf)]
                        # modify the template
                        data1,trash,trash=self.modify_template(data1,olddur=olddur,newdur=newdur)
                        # and grab the middle
                        data1=data1[nbuf:-nbuf]

                    # normalization for template
                    nml1=np.sqrt(np.sum(np.power(data1,2)))

                    # and the target data
                    data2=self.data[ky][(i1+ishf[0]):(i2+ishf[-1]+1),:]

                    # the events here
                    iev=self.ev[ky]

                    # x-c
                    xci=np.ndarray([Nshf,iev.size],dtype=float)
                    for k in range(0,Nshf):
                        xci[k,:]=np.dot(data1,data2[k:(k+Nt),:])

                    # normalize
                    nml2=np.cumsum(np.power(data2,2),axis=0)
                    nml2=np.append(np.zeros([1,nml2.shape[1]]),nml2,axis=0)
                    nml2=nml2[np.arange(Nt,Nt+Nshf),:]-nml2[np.arange(0,0+Nshf),:]
                    nml2=np.power(nml2,0.5)

                    # and accomodate normalization from input seismograms
                    xci=np.multiply(xci,data_norm.reshape([1,data_norm.size]))
                    nml2=np.multiply(nml2,data_norm.reshape([1,data_norm.size]))
                    
                    # save unnormalized x-c
                    self.xc[:,iev,ks,kc]=xci

                    # and save the normalizations
                    self.nml1[0,0,ks,kc]=nml1
                    self.nml2[:,iev,ks,kc]=nml2


        # normalize to get relative amplitudes
        nml1=np.nansum(np.power(self.nml1,2),axis=3,keepdims=True)
        self.xcamp=np.divide(self.xc,nml1)

        # and fully normalized x-c
        nml1=np.power(nml1,0.5)
        nml2=np.nansum(np.power(self.nml2,2),axis=3,keepdims=True)
        nml2=np.power(nml2,0.5)
        self.xc=np.divide(self.xc,np.multiply(nml1,nml2))

        # time shifts
        self.xctim=ishf/self.sampling_rate

        # pick the best-fitting times
        xcbystat=np.nanmean(self.xc,axis=3)
        xcmean=np.nanmean(xcbystat,axis=2)

        # maxima
        xcmax=np.max(xcmean,axis=0)
        imax=np.argmax(xcmean,axis=0)
        self.txcmax=self.xctim[imax]
        self.txcmax[np.isnan(xcmax)]=float('nan')
        self.xcmax=xcmax

        # moment
        amps=np.mean(self.xcamp,axis=3)
        amps=np.array([amps[imax[k],k,:] for k in range(0,imax.size)])
        self.allamps=amps

    def normalize_amplitudes(self):
        """
        reorganize amplitudes to get one relative moment per event
        and one relative amplitude per station
        """

        # initialize station amplitude to median
        statamp=np.nanmedian(self.allamps,axis=0,keepdims=True)
        df=[1000]

        ctr=0
        while ctr<100 and np.max(np.abs(df))>0.0001:
            ctr=ctr+1
            statamp0=statamp

            # event amplitudes are the medians after removing
            # the station corrections
            evamp=np.nanmedian(np.divide(self.allamps,statamp),axis=1,
                               keepdims=True)

            # but keep average median as 1
            evamp=evamp/np.nanmedian(evamp)

            # new station amplitudes are medians after removing event
            # corrections
            statamp=np.nanmedian(np.divide(self.allamps,evamp),axis=0,
                                 keepdims=True)

            # percentage change from previous station averages
            df=np.divide(statamp-statamp0,statamp)

        self.statamp=statamp.flatten()
        self.evamp=evamp.flatten()

        
    #-----END X-C FOR MOMENT ESTIMATION-----------------
        
    #-----BEGIN FOR COMPARING DURATIONS-----------------

    def modify_template(self,tdata=None,olddur=0.5,newdur=1.0):
        """
        change the duration of the template

        Parameters
        ----------
        tdata :
            template data 
            (default: a Hann function with length 0.5 s)
        olddur : 
            old duration in seconds (default: 0.5)
            can be None, in which case there will be no deconvolution
        newdur :
            new duration in seconds (default: 1.)

        Returns
        -------
        mdata : 
            modified template data
        flt : 
            the frequency-domain filter applied
        freq :
            the frequencies for that filter
        """

        # template data if not given
        if tdata is None:
            nw = int(np.round(0.3*self.sampling_rate))
            na = int(np.round(3.*self.sampling_rate))
            tdata = np.hstack([np.zeros(na),np.hanning(nw),np.zeros(na)])

        # to buffer the FFT
        Nt=tdata.size*5

        # how much noise to add as a fraction of the maximum
        noise_amp=0.001
        
        # old stf
        if olddur is not None:
            n_old = int(np.round(olddur*self.sampling_rate/2))*2
            odata=np.hanning(n_old)
            odata=odata/np.sum(odata)
            odata=np.append(odata,np.zeros(Nt-odata.size))

            # move so the stfs are centred at zero
            odata=np.roll(odata,-int(n_old/2))

            # add some noise
            odata=odata+np.random.randn(odata.size)*np.max(odata)*noise_amp
            self.old_stf=odata.copy()
            
            # FFT
            odata=np.fft.rfft(odata,n=Nt)

        # new stf
        n_new = int(np.round(newdur*self.sampling_rate/2))*2
        ndata=np.hanning(n_new)
        ndata=ndata/np.sum(ndata)
        ndata=np.append(ndata,np.zeros(Nt-ndata.size))

        # move so the stfs are centred at zero
        ndata=np.roll(ndata,-int(n_new/2))

        # add some noise
        ndata=ndata+np.random.randn(ndata.size)*np.max(ndata)*noise_amp
        self.new_stf=ndata.copy()

        # FFT
        ndata=np.fft.rfft(ndata,n=Nt)
        mdata=np.fft.rfft(tdata,n=Nt)

        # frequencies and filter
        freq=np.fft.rfftfreq(n=Nt,d=1/self.sampling_rate)
        flt=ndata
        if olddur is not None:
            ii=odata!=0
            flt[ii]=np.divide(ndata[ii],odata[ii])
            flt[~ii]=0.

        # times the template and back to time domain
        mdata=np.multiply(mdata,flt)
        mdata=np.fft.irfft(mdata)
        mdata=mdata[0:tdata.size]

        return mdata,flt,freq

    def split_stations(self,prc_classify=0.75,minsnr=10):
        """
        split the stations into two groups,
        one for duration classification and one for later analysis
        
        Parameters
        ----------
        prc_classify :
            what percentage of the stations to use for classification
            though must leave at least two for analysis
        minsnr :
            minimum signal to noise ratio to classify stations, in power
        """

        # grab signal to noise power ratio
        snr=self.stack_snrs

        # which stations to use
        stns=np.array(list(snr.keys()))
        iok=np.array([snr[stn] > minsnr for stn in stns])
        stns=stns[iok]

        # number for classification
        Nc=int(stns.size*prc_classify)
        Nc=np.minimum(Nc,stns.size-2)

        # and pick
        self.cl_stns=np.random.choice(stns,Nc)
        self.an_stns=np.array(list(set(stns)-set(self.cl_stns)))

    
    def duration_search(self,stack=None,xcwin=[0.,4],max_shift=0.15,
                        olddurs=[0.4,0.5],
                        newdurs=np.arange(0.3,1.3,0.2)):
        """
        compute cross-correlations with a variety of durations
         
        Parameters
        ----------
        stack : 
             the stack to use (default: self.totstk)
        xcwin : 
             time window to use for x-c (default: [0,4])
        max_shift :
             maximum time shift to allow, in s (default: 0.1)
        olddurs : 
             template durations to consider
        newdurs :
             proposed LFE durations to consider
             But doesn't allow new durations shorter than old ones
        """

        # defaults
        if stack is None:
            stack=self.totstk
        xcwin=np.atleast_1d(xcwin)

        # initialize
        olddurs=np.atleast_1d(olddurs)
        newdurs=np.atleast_1d(newdurs)
        self.olddurs=olddurs
        self.newdurs=newdurs

        # the stations
        stns=np.unique([tr.stats.station for tr in stack])
        stns=self.an_stns
        Nstat=stns.size
        
        # how many events
        Nev=np.max(np.hstack(list(self.ev.values())))+1
        self.xc_bydur=np.ndarray([Nev,self.olddurs.size,
                                  self.newdurs.size],dtype=float)*float('nan')
        self.allamps_bydur=np.ndarray([Nev,Nstat,self.olddurs.size,self.newdurs.size],
                                      dtype=float)*float('nan')

        for ko in range(0,olddurs.size):
            for kn in range(0,newdurs.size):
                # if the relative values can work, search
                if newdurs[kn]>=olddurs[ko]:
                    print('Old duration', olddurs[ko], 'New duration', newdurs[kn])

                    # x-c to get best-fitting duration from classification stations
                    self.xc_with_stack(stack=stack,xcwin=xcwin,max_shift=max_shift,
                                       olddur=olddurs[ko],newdur=newdurs[kn],
                                       stns=self.cl_stns)

                    # save max x-c values
                    self.xc_bydur[:,ko,kn]=self.xcmax.copy()

                    # x-c to get best-fitting amplitudes from analysis stations
                    self.xc_with_stack(stack=stack,xcwin=xcwin,max_shift=max_shift,
                                       olddur=olddurs[ko],newdur=newdurs[kn],
                                       stns=self.an_stns)

                    # save relative amplitudes too
                    self.allamps_bydur[:,:,ko,kn]=self.allamps.copy()


        isn=np.isnan(self.xc_bydur)
        self.xc_bydur[isn]=-float('inf')

        # pick the max x-c by duration
        xcm=np.nanmax(self.xc_bydur,axis=2)
        iold=np.argmax(xcm,axis=1)

        xcm=np.nanmax(self.xc_bydur,axis=1)
        inew=np.nanargmax(xcm,axis=1)

        # best-fitting x-c
        self.xc_best=np.nanmax(xcm,axis=1)
        self.olddur=self.olddurs[iold]
        self.newdur=self.newdurs[inew]

        self.xc_bydur[isn]=float('nan')

        # and grab amplitudes
        self.allamps_bdur=np.array([self.allamps_bydur[k,:,iold[k],inew[k]]
                                       for k in range(0,self.allamps.shape[0])])

        # get relative event and station amplitudes
        self.allamps=self.allamps_bdur
        self.normalize_amplitudes()
        self.evamp_bdur=self.evamp.copy()
        self.statamp_bdur=self.statamp.copy()
        

        # and a final x-c
        self.xc_with_stack(stack=stack,xcwin=xcwin,max_shift=max_shift)
        self.normalize_amplitudes()

    
    #-----END FOR COMPARING DURATIONS-------------------
    
# Jean added on 14/8/2024
#-----BEGIN RATIO OF SCALINGS OF LATE/EARLY EVENTS------------
    def Plot_Ratio_of_Late_and_Early(self,fnums=np.array([1,12,142,144,156,191,22,23,246,256,3,30,31,49,52,53,55,61,62,65,66,7,70,74],dtype=int)):
         # loop over
          fnums=np.unique(fnums)
          for fnum in fnums:
              # as before, we need to initialize an analysis object
              lf=self.lfanalyse(fnum=fnum)
          
              # and load in the detections and waveforms
              lf.load_prep_data(flm=[1,8],single_norm=False)
    
              # stack for each group
              lf.stack_by_group()

              # and pick arrival times
              lf.pick_stacks(minsnr=10)
    
              # compute scalings
              lf.relative_station_locations()
              lf.compute_scalings(wlen=[0,4])
    
              # let's go ahead and normalize the radial and transverse scalings for all stations
              lf.normalize_radtrans_scaling()
    
              # pick some stations for duration classification
              lf.split_stations(prc_classify=0.75)
    
              # and estimate best-fitting amplitudes and durations
              olddurs=np.array([0.2,0.3])
              newdurs=np.arange(0.1,0.61,0.1)
              lf.duration_search(max_shift=0.2,olddurs=olddurs,newdurs=newdurs)
    
              # save the results
              lf.save_results()
        
          # initialize and collect some results
          lf2=self.lfanalyse(fnum=1)

          # collect the energy calculations
          lf2.collect_energies(fnums=fnums)

          latescales=lf2.scalings['late']
          earlyscales=lf2.scalings['early']
          # compute ratios of late/early stacks
          ratioScales={key: latescales[key] / earlyscales.get(key, 0)
                       for key in latescales.keys()}
          # compute the median of late/early stacks ratios
          ratioScalesValues=list(ratioScales.values())
          ratioScalesValues=np.array(ratioScalesValues)
          medianRatios=np.median(ratioScalesValues)
          print(medianRatios)
          lists = ratioScales.items() # sorted by key, return a list of tuples
          x, y = zip(*lists) # unpack a list of pairs into two tuples
          x=list(x)
          xfam=[]
          for xx in x:
              xx=xx.split('-')
              xfam.append(xx[0])
        
          # plot the ratios of early and late stacks of each station
          plt.scatter(xfam, ratioScales.values())
          plt.xlabel("Families") 
          plt.ylabel("ratios of late and early stacks")
          plt.hlines(y=medianRatios,xmin=-1, xmax=25, colors='r', label='median')
          plt.legend()
          plt.show()
    

#-----END RATIO OF SCALINGS OF LATE/EARLY EVENTS------------

    
#-----BEGIN STANDALONE ARRIVAL TIME AND SNR ANALYSIS----------
    
def compute_snrs_and_picks(st,minsnr=20,wlen=5.):
    """
    Parameters
    ----------
    st :
        template seismograms
    minsnr : 
        minimum signal to noise (power) ratio to allow (default: 20s)
    wlen :
        window length for snr calculation (default: 5s)
        
    Returns
    -------
    snrs :
        signal to noise ratios
    st :
        templates with picks
    stns :
        stations with good snr
    """
    
    # signal to noise ratio
    snrs={}

    # identify the strongest power on the horizontal component
    stns=np.unique([tr.stats.station for tr in st])
    for stn in stns:
        sti=st.select(station=stn)
        nvl=int(wlen/sti[0].stats.delta)
    
        # shift
        nshf=int(nvl*1.2)
    
        # power on the horizontal
        pwr=np.power(sti.select(channel='E')[0].data,2)+\
            np.power(sti.select(channel='N')[0].data,2)
        pwr=np.cumsum(pwr)
        pwr=pwr[nvl:]-pwr[:-nvl]
        
        # maximum power
        imax=np.argmax(pwr[nshf:])
        
        # signal and noise power
        spow=pwr[nshf+imax]
        npow=pwr[imax]
        if npow==0:
            npow=float('nan')
    
        # start time
        tst=(imax+nshf-nvl*0.1)*sti[0].stats.delta
    
        snrs[stn]=spow/npow
    
        # set the onset times and signal to noise ratio
        for tr in sti:
            tr.stats.t0=tst
            tr.stats.t1=tst+wlen
            tr.stats.snr=spow/npow
        
    # select stations
    snrsv=np.array(list(snrs.values()))
    istat=snrsv>=minsnr
    stns=np.array(list(snrs.keys()))[istat]
    stns=np.array([stn.strip() for stn in stns])

    return snrs,st,stns    
    
#-----END STANDALONE ARRIVAL TIME AND SNR ANALYSIS------------

