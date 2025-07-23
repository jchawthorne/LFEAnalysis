import numpy as np
import obspy
import matplotlib.pyplot as plt
from matplotlib import gridspec

class lfeanalyse:
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
