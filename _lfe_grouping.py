import numpy as np
import obspy
import matplotlib.pyplot as plt
from matplotlib import gridspec
from obspy import UTCDateTime

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



    ## --- Added for Guerrero --- ##

    # -- Detect Bursts -- #
    # Three options: by elevated LFE rate, by running window coverage, by interevent time
           
    def detect_lfe_rate_bursts(self, bin_duration=7200,
                            bin_spacing=None,
                            start_rate=0.5, end_rate=0.05):
        """
        Detect events based on elevated LFE rate.
        Uses compute_lfe_rates

        Returns
        -------
        dict with:
            n_bursts : int
            start_times : np.ndarray
            end_times : np.ndarray
        """

        # Create one big window covering all data
        self.sse_tlms = np.array([[self.tms.min(), self.tms.max()]])
        self.Nsse = 1

        # Compute rate
        self.compute_lfe_rates(bin_duration=bin_duration, bin_spacing=bin_spacing)

        rates = np.asarray(self.lferate, dtype=float)
        times = np.asarray(self.tcent)

        if rates.size == 0:
            self.start_times = np.array([])
            self.end_times = np.array([])
            self.n_bursts = 0
            return {
                "n_bursts": self.n_bursts,
                "start_times": self.start_times,
                "end_times": self.end_times,
            }

        # start a burst at a high rate, end it only after the rate drops lower.
        baseline = np.nanmax(rates)
        start_threshold = baseline * start_rate
        end_threshold = baseline * end_rate

        start_indices = []
        end_indices = []

        in_burst = False
        current_start = None

        for i, rate in enumerate(rates):
            if (not in_burst) and rate >= start_threshold:
                current_start = i
                in_burst = True
            elif in_burst and rate <= end_threshold:
                start_indices.append(current_start)
                end_indices.append(i)
                in_burst = False
                current_start = None

        # Close any burst that continues until the last bin
        if in_burst and current_start is not None:
            start_indices.append(current_start)
            end_indices.append(len(rates) - 1)

        start_indices = np.asarray(start_indices, dtype=int)
        end_indices = np.asarray(end_indices, dtype=int)

        # Convert bin centers to burst start/end times
        half_window = bin_duration / 2.0
        self.start_times = times[start_indices] - half_window
        self.end_times = times[end_indices] + half_window
        self.n_bursts = len(self.start_times)

        return {
            "n_bursts": self.n_bursts,
            "start_times": self.start_times,
            "end_times": self.end_times,
        }


    def detect_lfe_bursts_coverage(self, detections=None, d=6.0, L=86400.0, dt=60.0):
        """
        Burst detection from running LFE coverage (Frank et al., 2014).
        """
        if detections is None:
            detections = self.tms

        detections = np.asarray(detections)

        if isinstance(detections[0], UTCDateTime):
            detections_sec = np.array(
                [tm.timestamp for tm in detections],
                dtype=float
            )
        else:
            detections_sec = detections.astype(float)

        t0 = detections_sec.min() - L / 2
        t1 = detections_sec.max() + d + L / 2
        times = np.arange(t0, t1 + dt, dt)

        c = np.zeros(len(times), dtype=float)
        for tau in detections_sec:
            c[(times > tau) & (times < tau + d)] = 1.0

        nwin = max(1, int(round(L / dt)))
        kernel = np.ones(nwin) / nwin
        coverage = np.convolve(c, kernel, mode="same")

        threshold = 2 * np.sqrt(np.mean(coverage ** 2))
        burst_mask = coverage > threshold

        edges = np.diff(np.r_[0, burst_mask.astype(int), 0])
        start_idx = np.where(edges == 1)[0]
        end_idx = np.where(edges == -1)[0] - 1

        start_times_sec = times[start_idx]
        end_times_sec = times[end_idx] + dt

        # Convert burst boundaries back to UTCDateTime
        start_times = np.array(
            [UTCDateTime(t) for t in start_times_sec],
            dtype=object
        )
        end_times = np.array(
            [UTCDateTime(t) for t in end_times_sec],
            dtype=object
        )

        # Save to object 
        self.times = times
        self.coverage = coverage
        self.threshold = threshold
        self.start_times = start_times
        self.end_times = end_times


    def plot_lfe_coverage(self):
        """
        Plot running LFE coverage with threshold line.
        Assumes detect_lfe_bursts_coverage() has already been run.
        """
        if not hasattr(self, "coverage"):
            raise ValueError("Run detect_lfe_bursts_coverage() first.")

        # Convert time to something readable (days relative to start)
        t = (self.times - self.times[0]) / 86400.0

        plt.figure(figsize=(18, 4))
        plt.plot(t, self.coverage, label="Running LFE coverage")
        plt.axhline(self.threshold, linestyle="--", label="Threshold")

        plt.xlabel("Time (days)")
        plt.ylabel("Coverage")
        plt.title("Running LFE Coverage and Burst Threshold")
        plt.legend()
        plt.tight_layout()
        plt.show()


    def detect_lfe_bursts_interevent(self, detections=None, start_threshold=0.015, end_threshold=0.015, dmin=15):
        """
        Burst detection based on interevent time (Peng and Rubin 2017). 
        Starts a burst when interevent time drops below threshold, 
        ends it when above threshold, and discards bursts with too few LFEs.
        Allows different start and end thresholds.

        Parameters
        ----------
        detections : array-like
            LFE detection/start times tau_i, in seconds or UTCDateTime-like objects.
        start_threshold : float
            Interevent time threshold for burst start detection, in days
            (default: 0.015 days = 1296 seconds).
        end_threshold : float
            Interevent time threshold for burst end detection, in days
            (default: 0.015 days = 1296 seconds).
        dmin : int
            Minimum number of detections required to form a valid burst
            (default: 15).

        """
        if detections is None:
            detections = self.tms

        detections = np.asarray(detections)

        # Convert absolute times to POSIX seconds for calculation
        if isinstance(detections[0], UTCDateTime):
            detections_sec = np.array(
                [tm.timestamp for tm in detections],
                dtype=float
            )
        else:
            detections_sec = detections.astype(float)

        # Interevent durations are stored in seconds
        self.interevent_times = np.diff(detections_sec)

        # Convert thresholds from days to seconds
        start_threshold_seconds = start_threshold * 86400.0 
        end_threshold_seconds = end_threshold * 86400.0

        bursts = []
        in_burst = False
        burst_start_idx = None

        for i, iet in enumerate(self.interevent_times):
            if not in_burst:
                # Start a burst when the gap becomes short enough
                if iet < start_threshold_seconds:
                    in_burst = True
                    burst_start_idx = i
            else:
                # End the burst when the gap becomes too long
                if iet > end_threshold_seconds:
                    burst_end_idx = i
                    bursts.append((burst_start_idx, burst_end_idx))
                    in_burst = False
                    burst_start_idx = None

        # If still inside a burst at the end, close it at the last detection
        if in_burst:
            bursts.append((burst_start_idx, len(detections_sec) - 1))

        # Convert index pairs to detection times and apply dmin
        valid_bursts = []
        for start_idx, end_idx in bursts:
            start = detections_sec[start_idx]
            end = detections_sec[end_idx]
            count = end_idx - start_idx + 1
            if count >= dmin:
                valid_bursts.append((start, end))

        burst_starts = np.array([UTCDateTime(start) for start, end in valid_bursts], dtype=object)
        burst_ends = np.array([UTCDateTime(end) for start, end in valid_bursts], dtype=object)

        self.start_times = burst_starts
        self.end_times = burst_ends
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold


    def plot_lfe_bursts_interevent(self):
        """
        Plot interevent time and detected bursts. Assumes detect_lfe_bursts_interevent() has already been run.
        """
        if not hasattr(self, "interevent_times"):
            raise ValueError("Run detect_lfe_bursts_interevent() first.")

        # Convert interevent times to days for plotting
        t = (self.tms[1:] - self.tms[0]) / 86400.0
        interevent_times_days = self.interevent_times / 86400.0

        plt.figure(figsize=(18, 4))
        plt.scatter(t, interevent_times_days, label="Interevent Time (days)")
        plt.axhline(self.start_threshold, linestyle="--", label="Start Threshold")

        t0_sec = self.tms[0].timestamp if hasattr(self.tms[0], "timestamp") else float(self.tms[0])

        burst_starts_days = np.array(
            [(UTCDateTime(start).timestamp - t0_sec) / 86400.0 for start in self.start_times],
            dtype=float,
        )
        burst_ends_days = np.array(
            [(UTCDateTime(end).timestamp - t0_sec) / 86400.0 for end in self.end_times],
            dtype=float,
        )

        for start, end in zip(burst_starts_days, burst_ends_days):
            plt.axvspan(start, end, color='gray', alpha=0.3)

        plt.xlabel("Time (days)")
        plt.ylabel("Interevent Time (days)")
        plt.title("Interevent Time and Detected Bursts")
        plt.legend()
        plt.tight_layout()
        plt.show()


    # - Divide events for stacking - #
    # Four options: by time, by detection number, by interevent time, by amplitude
    
    def split_events_by_time(self):
        """Split bursts into early and late groups based on time in burst."""

        # Split each burst in half by time
        self.middle_times = (
            self.start_times + (self.end_times - self.start_times) / 2
        )

        self.early_group_indices = []
        self.late_group_indices = []

        for i, tm in enumerate(self.tms):
            if any(
                (tm >= start) and (tm < mid)
                for start, mid in zip(self.start_times, self.middle_times)
            ):
                self.early_group_indices.append(i)

            elif any(
                (tm >= mid) and (tm <= end)
                for mid, end in zip(self.middle_times, self.end_times)
            ):
                self.late_group_indices.append(i)

        self.groups = {
            "early": np.array(self.early_group_indices, dtype=int),
            "late": np.array(self.late_group_indices, dtype=int),
        }

    
    def split_events_by_detections(self):        
        """Split bursts into early and late groups based on number of detections.
        """
        # find middle detection time for each burst
        self.middle_times = []
        for start, end in zip(self.start_times, self.end_times):
            burst_detections = self.tms[(self.tms >= start) & (self.tms <= end)]
            middle_time = burst_detections[len(burst_detections) // 2]
            self.middle_times.append(middle_time)

        self.early_group_indices = []
        self.late_group_indices = []
        for i, tm in enumerate(self.tms):
            if any((tm >= start) and (tm < mid) for start, mid in zip(self.start_times, self.middle_times)):
                self.early_group_indices.append(i)
            elif any((tm >= mid) and (tm <= end) for mid, end in zip(self.middle_times, self.end_times)):
                self.late_group_indices.append(i)

        self.groups = {
            "early": np.array(self.early_group_indices, dtype=int),
            "late": np.array(self.late_group_indices, dtype=int),
        }


    def split_events_by_interevent(self, threshold=0.05):
        """Split detected bursts into early and late groups based on interevent time. 
           Defines middle time as the first time that interevent time increases above a 
           threshold after the burst start time.
           If that threshold is never met, sets middle by default to the midpoint of the burst.

           Must have already run burst detection by interevent time function (to get self.interevent_times)
        """
        threshold = threshold * 86400  # convert days to seconds
        self.middle_times = []

        for start, end in zip(self.start_times, self.end_times):
            in_burst = (self.tms >= start) & (self.tms <= end)
            burst_times = self.tms[in_burst]
            burst_interevent = self.interevent_times[in_burst[:-1] & in_burst[1:]]  # interevent times between detections, so one less than number of detections

            above = np.where(burst_interevent > threshold)[0] # find interevent times that exceed threshold

            if above.size > 0: # if at least one interevent time exceeds threshold, take the first one as the middle time
                i_mid = int(above[0])
                self.middle_times.append(burst_times[i_mid + 1]) # Takes event after the long gap as middle, which is included in late group
            else:
                # if never increases enough inside burst window, take midpoint
                self.middle_times.append(start + (end - start) / 2)

        self.early_group_indices = []
        self.late_group_indices = []

        for i, tm in enumerate(self.tms):
            if any((tm >= start) and (tm < mid) for start, mid in zip(self.start_times, self.middle_times)):
                self.early_group_indices.append(i)
            elif any((tm >= mid) and (tm <= end) for mid, end in zip(self.middle_times, self.end_times)):
                self.late_group_indices.append(i)

        self.groups = {
            "early": np.array(self.early_group_indices, dtype=int),
            "late": np.array(self.late_group_indices, dtype=int),
        }   

    def split_events_by_amplitude(self):
        """ Split detected bursts into small and big groups based on amplitude. 
            Defines small group as events with amplitude below the median amplitude, 
            and big group as events with amplitude equal and above the median amplitude.

            Must have already computed the event amplitudes."""

        median_amp = np.median(self.evamp)
        self.small_group_indices = []
        self.big_group_indices = []

        for i, amp in enumerate(self.evamp):
            if amp < median_amp:
                self.small_group_indices.append(i)
            else:
                self.big_group_indices.append(i)
        self.groups = {
            "small": np.array(self.small_group_indices, dtype=int),
            "big": np.array(self.big_group_indices, dtype=int),
        }

        



    #-----END EVENT GROUPING AND LFE RATE ANALYSIS-----
