import numpy as np
from scipy import signal
import obspy
import matplotlib.pyplot as plt


class lfeanalyse:

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


    def discard_data_with_nans(self):
        """
        discard any time intervals that include NaNs
        """

        kys=np.array(list(self.data.keys()))
        stns=np.array([ky.split('.')[0] for ky in kys])
        astns=np.unique(stns)

        for stn in astns:
            # which keys are relevant here
            ix=stns==stn
            kysh=kys[ix]

            # grab the data from all channels
            data=np.stack([self.data[ky] for ky in kysh],axis=2)

            # identify the ones with no gaps
            iok=np.sum(np.sum(np.isnan(data),axis=0),axis=1)==0

            # and grab the the okay data
            for ky in kysh:
                self.data[ky]=self.data[ky][:,iok]
                self.ev[ky]=self.ev[ky][iok]


            # change the normalization if needed
            if 'data_norm' in self.__dict__.keys():
                self.data_norm[stn]=self.data_norm[stn][iok]
            if 'data_max' in self.__dict__.keys():
                self.data_max[stn]=self.data_max[stn][iok]
            
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
                    mx=np.ones(mx.shape)*np.nanmedian(mx.flatten())
                    
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
        elif stype in ['all','total']:
            st=self.totstk
            wlen=[0,self.ssnr_wlen]
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
                sti,trash=self.project_waveforms(sti,stns=stn[ks],y_azimuths=azm)
                for tr in sti.select(channel='X*'):
                    tr.stats.channel='T'
                for tr in sti.select(channel='Y*'):
                    tr.stats.channel='R'

                for k in range(0,len(stm)):
                    stm[k],trash=self.project_waveforms(stm[k],stns=stn,y_azimuths=azm)
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
    
