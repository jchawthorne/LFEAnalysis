import numpy as np
import obspy

class lfeanalyse:

    
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

