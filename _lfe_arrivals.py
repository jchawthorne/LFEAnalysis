import numpy as np
import obspy

#-----BEGIN ARRIVAL TIME AND SNR ANALYSIS----------

class lfeanalyse:
    
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
            self.ssnr_wlen=np.atleast_1d(wlen)
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

    
    def pick_templates(self,wlen=5,minsnr=20):
        """
        pick arrival times on the template and compute snr
        
        Parameters
        ----------
        wlen :
             window length to use for computation, in seconds
                    (default: 5)
        minsnr : 
             minimum signal to noise ratio to accept stacks
                    in power (default: 20)
        """

        # select picks on templates and compute snr
        self.temp_snrs,self.sttemp,self.stns =\
            compute_snrs_and_picks(self.sttemp,minsnr=minsnr,
                                   wlen=wlen)

        # save for later
        self.tsnr_wlen=wlen
        self.tminsnr=minsnr

    #-----END ARRIVAL TIME AND SNR ANALYSIS------------


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
    
        snrs[str(stn)]=float(spow/npow)
    
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
