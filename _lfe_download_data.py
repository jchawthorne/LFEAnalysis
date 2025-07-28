import numpy as np
import obspy
import os
import pickle
import glob
from . import seisproc


#-----BEGIN DATA DOWNLOAD AND SAVING----------------
class lfeanalyse:

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



    def download_data(self,max_events=float('inf'),
                      pre_filt=(1/15, 1/5, float('inf'),float('inf')),
                      water_level=60):
        """
        retrieve data from IRIS via an FDSN downloader

        Parameters
        ----------
        max_events :
            the maximum number of data intervals to download, 
              usually used if testing a new area
              (default: float('inf')---all of them)
        pre_filt ;
            the pre-filter given to the response removal
              (default: (1/15,1/5,Inf,Inf))

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
        ids=ids.astype(str)
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
                    tr.detrend()
                    tr.remove_response(output='VEL',pre_filt=pre_filt,
                                       water_level=water_level)
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

    def clear_seismic_data(self):
        """
        delete all the files with downloaded seismic data
        """

        # the directory with data
        fdir=self.directory()

        # remove all files inside this directory
        for fname in glob.glob(os.path.join(fdir,'*')):
            os.remove(fname)

    
    #-----END DATA DOWNLOAD AND SAVING------------------
