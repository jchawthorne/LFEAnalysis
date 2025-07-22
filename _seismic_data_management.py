import numpy as np
import os,glob
import pickle
import obspy

class lfeanalyse:

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
