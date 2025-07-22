import numpy as np
import os,glob
import pickle

class lfeanalyse:

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
