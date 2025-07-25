import numpy as np
import pickle
import os


#-----BEGIN SAVING AND RELOADING--------------------

class object_tosave:
    """
    an object that just contains a bunch of data parameters
    to be used for saving the results
    """

    def __init__(self,lfi):
        """
        Parameters
        ----------
        lfi :
            the object with data to save
            all the parameters in lfi.__dict__ will be copied to
            the new object
        """

        for ky in lfi.__dict__.keys():
            self.__setattr__(ky,lfi.__getattribute__(ky))


class lfeanalyse:

    def collect_results(self,fnums=[41]):
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
                 'arrival_times','stataz','statdst','phases']

        # try adding 'scalings' to the list above
        

        
        #subadd=['diffen','diffenb','diffpol','diffpolb','scalingsc','scalingscb']
        subadd=['scalingsc','scalingscb','scalings','gxc']
        listadd=['stacked_velocity_reduction']
        listadd=[]
        
        # initialize an object
        from . import lfeanalyse as lfeobject
        lfi=lfeobject(lfi=self)
        
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
                    if ky in lfi.__dict__.keys():
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
                    if dname in lfi.__dict__.keys():
                        dct1=self.__getattribute__(dname)
                        dct2=lfi.__getattribute__(dname)
                        for grp in dct1.keys():
                            dct1[grp].update(dct2[grp])
                for dname in listadd:
                    if dname in lfi.__dict__.keys():
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
        if 'scalingsc' in self.__dict__.keys():
            dcts=dcts+list(self.scalingsc.values())+list(self.scalingscb.values())
        if 'gxc' in self.__dict__.keys():
            dcts=dcts+list(self.gxc.values())
        
        # station info
        dcts=dcts+[self.statxy,self.statloc]
        if 'scalings' in self.__dict__.keys():
            dcts=dcts+list(self.scalings.values())
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

        # make a copy
        lfi=object_tosave(lfi=self)

        # remove the data from the copy
        if 'data' in lfi.__dict__.keys():
            lfi.__delattr__('data')

        # where to save
        fname=os.path.join(self.directory(),'Results')

        # save
        with open(fname, "wb") as fl:
            pickle.dump(lfi, fl)




    
    #-----END SAVING AND RELOADING----------------------
