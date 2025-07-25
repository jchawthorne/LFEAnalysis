import numpy as np
from . import radpatterns

class lfeanalyse:

    def calc_radcoeff_ratios(self,phase='weighted',wlen=None):
        """
        calculate the ratio of the opening to shear radiation coefficients

        Parameters
        ----------
        phase :
            which phase to consider (default:'weighted')
               'first': just use the first-arriving wave
               'weighted': weight each arrival by the amount of
                      time each wave is in the window
               
        wlen :
            window length to consider
              (default: self.wlen)
        """

        # create dictionaries of the expected radiation coefficient ratios
        self.radcoeff_ratios={'P':{},'SH':{},'SV':{}}

        # default window length
        if wlen is None and phase in ['weighted']:
            wlen=self.wlen

        # and loop through the stations
        for stn in self.takeoff_angles.keys():
            # initialize
            for phs in self.radcoeff_ratios.keys():
                self.radcoeff_ratios[phs][stn]=float('nan')
            
            if phase in ['first'] and len(self.arrival_times):
                # if we want the phase of the first-arriving wave
                ix=np.argmin(self.arrival_times[stn])
                for phs in self.radcoeff_ratios.keys():
                    self.radcoeff_ratios[phs][stn]=\
                        self.radcoeff_open[phs][stn][ix]/self.radcoeff_shear[phs][stn][ix]

            elif phase in ['weighted'] and len(self.arrival_times):
                # duration of the arrival within the window
                dur=wlen[1]-(self.arrival_times[stn]-np.min(self.arrival_times[stn]))
                dur=np.maximum(dur,0.)
                dur[np.isnan(dur)]=0.

                # weight by these durations
                for phs in self.radcoeff_ratios.keys():
                    self.radcoeff_ratios[phs][stn]=\
                        np.dot(dur,self.radcoeff_open[phs][stn]) / \
                        np.dot(dur,self.radcoeff_shear[phs][stn]) 
        
        
    
    def calc_radiation_coefficients(self,strike=315,dip=20,rake=90):
        """
        calculate the radiation coefficients expected for the
        P, SH, and SV waves
        for both shear slip and fault opening

        Parameters
        ----------
        strike :
            strike of the fault, in degrees east of north
        dip :
            dip of the fault, in degrees from the horizontal
        rake :
            direction of slip
        """

        # create a moment tensor for shear slip
        Mshear=radpatterns.create_moment_tensor(strike=strike,dip=dip,rake=rake)
        # and a moment tensor for opening
        Mopen=radpatterns.create_moment_tensor(strike=strike,dip=dip,rake=rake,
                                               opening_angle=90.)

        # create dictionaries of the expected radiation coefficients
        self.radcoeff_shear={'P':{},'SH':{},'SV':{}}
        self.radcoeff_open={'P':{},'SH':{},'SV':{}}

        # and loop through the stations
        for stn in self.takeoff_angles.keys():
            # note the azimuth
            azm=self.stataz[stn]

            # and the takeoff angles
            tkoff=self.takeoff_angles[stn]

            # initialize coefficient array for this station
            for wv in self.radcoeff_shear.keys():
                self.radcoeff_shear[wv][stn]=np.ndarray(len(tkoff),dtype=float)
                self.radcoeff_open[wv][stn]=np.ndarray(len(tkoff),dtype=float)

            # and compute for each takeoff angle
            for k in range(0,len(tkoff)):
                rd_p,rd_sh,rd_sv = \
                    radpatterns.calc_radiation_pattern(Mshear,takeoff_angle=tkoff[k],azimuth=azm)
                self.radcoeff_shear['P'][stn][k]=rd_p
                self.radcoeff_shear['SH'][stn][k]=rd_sh
                self.radcoeff_shear['SV'][stn][k]=rd_sv
            
                rd_p,rd_sh,rd_sv = \
                    radpatterns.calc_radiation_pattern(Mopen,takeoff_angle=tkoff[k],azimuth=azm)
                self.radcoeff_open['P'][stn][k]=rd_p
                self.radcoeff_open['SH'][stn][k]=rd_sh
                self.radcoeff_open['SV'][stn][k]=rd_sv
