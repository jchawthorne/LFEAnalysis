import numpy as np
import obspy
import matplotlib.pyplot as plt
from matplotlib import gridspec

class lfeanalyse:
        #-----BEGIN SCALING---------------------------------

    def compute_scalings(self,wlen=None):
        """
        compute scalings between the various stacks and the total stack
        
        Parameters
        ----------
        wlen :
           window to consider in seconds relative to pick
            (default: self.wlen)
        """

        # window to consider
        if wlen is not None:
            self.wlen=np.atleast_1d(wlen)
        self.scaling_wlen=self.wlen.copy()

        # identify each station with a stack
        stns=np.unique([tr.stats.station for tr in self.totstk])

        # intialize scalings
        grps=self.groups.keys()
        scl=dict([(ky,{}) for ky in grps])
        sclb=dict([(ky,{}) for ky in grps])
        sclc=dict([(ky,{}) for ky in grps])
        sclcb=dict([(ky,{}) for ky in grps])

        for stn in stns:
            # for each station with E, N, Z
            stt_pos=self.totstk.select(station=stn)
            stt=stt_pos.select(channel='E')
            stt=stt+stt_pos.select(channel='N')
            stt=stt+stt_pos.select(channel='Z')
            if len(stt)==3:
                # also rotate to radial and transverse
                # here Y is radial, X is transverse
                stt_rot,trash=self.project_waveforms(stt,stns=[stn],
                                                     y_azimuths=self.stataz[stn])
                
                # find portion of the data to extract
                tr=stt[0]
                tms=tr.times()-tr.stats.t0
                ii=np.logical_and(tms>=self.wlen[0],tms<self.wlen[1])

                for ky in grps:
                    # seismograms for this group
                    stg=self.grpstk[ky].select(station=stn)
                    stg_rot,trash=self.project_waveforms(stg,stns=[stn],
                                                y_azimuths=self.stataz[stn])

                    # keep track of radial and transverse value
                    t_radb,t_transb=0.,0.
                    g_radb,g_transb=0.,0.
                    cs=np.cos(self.stataz[stn]*np.pi/180),
                    sn=np.sin(self.stataz[stn]*np.pi/180)
                    
                    if len(stg)==3:
                        # have the data, so initialize computation
                        sm,nml=0.,0.
                        smb,nmlb=np.zeros(self.Nboot,dtype=float),np.zeros(self.Nboot,dtype=float)
                        sclc[ky][stn]={}
                        sclcb[ky][stn]={}
                        
                        for trt in stt:
                            # find data for the same component
                            datat=trt.data[ii]
                            trg=stg.select(channel=trt.stats.channel)[0]
                            datag=trg.data[ii]

                            # the bootstrapped total stack
                            idi='.'.join([stn,trt.stats.channel])
                            datatb=self.totstkb[idi][ii,:]
                            datagb=self.grpstkb[ky][idi][ii,:]
                            
                            # unnormalized x-c
                            smi=np.dot(datat,datag)
                            sm=sm+smi
                            smbi=np.sum(np.multiply(datatb,datagb),axis=0)
                            smb=smb+smbi

                            # normalize by total in template
                            nmli=np.dot(datat,datat)
                            nmlbi=np.sum(np.multiply(datatb,datatb),axis=0)
                            nml=nml+nmli
                            nmlb=nmlb+nmlbi

                            # save per-component scaling
                            sclc[ky][stn][trt.stats.channel]=smi/nmli
                            sclcb[ky][stn][trt.stats.channel]=\
                                np.divide(smbi,nmlbi)

                            # add any contributions to radial and transverse
                            # transverse positive to the right along path
                            if trt.stats.channel=='E':
                                t_radb,t_transb=t_radb+sn*datatb,t_transb+cs*datatb
                                g_radb,g_transb=g_radb+sn*datagb,g_transb+cs*datagb
                            elif trt.stats.channel=='N':
                                t_radb,t_transb=t_radb+cs*datatb,t_transb-sn*datatb
                                g_radb,g_transb=g_radb+cs*datagb,g_transb-sn*datagb

                        # save this scaling
                        scl[ky][stn]=sm/nml
                        sclb[ky][stn]=np.divide(smb,nmlb)

                        # also compute the scalings for rotated components
                        t_rad=stt_rot.select(channel='Y*')[0][ii]
                        t_trans=stt_rot.select(channel='X*')[0][ii]
                        g_rad=stg_rot.select(channel='Y*')[0][ii]
                        g_trans=stg_rot.select(channel='X*')[0][ii]

                        #---for radial---------------------------
                        datat,datag,datatb,datagb=t_rad,g_rad,t_radb,g_radb
                        chn='R'

                        # unnormalized x-c
                        smi=np.dot(datat,datag)
                        smbi=np.sum(np.multiply(datatb,datagb),axis=0)
                        
                        # normalize by total in template
                        nmli=np.dot(datat,datat)
                        nmlbi=np.sum(np.multiply(datatb,datatb),axis=0)
                        
                        # save per-component scaling
                        sclc[ky][stn][chn]=smi/nmli
                        sclcb[ky][stn][chn]=\
                            np.divide(smbi,nmlbi)

                        #---for transverse---------------------------
                        datat,datag,datatb,datagb=t_trans,g_trans,t_transb,g_transb
                        chn='T'

                        # unnormalized x-c
                        smi=np.dot(datat,datag)
                        smbi=np.sum(np.multiply(datatb,datagb),axis=0)
                        
                        # normalize by total in template
                        nmli=np.dot(datat,datat)
                        nmlbi=np.sum(np.multiply(datatb,datatb),axis=0)
                        
                        # save per-component scaling
                        sclc[ky][stn][chn]=smi/nmli
                        sclcb[ky][stn][chn]=\
                            np.divide(smbi,nmlbi)


                        # divide by the average scaling
                        for chn in sclc[ky][stn].keys():
                            sclc[ky][stn][chn]=sclc[ky][stn][chn] / \
                                scl[ky][stn]
                            sclcb[ky][stn][chn]=np.divide(sclcb[ky][stn][chn], \
                                                          sclb[ky][stn])

                        

        self.scalings=scl
        self.scalingsb=sclb
        self.scalingsc=sclc
        self.scalingscb=sclcb



    def normalize_radtrans_scaling(self):
        """
        normalize the scalings on the radial and transverse so 
        that the total power is 2
        """

        scl=self.scalingsc
        for ky in scl.keys():
            for stn in scl[ky].keys():
                # grab values
                R=scl[ky][stn]['R']
                T=scl[ky][stn]['T']

                # normalize
                mlt=np.sqrt(2/(np.power(R,2)+np.power(T,2)))
                R,T=R*mlt,T*mlt

                # reset
                scl[ky][stn]['R']=R
                scl[ky][stn]['T']=T

        scl=self.scalingscb
        for ky in scl.keys():
            for stn in scl[ky].keys():
                # grab values
                R=scl[ky][stn]['R']
                T=scl[ky][stn]['T']

                # normalize
                mlt=np.power(0.5*(np.power(R,2)+np.power(T,2)),-0.5)
                R,T=np.multiply(R,mlt),np.multiply(T,mlt)

                # reset
                scl[ky][stn]['R']=R
                scl[ky][stn]['T']=T

    def plot_scaling_by_location(self,cplot=['E','N','Z'],adjusted_distances=False,
                                 plot_background=False,grps=['late - early'],
                                 minstat=3,width=None):
        """
        plot scalings from averaged stacks to differenced stacks
        
        Parameters
        ----------
        cplot :
            which component or components to plot: from 'E','N','Z','T'
        adjusted_distances :
            adjust the distances to a reference depth? 
               (default: False, just use normal distances)
        plot_background : 
            plot the spatial average in the background 
               (default: True)
        minstat : 
            minimum number of stations to average over?
        width :
            width of plot, in km
        """

        # groups
        if isinstance(grps,str):
            grps=[grps]
        grps=np.atleast_1d(grps)
        Ng=len(grps)

        acmps=['E','N','Z','R','T']
        cplot=np.atleast_1d(cplot)
        Np=cplot.shape[0]
        
        f=plt.figure(figsize=(Np*3+1,Ng*3+1))
        gs,p=gridspec.GridSpec(Ng,Np),[]
        gs.update(left=0.1,right=0.85,bottom=0.2,top=0.92)
        gs.update(hspace=0.1,wspace=0.1)
        for k in range(0,Np*Ng):
            p.append(plt.subplot(gs[k]))
        p=np.array(p)
        pm=p.reshape([Ng,Np]).T

        gs2,p2=gridspec.GridSpec(Ng,1),[]
        gs2.update(left=0.87,right=0.9,bottom=0.2,top=0.9)
        for k in range(0,Ng):
            p2.append(plt.subplot(gs2[k]))
        
        # percentages
        prc=np.array([0.15,0.85])
        iprc=(prc*self.Nboot).astype(int)

        if plot_background:
            mksize=5
        else:
            mksize=35
            
        for kg in range(0,len(grps)):
            grp=grps[kg]
            if grp in self.scalingsc.keys():
                data1=self.scalingsc[grp]
                data1b=self.scalingscb[grp]
                data2,data2b=0.,0.
                usediff=0.
            else:
                usediff=1.
                grp1=grp.split(' - ')
                grp1,grp2=grp1[0],grp1[1]
                data1=self.scalingsc[grp1]
                data1b=self.scalingscb[grp1]
                data2=self.scalingsc[grp2]
                data2b=self.scalingscb[grp2]
            
            # find all stations
            kys=np.array(list(data1.keys()))
            stns=np.unique([ky.split('.')[0] for ky in kys])

            # create a matrix with all the energies
            enmat=np.ndarray([len(stns),len(acmps)],dtype=float)
            enmatb=np.ndarray([len(stns),len(acmps),self.Nboot],dtype=float)

            # note locations
            if adjusted_distances:
                locs=np.vstack([self.statxym[stn] for stn in stns])
            else:
                locs=np.vstack([self.statxy[stn] for stn in stns])
            
            for ks in range(0,len(stns)):
                ky=stns[ks]
                vli=np.array([data1[ky][chn] for chn in acmps])
                vlbi=np.vstack([data1b[ky][chn] for chn in acmps])
                if data2:
                    vli=vli-np.array([data2[ky][chn] for chn in acmps])
                    vlbi=vlbi-np.vstack([data2b[ky][chn] for chn in acmps])
                enmat[ks,:]=vli
                enmatb[ks,:,:]=vlbi.reshape([1,5,self.Nboot])

            # extract mean of horizontal
            mn=np.mean(enmat[:,0:2],axis=1,keepdims=True)
            enmat=np.append(enmat,mn,axis=1)
            mn=np.mean(enmatb[:,0:2,:],axis=1,keepdims=True)
            enmatb=np.append(enmatb,mn,axis=1)
            bcmps=np.append(acmps,'H')

            # sort bootstraps and extract percentiles
            enmatb.sort(axis=2)
            enmat1=enmatb[:,:,iprc[0]]
            enmat2=enmatb[:,:,iprc[1]]

            # location bounds
            if width is None:
                xd=np.max(np.abs(locs))*1.1
            else:
                xd=width/2.

            # limits on colorbar
            bd=np.median(np.abs(enmat-(1-usediff)))*2

            # use a local median?
            usemedian=False
            
            # note the components
            for k in range(0,Np):
                i1=np.where(bcmps==cplot[k])[0][0]

                #uncertainties
                xerr=np.vstack([enmat[:,i1]-enmat1[:,i1],
                                enmat2[:,i1]-enmat[:,i1]])
                xerr=np.maximum(xerr,0)
                
                # make plots
                if cplot[k]=='A':
                    bdi,cmap,clbl=90.,'twilight','azimuth'
                else:
                    lmap={'R':'radial','T':'transverse'}
                    lmap=lmap.get(bcmps[i1],bcmps[i1])
                    bdi,cmap,clbl=bd,'RdBu_r','change in {:s} component'
                    if len(cplot)>1:
                        clbl='change in scaling factor'
                    if usediff:
                        clbl=clbl.format(lmap)
                        clbl=clbl+'\n'+(r'$\leftarrow$more '+grp2).ljust(30)
                        clbl=clbl+('more '+grp1+r' $\rightarrow$').rjust(30)
                    else:
                        clbl='{:s} scaling factor'.format(grp)

                # compute median values?
                if usemedian:
                    vl=local_medians(locs[:,0:2],enmat[:,i1],N=usemedian)
                else:
                    vl=enmat[:,i1]
                    
                h=pm[k,kg].scatter(x=locs[:,0],y=locs[:,1],c=vl,
                                   marker='o',vmin=(1-usediff)-bdi,
                                   vmax=(1-usediff)+bdi,cmap=cmap,
                                   edgecolor='gray',s=mksize)

                if plot_background and kg<len(grps):
                    ibk=np.where(self.spatscmps==cplot[k])[0][0]
                    enmap=self.spatscl[grp][:,:,ibk]
                    enmap[self.spat_Nstat<minstat]=float('nan')
                    x,y=np.unique(self.spat_grid[0]),np.unique(self.spat_grid[1])
                    pm[k,kg].pcolormesh(x,y,enmap,zorder=0,cmap=cmap,vmin=(1-usediff)-bdi,
                                        vmax=(1-usediff)+bdi)

                if kg==len(grps)-1:
                    pm[k,kg].set_xlabel('distance E (km)')
                else:
                    pm[k,kg].set_xticklabels('')
                if k==0:
                    pm[k,kg].set_ylabel('distance N (km)')
                else:
                    pm[k,kg].set_yticklabels('')
                pm[k,kg].set_aspect('equal')
                pm[k,kg].set_xlim(np.array([-1,1])*xd)
                pm[k,kg].set_ylim(np.array([-1,1])*xd)
                pm[k,kg].axvline(0,color='k',linestyle='--',zorder=0)
                pm[k,kg].axhline(0,color='k',linestyle='--',zorder=0)
                if kg==0:
                    pm[k,kg].set_title(bcmps[i1])

            cb=plt.colorbar(h,cax=p2[kg])
            cb.set_label(clbl)

            
        # plot reference distances
        yslip=np.array([-1,1])*xd*2
        xslip=yslip*np.tan(self.slipdir*np.pi/180)
        ystrike=np.array([-1,1])*xd*2
        xstrike=yslip*np.tan((90+self.slipdir)*np.pi/180)

        thet=np.linspace(0,np.pi*2,1000)
        x,y=np.cos(thet),np.sin(thet)
        for ph in pm.flatten():
            ph.plot(xslip,yslip,color='k',linestyle='--',linewidth=0.5)
            ph.plot(xstrike,ystrike,color='k',linestyle='--',linewidth=0.5)

            for rd in self.ref_distances:
                ph.plot(x*rd,y*rd,color='k',linewidth=0.5,linestyle='--')


    def plot_radcoeff_ratios(self):
        """
        plot the SH and SV ratios against each other
        """

        p=plt.axes()
        stats=self.radcoeff_ratios['SH'].keys()

        sh=np.array([self.radcoeff_ratios['SH'][stn] for stn in stats])
        sv=np.array([self.radcoeff_ratios['SV'][stn] for stn in stats])

        p.plot(sh,sv,color='k',marker='x',linestyle='none');
        p.set_xlabel('SH coefficient ratio');
        p.set_ylabel('SV coefficient ratio');
        p.set_xlim([-8,8])
        p.set_ylim([-8,8])
        p.axvline(0,zorder=0,linestyle=':',color='k')
        p.axhline(0,zorder=0,linestyle=':',color='k')
        
    def bin_stations_by_radcoeff(self,svlms=[-10,0,10],shlms=[-10,10]):
        """
        divide the stations according the the radiation coefficient ratios

        Parameters
        ----------
        svlms :
            bounds for the SV coefficient ratio
        shlms :
            bounds for the SH coefficient ratio
        """

        svlms=np.atleast_1d(svlms)
        shlms=np.atleast_1d(shlms)
        Nv,Nh=svlms.size-1,shlms.size-1
        iv,ih=np.meshgrid(np.arange(0,Nv),np.arange(0,Nh))
        iv,ih=iv.flatten(),ih.flatten()

        statlist,binlabel=[],[]
        
        for k in range(0,iv.size):

            # the label
            vlbl='SV ratio: {:0.1f} - {:0.1f}'.format(svlms[iv[k]],svlms[iv[k]+1])
            hlbl='SH ratio: {:0.1f} - {:0.1f}'.format(shlms[ih[k]],shlms[ih[k]+1])
            binlabel.append(', '.join([vlbl,hlbl]))

            # go through stations
            stats=self.radcoeff_ratios['SH'].keys()
            statlist.append([])

            for stn in stats:
                vok=self.radcoeff_ratios['SV'][stn]>=svlms[iv[k]] and \
                    self.radcoeff_ratios['SV'][stn]<svlms[iv[k]+1]
                hok=self.radcoeff_ratios['SH'][stn]>=shlms[ih[k]] and \
                    self.radcoeff_ratios['SH'][stn]<shlms[ih[k]+1]

                if vok and hok:
                    statlist[k].append(stn)

        return statlist,binlabel

                
    def bin_stations_by_takeoff(self,lms=[[0,180],[135,180],[112,135],[80,112]]):
        """
        divide the stations into groups according to takeoff angle

        Parameters
        ----------
        lms :
           list of takeoff ranges or 2-D array of takeoff angles
        
        Returns
        -------
        stnlists :
           lists of stations for each bin
        binlabel :
           label for each bin
        """

        # initialize station list and labels
        statlist=[]
        binlabel=[]

        # make a 2-D array
        lms=np.atleast_2d(lms)
        
        for k in range(0,lms.shape[0]):
            # initialize station list for this range
            statlist.append([])
            
            # find usable takeoff angles
            kys=self.takeoff_angles.keys()
            for stn in kys:
                tkg=self.takeoff_angles[stn][0]
                if tkg>=lms[k,0] and tkg<lms[k,1]:
                    statlist[k].append(stn)

            # create a label
            lbl=r'takeoff: {:0.0f} - {:0.0f} degrees'.format(lms[k,0],lms[k,1])
            binlabel.append(lbl)

        return statlist,binlabel

    def plot_binned_coefficients(self,chn='R',avetype='median',group1='early',group2='late',statlist=None,binlabel=None):
        """
        Parameters
        ----------
        chn : 
             which channel to plot (default: 'R')
        avetype : 
             how to average: 'median' (default) or 'mean'
        group1 :
             name of the 1st/reference group (default: 'early')
        group2 : 
             name of the 2nd group (default: 'late')
        """


        # find the groups of stations to plot
        if statlist is None or binlabel is None:
            statlist,binlabel=self.bin_stations_by_takeoff()
        

        Np=len(statlist)
        f=plt.figure(figsize=(Np*3+1,2*3+1))
        gs,p=gridspec.GridSpec(Np,2),[]
        gs.update(left=0.1,right=0.9,bottom=0.2,top=0.92)
        gs.update(hspace=0.1,wspace=0.05)
        for k in range(0,Np*2):
            p.append(plt.subplot(gs[k]))
        p=np.array(p)
        pm=p.reshape([Np,2])
        fs='large'

        
        bns=np.linspace(-.3,.3,30)
        bbns=np.linspace(-.03,.03,30)

        bns=np.linspace(-.5,.5,30)
        bbns=np.linspace(-.05,.05,30)

        
        if avetype=='mean':
            avefun=np.mean
        elif avetype=='median':
            avefun=np.median
            
        lmap={'R':'radial','T':'transverse'}
        lmap=lmap.get(chn,chn)
        for k in range(0,Np):
            # which values to use

            stns=statlist[k]
            
            # scalings
            scl,sclb=[],[]
            for stn in stns:
                if stn in self.scalingsc[group2].keys():
                    scl.append(self.scalingsc[group2][stn][chn]-
                               self.scalingsc[group1][stn][chn])
                    sclb.append(self.scalingscb[group2][stn][chn]-
                                self.scalingscb[group1][stn][chn])
            scl=np.array(scl)
            sclb=np.array(sclb)
            scls=np.array([np.std(vl) for vl in sclb])
            scl=scl[scls<0.05]

            # median
            mdn=avefun(scl)
            median_by_station=True
            if median_by_station:
                bmdn=[avefun(scl[np.random.choice(scl.size,scl.size,replace=True)])
                      for k in range(0,1000)]
                bmdn=np.array(bmdn)
            else:
                bmdn=avefun(np.array(sclb),axis=0)

            # how many are negative
            frcneg=np.sum(bmdn<0)/np.sum(bmdn<float('inf'))
            neglbl='{:0.0f}% < 0'.format(frcneg*100)

            # mostly positive or negative
            pm[k,0].hist(scl,bins=bns)
            pm[k,1].hist(bmdn,bins=bbns)
            pm[k,0].set_xlim([bns[0],bns[-1]])
            pm[k,1].set_xlim([bbns[0],bbns[-1]])
            if k<Np-1:
                pm[k,1].set_xticklabels('')
                pm[k,0].set_xticklabels('')

            lbl=binlabel[k]
            pm[k,1].text(1,1,lbl,transform=pm[k,0].transAxes,verticalalignment='top',
                         horizontalalignment='center',backgroundcolor='w',zorder=10,
                         fontsize=fs)
            pm[k,1].text(0.95,0.95,neglbl,horizontalalignment='right',
                         backgroundcolor='w',fontsize=fs,
                         transform=pm[k,1].transAxes,verticalalignment='top')
            pm[k,0].xaxis.grid('on',linestyle='--',zorder=0)
            pm[k,1].xaxis.grid('on',linestyle='--',zorder=0)

            pm[k,1].axvline(mdn,color='k',linestyle='-')
            pm[k,0].axvline(mdn,color='k',linestyle='-')
            
        imid=int(Np/2)
        pm[imid,0].set_ylabel('number of observations',fontsize=fs)
        pm[imid,1].set_ylabel('number of bootstrapped '+avetype+'s',fontsize=fs)

        clbl='change in {:s} component'.format(lmap)
        clbl=clbl+'\n'+r'$\leftarrow$less later'.ljust(40)
        clbl=clbl+r'more later$\rightarrow$'.rjust(40)
        pm[-1,0].set_xlabel(clbl,fontsize=fs)
        pm[-1,1].set_xlabel(clbl,fontsize=fs)
        for ph in pm[:,1]:
            ph.yaxis.tick_right()
            ph.yaxis.set_label_position('right')

                
        
    #-----END SCALING-----------------------------------

