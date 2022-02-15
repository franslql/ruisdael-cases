import numpy as np
from netCDF4 import Dataset
from datetime import datetime

def main():
    # -------------------------- Input -------------------------- #
    # General
    date = '20210924'   # date 'yyyymmdd'
    t0   = 0            # Start time 0-24 hours
    trun = 24           # runtime in hours
    iexpnr = 0          # Experiment number 0-999
    pathRead = '../'    # Top of harmonie data path
    pathWrite = '../input/' # Top of input files path
    # LES grid
    x0 = 917500.0       # x-coordinate southwest corner LES domain
    y0 = 965000.0       # y-coordinate southwest corner LES domain
    itot = 1024         # Number of grid points in x-direction
    jtot = 512          # Number of grid points in y-direction
    ktot = 128          # Number of grid points in z-direction
    xsize = 160000.     # Resolution in x-direction
    ysize = 80000.      # Resolution in y-direction
    stretched = True    # Stretched grid True/False
    dz = 20.            # Stretched grid = False
    dz0 = 25.           # Stretched grid = True, resolution first height level
    alpha = 0.014       # Stretched grid = True, growth factor height levels
    # LES input
    e12   = 0.1         # Turbulent kinetic energy (constant input)
    # Required input
    lprof = False       # Input profiles required True/False
    lboundary = False   # Boundary input required True/False
    linithetero = False # Heterogeneous input fields required True/False
    lsynturb  = False   # Synthetic turbulence input required True/False
    lrad = False        # Input for rrtmg input required True/False
    # DALES constants (modglobal.f90)
    cd = dict(p0=1.e5, Rd=287.04, Rv=461.5, cp=1004., Lv=2.53e6, grav=9.81)
    cd['eps'] = cd['Rv']/cd['Rd']-1.
    # --------------------- Determine path ---------------------- #
    # Don't change if regular path structure is used
    pathRead = pathRead+date[0:4]+'/'+date[4:6]+'/'+date[6:8]+'/00/'
    pathWrite = pathWrite+date[0:4]+'/'+date[4:6]+'/'+date[6:8]+'/00/'
    # ------------------ Construct DALES grid ------------------- #
    grid = Grid(xsize,ysize,itot,jtot,ktot,dz,dz0,alpha,stretched)
    # ------------------- Open Harmonie data -------------------- #
    nch_u  = Dataset(pathRead+'ua.Slev.his.NETHERLANDS.ruisdael_IOP092021.'+date+'.1hr.nc','r')
    nch_v  = Dataset(pathRead+'va.Slev.his.NETHERLANDS.ruisdael_IOP092021.'+date+'.1hr.nc','r')
    nch_w  = Dataset(pathRead+'w.Slev.fp.NETHERLANDS.ruisdael_IOP092021.'+date+'.1hr.nc','r')
    nch_T  = Dataset(pathRead+'ta.Slev.his.NETHERLANDS.ruisdael_IOP092021.'+date+'.1hr.nc','r')
    nch_qt = Dataset(pathRead+'hus.Slev.his.NETHERLANDS.ruisdael_IOP092021.'+date+'.1hr.nc','r')
    nch_ql = Dataset(pathRead+'clw.Slev.his.NETHERLANDS.ruisdael_IOP092021.'+date+'.1hr.nc')
    nch_phi= Dataset(pathRead+'phi.Slev.fp.NETHERLANDS.ruisdael_IOP092021.'+date+'.1hr.nc','r')
    nch_ps = Dataset(pathRead+'ps.his.NETHERLANDS.ruisdael_IOP092021.'+date+'.1hr.nc','r')
    nch_Ts = Dataset(pathRead+'tas.his.NETHERLANDS.ruisdael_IOP092021.'+date+'.1hr.nc','r')
    nch_qts= Dataset(pathRead+'huss.his.NETHERLANDS.ruisdael_IOP092021.'+date+'.1hr.nc','r')
    time   = nch_u.variables['time'][:]; time = (time-time[0])*3600*24
    # Crop data to required domain/time
    it0    = np.argmin(abs(time-t0*3600))
    ite    = np.argmin(abs(time-(t0+trun)*3600))
    Nt     = ite-it0+1
    x      = nch_u.variables['x'][:]
    y      = nch_u.variables['y'][:]
    ix0    = np.argmin(abs(x-x0))
    ixe    = np.argmin(abs(x-(grid.xsize+x0)))
    iy0    = np.argmin(abs(y-y0))
    iye    = np.argmin(abs(y-(grid.ysize+y0)))
    x      = x[ix0:ixe+1]
    y      = y[iy0:iye+1]
    # ------------ Create heterogeneous input fields ------------ #
    if linithetero:
        start_time = datetime.now()
        print("Start creation of initfields.inp."+str(iexpnr).zfill(3)+".nc")
        # Create netcdf file
        heteroInput = HeteroInput(pathWrite,iexpnr,grid)
        # Load fields first time step
        z  = nch_phi.variables['phi'][it0,::-1,iy0:iye+1,ix0:ixe+1]/cd['grav']
        u  = nch_u.variables['ua'][it0,::-1,iy0:iye+1,ix0:ixe+1]
        v  = nch_v.variables['va'][it0,::-1,iy0:iye+1,ix0:ixe+1]
        w  = nch_w.variables['w'][it0,::-1,iy0:iye+1,ix0:ixe+1]
        T  = nch_T.variables['ta'][it0,::-1,iy0:iye+1,ix0:ixe+1]
        qt = nch_qt.variables['hus'][it0,::-1,iy0:iye+1,ix0:ixe+1]
        ql = nch_ql.variables['clw'][it0,::-1,iy0:iye+1,ix0:ixe+1]
        # Surface variables
        zs  = np.zeros(np.shape(z)[1:3])
        us  = np.zeros(np.shape(u)[1:3])
        vs  = np.zeros(np.shape(v)[1:3])
        ws  = np.zeros(np.shape(w)[1:3])
        Ts  = nch_Ts.variables['tas'][it0,iy0:iye+1,ix0:ixe+1]
        qts = nch_qts.variables['huss'][it0,iy0:iye+1,ix0:ixe+1]
        ps  = nch_ps.variables['ps'][it0,iy0:iye+1,ix0:ixe+1]
        # Field conversions HARMONIE -> LES
        p = calcPressure(ps)
        exner  = (p/cd['p0'])**(cd['Rd']/cd['cp'])
        exners = (ps/cd['p0'])**(cd['Rd']/cd['cp'])
        th   = T  / exner
        ths  = Ts / exners
        thl  = th - cd['Lv'] / (cd['cp'] * exner) * ql
        # Concatenate surface fields to 3d fields
        z   = np.concatenate((zs[None,:,:],z),axis=0)
        u   = np.concatenate((us[None,:,:],u),axis=0)
        v   = np.concatenate((vs[None,:,:],v),axis=0)
        w   = np.concatenate((ws[None,:,:],w),axis=0)
        thl = np.concatenate((ths[None,:,:],thl),axis=0)
        qt  = np.concatenate((qts[None,:,:],qt),axis=0)
        # Interpolate fields and write to netcdf
        heteroInput.u[:,:,:]   = interp3d(x,y,z,u,grid.xh+x0,grid.y+y0,grid.z)
        heteroInput.v[:,:,:]   = interp3d(x,y,z,v,grid.x+x0,grid.yh+y0,grid.z)
        heteroInput.w[:,:,:]   = interp3d(x,y,z,w,grid.x+x0,grid.y+y0,grid.zh)
        heteroInput.thl[:,:,:] = interp3d(x,y,z,thl,grid.x+x0,grid.y+y0,grid.z)
        heteroInput.qt[:,:,:]  = interp3d(x,y,z,qt,grid.x+x0,grid.y+y0,grid.z)
        heteroInput.e12[:,:,:] = e12
        # Close netcdf file
        heteroInput.exit()
        del heteroInput
        end_time = datetime.now()
        hours = (end_time-start_time).days*24+(end_time-start_time).seconds//3600
        minutes = ((end_time-start_time).seconds//60)%60
        seconds = ((end_time-start_time).seconds-hours*3600-minutes*60)
        print(f"Finished in {hours:02}:{minutes:02}:{seconds:02}")
    # ------------------ Create boundary input ------------------ #
    if lboundary:
        start_time = datetime.now()
        print("Start creation of openboundaries.inp."+str(iexpnr).zfill(3)+".nc")
        # Create netcdf file
        boundaryInput = BoundaryInput(pathWrite,iexpnr,grid)
        # Write fields per time step
        for it in range(Nt):
            # Load fields for current time step
            z  = nch_phi.variables['phi'][it+it0,::-1,iy0:iye+1,ix0:ixe+1]/cd['grav']
            u  = nch_u.variables['ua'][it+it0,::-1,iy0:iye+1,ix0:ixe+1]
            v  = nch_v.variables['va'][it+it0,::-1,iy0:iye+1,ix0:ixe+1]
            w  = nch_w.variables['w'][it+it0,::-1,iy0:iye+1,ix0:ixe+1]
            T  = nch_T.variables['ta'][it+it0,::-1,iy0:iye+1,ix0:ixe+1]
            qt = nch_qt.variables['hus'][it+it0,::-1,iy0:iye+1,ix0:ixe+1]
            ql = nch_ql.variables['clw'][it+it0,::-1,iy0:iye+1,ix0:ixe+1]
            # Surface variables
            zs  = np.zeros(np.shape(z)[1:3])
            us  = np.zeros(np.shape(u)[1:3])
            vs  = np.zeros(np.shape(v)[1:3])
            ws  = np.zeros(np.shape(w)[1:3])
            Ts  = nch_Ts.variables['tas'][it+it0,iy0:iye+1,ix0:ixe+1]
            qts = nch_qts.variables['huss'][it+it0,iy0:iye+1,ix0:ixe+1]
            ps  = nch_ps.variables['ps'][it+it0,iy0:iye+1,ix0:ixe+1]
            # Field conversions HARMONIE -> LES
            p = calcPressure(ps)
            exner  = (p/cd['p0'])**(cd['Rd']/cd['cp'])
            exners = (ps/cd['p0'])**(cd['Rd']/cd['cp'])
            th   = T  / exner
            ths  = Ts / exners
            thl  = th - cd['Lv'] / (cd['cp'] * exner) * ql
            # Concatenate surface fields to 3d fields
            z   = np.concatenate((zs[None,:,:],z),axis=0)
            u   = np.concatenate((us[None,:,:],u),axis=0)
            v   = np.concatenate((vs[None,:,:],v),axis=0)
            w   = np.concatenate((ws[None,:,:],w),axis=0)
            thl = np.concatenate((ths[None,:,:],thl),axis=0)
            qt  = np.concatenate((qts[None,:,:],qt),axis=0)
            # Interpolate fields and write fields
            boundaryInput.time[it] = time[it+it0]-time[it0]
            # Interpolate west boundary
            boundaryInput.uwest[it,:,:]    = interpLateral(y,z[:,:,0],u[:,:,0],grid.y+y0,grid.z)
            boundaryInput.vwest[it,:,:]    = interpLateral(y,z[:,:,0],v[:,:,0],grid.yh+y0,grid.z)
            boundaryInput.wwest[it,:,:]    = interpLateral(y,z[:,:,0],w[:,:,0],grid.y+y0,grid.zh)
            boundaryInput.thlwest[it,:,:]  = interpLateral(y,z[:,:,0],thl[:,:,0],grid.y+y0,grid.z)
            boundaryInput.qtwest[it,:,:]   = interpLateral(y,z[:,:,0],qt[:,:,0],grid.y+y0,grid.z)
            boundaryInput.e12west[it,:,:]  = e12
            # Interpolate east boundary
            boundaryInput.ueast[it,:,:]    = interpLateral(y,z[:,:,-1],u[:,:,-1],grid.y+y0,grid.z)
            boundaryInput.veast[it,:,:]    = interpLateral(y,z[:,:,-1],v[:,:,-1],grid.yh+y0,grid.z)
            boundaryInput.weast[it,:,:]    = interpLateral(y,z[:,:,-1],w[:,:,-1],grid.y+y0,grid.zh)
            boundaryInput.thleast[it,:,:]  = interpLateral(y,z[:,:,-1],thl[:,:,-1],grid.y+y0,grid.z)
            boundaryInput.qteast[it,:,:]   = interpLateral(y,z[:,:,-1],qt[:,:,-1],grid.y+y0,grid.z)
            boundaryInput.e12east[it,:,:]  = e12
            # Interpolate south boundary
            boundaryInput.usouth[it,:,:]   = interpLateral(x,z[:,0,:],u[:,0,:],grid.xh+x0,grid.z)
            boundaryInput.vsouth[it,:,:]   = interpLateral(x,z[:,0,:],v[:,0,:],grid.x+x0,grid.z)
            boundaryInput.wsouth[it,:,:]   = interpLateral(x,z[:,0,:],w[:,0,:],grid.x+x0,grid.zh)
            boundaryInput.thlsouth[it,:,:] = interpLateral(x,z[:,0,:],thl[:,0,:],grid.x+x0,grid.z)
            boundaryInput.qtsouth[it,:,:]  = interpLateral(x,z[:,0,:],qt[:,0,:],grid.x+x0,grid.z)
            boundaryInput.e12south[it,:,:] = e12
            # Interpolate north boundary
            boundaryInput.unorth[it,:,:]   = interpLateral(x,z[:,-1,:],u[:,-1,:],grid.xh+x0,grid.z)
            boundaryInput.vnorth[it,:,:]   = interpLateral(x,z[:,-1,:],v[:,-1,:],grid.x+x0,grid.z)
            boundaryInput.wnorth[it,:,:]   = interpLateral(x,z[:,-1,:],w[:,-1,:],grid.x+x0,grid.zh)
            boundaryInput.thlnorth[it,:,:] = interpLateral(x,z[:,-1,:],thl[:,-1,:],grid.x+x0,grid.z)
            boundaryInput.qtnorth[it,:,:]  = interpLateral(x,z[:,-1,:],qt[:,-1,:],grid.x+x0,grid.z)
            boundaryInput.e12north[it,:,:] = e12
            # Interpolate top boundary
            boundaryInput.utop[it,:,:]     = interpTop(x,y,z,u,grid.xh+x0,grid.y+y0,grid.zh[-1])
            boundaryInput.vtop[it,:,:]     = interpTop(x,y,z,v,grid.x+x0,grid.yh+y0,grid.zh[-1])
            boundaryInput.wtop[it,:,:]     = interpTop(x,y,z,w,grid.x+x0,grid.y+y0,grid.zh[-1])
            boundaryInput.thltop[it,:,:]   = interpTop(x,y,z,thl,grid.x+x0,grid.y+y0,grid.zh[-1])
            boundaryInput.qttop[it,:,:]    = interpTop(x,y,z,qt,grid.x+x0,grid.y+y0,grid.zh[-1])
            boundaryInput.e12top[it,:,:]   = e12
        # Close netcdf file
        boundaryInput.exit()
        del boundaryInput
        end_time = datetime.now()
        hours = (end_time-start_time).days*24+(end_time-start_time).seconds//3600
        minutes = ((end_time-start_time).seconds//60)%60
        seconds = ((end_time-start_time).seconds-hours*3600-minutes*60)
        print(f"Finished in {hours:02}:{minutes:02}:{seconds:02}")
    # -------------------- Create profiles ---------------------- #
    if lprof:
        start_time = datetime.now()
        print("Start creation of profiles")
        # Create files and headers
        profFile = open(pathWrite+"prof.inp."+str(iexpnr).zfill(3), "w")
        lscaleFile = open(pathWrite+"lscale.inp."+str(iexpnr).zfill(3),"w")
        scalarFile = open(pathWrite+"scalar.inp."+str(iexpnr).zfill(3),'w')
        profFile.write("# Harmonie forced case with open boundaries\n")
        profFile.write("# zf thl qt u v tke\n")
        lscaleFile.write("# #large_scale forcing terms\n")
        lscaleFile.write("# zf ug vg wfls dqtdxls dqtdyls dqtdtls dthlrad\n")
        scalarFile.write("Scalar input\n")
        scalarFile.write("zt (m) qr (kg kg-1) nr (kg kg-1))")
        # Load fields first time step
        z  = nch_phi.variables['phi'][it0,::-1,iy0:iye+1,ix0:ixe+1]/cd['grav']
        u  = nch_u.variables['ua'][it0,::-1,iy0:iye+1,ix0:ixe+1]
        v  = nch_v.variables['va'][it0,::-1,iy0:iye+1,ix0:ixe+1]
        w  = nch_w.variables['w'][it0,::-1,iy0:iye+1,ix0:ixe+1]
        T  = nch_T.variables['ta'][it0,::-1,iy0:iye+1,ix0:ixe+1]
        qt = nch_qt.variables['hus'][it0,::-1,iy0:iye+1,ix0:ixe+1]
        ql = nch_ql.variables['clw'][it0,::-1,iy0:iye+1,ix0:ixe+1]
        # Surface variables
        zs  = np.zeros(np.shape(z)[1:3])
        us  = np.zeros(np.shape(u)[1:3])
        vs  = np.zeros(np.shape(v)[1:3])
        ws  = np.zeros(np.shape(w)[1:3])
        Ts  = nch_Ts.variables['tas'][it0,iy0:iye+1,ix0:ixe+1]
        qts = nch_qts.variables['huss'][it0,iy0:iye+1,ix0:ixe+1]
        ps  = nch_ps.variables['ps'][it0,iy0:iye+1,ix0:ixe+1]
        # Field conversions HARMONIE -> LES
        p = calcPressure(ps)
        exner  = (p/cd['p0'])**(cd['Rd']/cd['cp'])
        exners = (ps/cd['p0'])**(cd['Rd']/cd['cp'])
        th   = T  / exner
        ths  = Ts / exners
        thl  = th - cd['Lv'] / (cd['cp'] * exner) * ql
        # Concatenate surface fields to 3d fields
        z   = np.concatenate((zs[None,:,:],z),axis=0)
        u   = np.concatenate((us[None,:,:],u),axis=0)
        v   = np.concatenate((vs[None,:,:],v),axis=0)
        w   = np.concatenate((ws[None,:,:],w),axis=0)
        thl = np.concatenate((ths[None,:,:],thl),axis=0)
        qt  = np.concatenate((qts[None,:,:],qt),axis=0)
        # Interpolate fields to DALES levels and get slab averages
        uprof   = np.mean(interp3d(x,y,z,u,x,y,grid.z),axis=(1,2))
        vprof   = np.mean(interp3d(x,y,z,v,x,y,grid.z),axis=(1,2))
        thlprof = np.mean(interp3d(x,y,z,thl,x,y,grid.z),axis=(1,2))
        qtprof  = np.mean(interp3d(x,y,z,qt,x,y,grid.z),axis=(1,2))

        for i in range(grid.ktot):
            profFile.write(str(grid.z[i])+" "+str(thlprof[i])+" "+str(qtprof[i])+" "+str(uprof[i])+" "+str(vprof[i])+" "+str(e12)+"\n")
            lscaleFile.write(str(grid.z[i])+" "+str(0.)+" "+str(0.)+" "+str(0.)+" "+str(0.)+" "+str(0.)+" "+str(0.)+" "+str(0.)+"\n")
            scalarFile.write(str(grid.z[i])+" "+str(0.)+" "+str(0.)+"\n")

        profFile.close()
        lscaleFile.close()
        scalarFile.close()
        end_time = datetime.now()
        hours = (end_time-start_time).days*24+(end_time-start_time).seconds//3600
        minutes = ((end_time-start_time).seconds//60)%60
        seconds = ((end_time-start_time).seconds-hours*3600-minutes*60)
        print(f"Finished in {hours:02}:{minutes:02}:{seconds:02}")
    # ------------------- Create rrtmg input -------------------- #
    if lrad:
        start_time = datetime.now()
        print("Start creation of backrad.inp."+str(iexpnr).zfill(3)+".nc")
        # Get number of levels
        nlev = nch_u.dimensions['lev'].size
        # Create netcdf file
        radInput = RadInput(pathWrite,iexpnr,nlev)
        # Calculate mean pressure profile
        p = np.zeros(nlev)
        for it in range(Nt):
            ps  = nch_ps.variables['ps'][it+it0,iy0:iye+1,ix0:ixe+1]
            p   = p + np.mean(calcPressure(ps)/Nt,axis=(1,2))
        radInput.p[:]  = p
        radInput.T[:]  = np.mean(nch_T.variables['ta'][it0:it0+Nt,::-1,iy0:iye+1,ix0:ixe+1],axis=(0,2,3))
        radInput.qt[:] = np.mean(nch_qt.variables['hus'][it0:it0+Nt,::-1,iy0:iye+1,ix0:ixe+1],axis=(0,2,3))
        # Close netcdf file
        radInput.exit()
        del radInput
        end_time = datetime.now()
        hours = (end_time-start_time).days*24+(end_time-start_time).seconds//3600
        minutes = ((end_time-start_time).seconds//60)%60
        seconds = ((end_time-start_time).seconds-hours*3600-minutes*60)
        print(f"Finished in {hours:02}:{minutes:02}:{seconds:02}")
    # ------------------------ Finalize ------------------------- #
    # Close Harmonie netcdf files
    nch_u.close()
    nch_v.close()
    nch_w.close()
    nch_T.close()
    nch_qt.close()
    nch_ql.close()
    nch_phi.close()
    nch_ps.close()
    nch_Ts.close()
    nch_qts.close()

class Grid:
    def __init__(self, xsize, ysize, itot, jtot, ktot, dz, dz0, alpha, stretched):

        # Calculate and store DALES grid
        self.xsize = xsize
        self.ysize = ysize

        self.itot = itot
        self.jtot = jtot
        self.ktot = ktot

        self.dx = xsize/itot
        self.dy = ysize/jtot

        self.x = np.arange(0.5*self.dx, self.xsize, self.dx)
        self.y = np.arange(0.5*self.dy, self.ysize, self.dy)

        self.xh = np.arange(0, self.xsize+self.dx, self.dx)
        self.yh = np.arange(0, self.ysize+self.dy, self.dy)

        if stretched:
            self.dz = np.zeros(ktot)
            self.z = np.zeros(ktot)
            self.zh = np.zeros(ktot+1)
            self.dz[:]  = dz0 * (1 + alpha)**np.arange(ktot)
            self.zh[1:] = np.cumsum(self.dz)
            self.z[:]   = 0.5 * (self.zh[1:] + self.zh[:-1])
            self.zsize  = self.zh[-1]
        else:
            self.dz = np.ones(ktot)*dz
            self.zsize = ktot*dz
            self.z  = np.arange(0.5*dz, self.zsize, dz)
            self.zh = np.arange(0, self.zsize+dz, dz)

class HeteroInput:
    def __init__(self,pathWrite,iexpnr,grid):
        # Create netcdf file
        self.nc = Dataset(pathWrite+'initfields.inp.'+str(iexpnr).zfill(3)+'.nc','w')
        self.nc.createDimension('zt',grid.ktot)
        self.nc.createDimension('yt',grid.jtot)
        self.nc.createDimension('xt',grid.itot)
        self.nc.createDimension('zm',grid.ktot+1)
        self.nc.createDimension('ym',grid.jtot+1)
        self.nc.createDimension('xm',grid.itot+1)
        self.zt = self.nc.createVariable('zt','f4',('zt'))
        self.yt = self.nc.createVariable('yt','f4',('yt'))
        self.xt = self.nc.createVariable('xt','f4',('xt'))
        self.zm = self.nc.createVariable('zm','f4',('zm'))
        self.ym = self.nc.createVariable('ym','f4',('ym'))
        self.xm = self.nc.createVariable('xm','f4',('xm'))
        self.u = self.nc.createVariable('u0','f4',('zt','yt','xm'))
        self.v = self.nc.createVariable('v0','f4',('zt','ym','xt'))
        self.w = self.nc.createVariable('w0','f4',('zm','yt','xt'))
        self.thl = self.nc.createVariable('thl0','f4',('zt','yt','xt'))
        self.qt = self.nc.createVariable('qt0','f4',('zt','yt','xt'))
        self.e12 = self.nc.createVariable('e120','f4',('zt','yt','xt'))
        # Write dimensions to netcdf file
        self.zt[:] = grid.z
        self.yt[:] = grid.y
        self.xt[:] = grid.x
        self.zm[:] = grid.zh
        self.ym[:] = grid.yh
        self.xm[:] = grid.xh
    def exit(self):
        # Close netcdf file
        self.nc.close()

class BoundaryInput:
    def __init__(self,pathWrite,iexpnr,grid):
        # Create netcdf file
        self.nc = Dataset(pathWrite+'openboundaries.inp.'+str(iexpnr).zfill(3)+'.nc','w')
        self.nc.createDimension('time',None)
        self.nc.createDimension('zt',grid.ktot)
        self.nc.createDimension('yt',grid.jtot)
        self.nc.createDimension('xt',grid.itot)
        self.nc.createDimension('zm',grid.ktot+1)
        self.nc.createDimension('ym',grid.jtot+1)
        self.nc.createDimension('xm',grid.itot+1)
        self.time = self.nc.createVariable('time','f4',('time'))
        self.zt = self.nc.createVariable('zt','f4',('zt'))
        self.yt = self.nc.createVariable('yt','f4',('yt'))
        self.xt = self.nc.createVariable('xt','f4',('xt'))
        self.zm = self.nc.createVariable('zm','f4',('zm'))
        self.ym = self.nc.createVariable('ym','f4',('ym'))
        self.xm = self.nc.createVariable('xm','f4',('xm'))
        self.uwest = self.nc.createVariable('uwest','f4',('time','zt','yt'))
        self.vwest = self.nc.createVariable('vwest','f4',('time','zt','ym'))
        self.wwest = self.nc.createVariable('wwest','f4',('time','zm','yt'))
        self.thlwest = self.nc.createVariable('thlwest','f4',('time','zt','yt'))
        self.qtwest = self.nc.createVariable('qtwest','f4',('time','zt','yt'))
        self.e12west = self.nc.createVariable('e12west','f4',('time','zt','yt'))
        self.ueast = self.nc.createVariable('ueast','f4',('time','zt','yt'))
        self.veast = self.nc.createVariable('veast','f4',('time','zt','ym'))
        self.weast = self.nc.createVariable('weast','f4',('time','zm','yt'))
        self.thleast = self.nc.createVariable('thleast','f4',('time','zt','yt'))
        self.qteast = self.nc.createVariable('qteast','f4',('time','zt','yt'))
        self.e12east = self.nc.createVariable('e12east','f4',('time','zt','yt'))
        self.usouth = self.nc.createVariable('usouth','f4',('time','zt','xm'))
        self.vsouth = self.nc.createVariable('vsouth','f4',('time','zt','xt'))
        self.wsouth = self.nc.createVariable('wsouth','f4',('time','zm','xt'))
        self.thlsouth = self.nc.createVariable('thlsouth','f4',('time','zt','xt'))
        self.qtsouth = self.nc.createVariable('qtsouth','f4',('time','zt','xt'))
        self.e12south = self.nc.createVariable('e12south','f4',('time','zt','xt'))
        self.unorth = self.nc.createVariable('unorth','f4',('time','zt','xm'))
        self.vnorth = self.nc.createVariable('vnorth','f4',('time','zt','xt'))
        self.wnorth = self.nc.createVariable('wnorth','f4',('time','zm','xt'))
        self.thlnorth = self.nc.createVariable('thlnorth','f4',('time','zt','xt'))
        self.qtnorth = self.nc.createVariable('qtnorth','f4',('time','zt','xt'))
        self.e12north = self.nc.createVariable('e12north','f4',('time','zt','xt'))
        self.utop = self.nc.createVariable('utop','f4',('time','yt','xm'))
        self.vtop = self.nc.createVariable('vtop','f4',('time','ym','xt'))
        self.wtop = self.nc.createVariable('wtop','f4',('time','yt','xt'))
        self.thltop = self.nc.createVariable('thltop','f4',('time','yt','xt'))
        self.qttop = self.nc.createVariable('qttop','f4',('time','yt','xt'))
        self.e12top = self.nc.createVariable('e12top','f4',('time','yt','xt'))
        #self.thls   = self.nc.createVariable('thls','f4',('time','yt','xt'))
        # Write dimensions to netcdf file
        self.zt[:] = grid.z
        self.yt[:] = grid.y
        self.xt[:] = grid.x
        self.zm[:] = grid.zh
        self.ym[:] = grid.yh
        self.xm[:] = grid.xh
    def exit(self):
        # Close netcdf file
        self.nc.close()

class RadInput:
    def __init__(self,pathWrite,iexpnr,nlev):
        # Create netcdf file
        self.nc = Dataset(pathWrite+'backrad.inp.'+str(iexpnr).zfill(3)+'.nc','w')
        self.nc.createDimension('lev',nlev)
        self.p  = self.nc.createVariable('lev','f4',('lev'))
        self.T  = self.nc.createVariable('T','f4',('lev'))
        self.qt = self.nc.createVariable('q','f4',('lev'))
    def exit(self):
        self.nc.close()

def calcPressure(ps):
    # Calculate pressure with model coefficients given in H43_65lev.txt
    coeff = np.loadtxt('H43_65lev.txt')
    ph = coeff[:,1,None,None]+(ps[None,:,:]*coeff[:,2,None,None])
    p = 0.5*(ph[1:,:,:]+ph[:-1,:,:])
    return p[::-1]

def interp3d(xLS,yLS,zLS,val,x,y,z):
    fieldLES = np.zeros((z.size,y.size,x.size))
    for k in range(z.size):
        fieldLES[k,:,:] = interpTop(xLS,yLS,zLS,val,x,y,z[k])
    return fieldLES

def interpTop(xLS,yLS,zLS,val,x,y,ztop):
    fieldLES = np.zeros((y.size,x.size))
    valtemp = np.zeros((yLS.size,xLS.size))
    # Vertical interpolation to DALES height level
    for i in range(xLS.size):
        for j in range(yLS.size):
            kb = np.where(zLS[:,j,i] - ztop <= 0)[0][-1]
            kt = kb+1
            fkb = (zLS[kt,j,i]-ztop)/(zLS[kt,j,i]-zLS[kb,j,i])
            fkt = 1-fkb
            valtemp[j,i] = fkb*val[kb,j,i]+fkt*val[kt,j,i]
    # Horizontal interpolation
    for i in range(x.size):
        il   = np.where(xLS - x[i] <= 0)[0][-1]
        if abs(x[i]-xLS[-1])<10**-5 :
            ir = il
            fil = 1
            fir = 0
        else:
            ir   = il+1
            fil  = (xLS[ir]-x[i])/(xLS[ir]-xLS[il])
            fir  = 1-fil
        for j in range(y.size):
            jl   = np.where(yLS - y[j] <= 0)[0][-1]
            if abs(y[j]-yLS[-1])<10**-5 :
                jr = jl
                fjl = 1
                fjr = 0
            else:
                jr   = jl+1
                fjl  = (yLS[jr]-y[j])/(yLS[jr]-yLS[jl])
                fjr  = 1-fjl
            fieldLES[j,i] = fil*(fjl*valtemp[jl,il]+fjr*valtemp[jr,il])+fir*(fjl*valtemp[jl,ir]+fjr*valtemp[jr,ir])
    return fieldLES

def interpLateral(xLS,zLS,val,x,z):
    fieldLES = np.zeros((z.size,x.size))
    for i in range(x.size):
        for k in range(z.size):
            # Get horizontal factors
            il   = np.where(xLS - x[i] <= 0)[0][-1]
            if abs(x[i]-xLS[-1])<10**-5 :
                ir = il
                fil = 1
                fir = 0
            else:
                ir = il+1
                fil  = (xLS[ir]-x[i])/(xLS[ir]-xLS[il])
                fir  = 1-fil
            # Get vertical factors
            kbl  = np.where(zLS[:,il]-z[k] <= 0)[0][-1]
            ktl  = kbl+1
            fkbl = (zLS[ktl,il]-z[k])/(zLS[ktl,il]-zLS[kbl,il])
            fktl = 1-fkbl

            kbr  = np.where(zLS[:,ir]-z[k] <= 0)[0][-1]
            ktr  = kbr+1
            fkbr = (zLS[ktr,ir]-z[k])/(zLS[ktr,ir]-zLS[kbr,ir])
            fktr = 1-fkbr
            fieldLES[k,i] = fil*(fkbl*val[kbl,il]+fktl*val[ktl,il])+fir*(fkbr*val[kbr,ir]+fktr*val[ktr,ir])
    return fieldLES

if __name__ == '__main__':
    main()
