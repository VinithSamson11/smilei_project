#Field and Track Data Reader

from extractdata import *
from smilei import *
import numpy as np

'''
Currently the two classes Field and Track can only read one FieldDiagnostic h5 file 
and one TrackDiagnostic h5 file, if there are multiple Field.h5 files or Track.h5 files, this code shoots error.
This problem will be fixed soon :)

And for unit conversions, this code uses smilei.py for now, but soon it will be replaced by dedicated units module.

This code contains user friendly error messages XD 
'''

class Field:
    '''
    This class is responsible for extracting the field [Er, El, Et, Br, Bl, Bt, Jr, Jl, Jt, Rho] 
    according to the modes. 
    You can call this class Field using the syntax
    
    field = Field(h5path = '/path/to/the/file', field = 'field name', moving = True or False, 
                  timesteps = timestep, modes = [mode0, mode1,... ], theta = theta in radians) 
    NOTE : mention 'field names' with prefix '-' if you need negative of the field values.

    To get the data array of a 'field name', 
    field._getData(timestep = timestep)

    To get the axis array of a field name',
    field.getAxis(timestep = timestep, axis = 'axis name') 
    NOTE: Only one axis at a time

    Currently the data units are in [fs, pC/cm^3, um, TV/m], the defined units are taken from smilei.py file 
    '''   
    
    #To return array or list of a h5 data
    def _grp(self,dir,flag=None): #grp stands for group in a HDF5 file
        
        if flag=='ar':
            return np.array(self.h5item.get(dir))
        elif flag=='l':
            return list(self.h5item.get(dir))
        else:
            return self.h5item.get(dir)
        
    #To return the number of AM modes and Field names in the dump.
    def getAMmodes(self):    
        strings=self._grp(dir='data/0000000000',flag='l')
        max_suffix = float('-inf')
        max_suffix_string = None
        prefix_list=[]

        for s in strings:
            prefix, suffix = s.split('_mode_')
            suffix_int = int(suffix)
            if suffix_int > max_suffix:
                max_suffix = suffix_int
                max_suffix_string = s
            prefix_list.append(prefix)

        return [prefix_list, int(max_suffix_string[-1])+1]
    
    #initiates the variables and validates the user input.
    def __init__(self, h5path, field, moving=False, timesteps=None, modes=None,theta=None):
        
        self.h5path = h5path + "Fields0.h5"
        if os.path.isfile(self.h5path) is False:
            raise IOError('{} file or path does not exist'.format(self.h5path)) 
        
        self.moving = moving
        self.user_timesteps = timesteps
        
        self.h5item = gethdf5(h5path)
        self.timestep_list = [int(x) for x in self._grp(dir='data',flag='l')]

        self.avail_field_list = self.getAMmodes()[0]
        if field[0] == '-' or field[0] == '+':
            self.field = field
        else:
            self.field = '+'+field 
        if self.field[1:] not in self.avail_field_list:
            raise IOError('Field {} not available in simulation dump'.format(self.field))
        
        self.AM_modes = self.getAMmodes()[1]
        if modes is not None:
            self.modes = modes
        else:
            self.modes = [i for i in range(self.AM_modes)]
        
        if len(self.modes) == 0:
             raise IOError('Invalid mode : {} '.format(self.modes))
        for im in self.modes:
            if im<0 or im > (self.AM_modes-1):
                raise IOError('Invalid mode : {} '.format(self.modes))
            
        if theta is not None:
            self.theta = theta
        else:
            self.theta = 0 

    #To get time list.
    def _getTime(self,usertime):    
        
        #Finding the closest integer timestep available from the user timestep
        if usertime is not None:
            int_timesteps = min(self.timestep_list, key=lambda x: (abs(x - usertime), x))
        else:
            int_timesteps = self.timestep_list[-1]
        
        sim_timesteps = str(int_timesteps).zfill(10)   #timestep format like in the simulation dump
        real_time=int(sim_timesteps)*dt/fs    #timestep into femto-seconds unit
        
        return [int_timesteps,sim_timesteps,real_time]  #[integer timestep of simulation time, simulation timestep(which is in string), real time (in femto fs)]

    #To get the array of the field data, the element of each subarray is a complex number.
    def getArray(self, timestep=None):

        timestep = self._getTime(usertime=timestep)[1]

        array_list=[]
        for mode in range(self.AM_modes):

            field_name = self.field[1:]+"_mode_"+str(mode)
            field_array = self._grp(dir='data/'+timestep+'/'+field_name,flag='ar')*int(self.field[0]+"1")
            field_array_shape = field_array.shape
            reshaped_array = field_array.reshape(field_array_shape[0], field_array_shape[1]//2, 2)
            new_array = reshaped_array[:, :, 0] + 1j * reshaped_array[:, :, 1]
            array_list.append(new_array)
        
        mod_data= np.stack(array_list,axis=0)     #Modified array of shape (no.of.modes, Nx, Nr)
        return mod_data             
    
    #Discrete Fourier Transform (DFT)
    def mode_expansion_DFT(self,timestep):
        
        mod_data=self.getArray(timestep=timestep)
        F = np.zeros_like(np.real(mod_data[0]))                       
        
        '''for m in self.modes:
            F += self.mod_data[m]*np.exp(-1j*m*self.theta)
        '''
        for m in self.modes:
            F += np.real(mod_data[m])*np.cos(m*self.theta)+np.imag(mod_data[m])*np.sin(m*self.theta)

        return F   
    
    #DFT using Fast Fourier Transform (FFT)
    def mode_expansion_FFT(self,timestep):

        rawdata=self.getArray(timestep=timestep)
        Nm,Nx,Nr = rawdata.shape
        Nth = (Nm+1)//2
        #if Ntheta is None or Ntheta < Nth:
        Ntheta = Nth
        fd = np.empty((Nx, Ntheta, Nr), dtype=np.complex128)

        fd[:, 0, :].real = rawdata[0, :, :]
        rawdatasw = np.swapaxes(rawdata, 0, 1)
        fd[:, 1:Nth, :].real = rawdatasw[:, 1::2, :]
        fd[:, 1:Nth, :].imag = rawdatasw[:, 2::2, :]

        fd = np.fft.fft(fd, axis=1).real
        mod_fd = np.swapaxes(fd,0,1)
        return mod_fd

    #Mode expansion algorithm is determined here.
    def mode_expansion(self,timestep):
        
        #If theta is gievn by user, mode expansion is done by traditional Discrete Fourier Transform
        if self.theta is not None:
            F_total = self.mode_expansion_DFT(timestep=timestep)

        #If theta value is None, mode expansion is carried out by Fast Fourier Transform algorithm
        elif self.theta is None:
            F_total = self.mode_expansion_FFT(timestep=timestep)

        return F_total

    #To get the axis array.
    def getAxis(self, axis,timestep=None):
        
        timestep=self._getTime(usertime=timestep)[1]
        Nx,Nr = self.getArray(timestep=timestep)[0].shape

        if axis is None:
            raise IOError("Invalid axis")
        elif len(axis)>1:
            raise IOError("Only one axis at a time")
        
        if axis == "x" and self.moving:      #If moving = True, the x axis data modifies according to the moving window.
            x_min = self._grp(dir='data/'+timestep).attrs['x_moved']*c_over_omega0*1e6  
            x_max = x_min + Lx*c_over_omega0*1e6
            x_axis = np.linspace(x_min, x_max, Nx-1)
            return x_axis
        elif axis == "x":
            x_min = 0  
            x_max = x_min+Lx*c_over_omega0*1e6
            x_axis = np.linspace(x_min, x_max, Nx-1)
            return x_axis    
        elif axis == "r":
            r_max = Lr*c_over_omega0*1e6
            r_axis = np.linspace(0, r_max, Nr-1)
            return r_axis
        else:
            raise IOError("Invalid axis")
    
    #To get the data array of a field.
    def _getData(self,timestep=None):
        
        timestep = self._getTime(usertime=timestep)[1]

        #conversion factors
        density_factor = (ncrit*e)*(1e12/1e6) #density normalized units --> pC/cm^3
        efield_factor = me*c*omega0/e/1e12 #E field normalized units --> TV/m           
        
        if self.field[1:] == "Rho":
            scalar_field_array = self.mode_expansion(timestep=timestep) * density_factor
       
        if self.field[1:2] == "E":
            scalar_field_array = self.mode_expansion(timestep=timestep) * efield_factor
        
        return scalar_field_array

    def plot(self,vmin=None, vmax=None, cmap=None,saveas=None,timestep=None):

        time=self._getTime(usertime=timestep)

        if self.moving:
            xmin = self._grp(dir='data/'+time[1]).attrs['x_moved']*c_over_omega0*1e6
        else:
            xmin = 0
        xmax = xmin + (Lx*c_over_omega0*1e6)
    
        ymin = 0
        ymax = Lr*c_over_omega0*1e6
    
        fig, ax = plt.subplots()
    
        im=ax.imshow(self._getData(timestep=timestep).T ,cmap=cmap, origin='lower', 
           extent=(xmin, xmax, ymin, ymax),vmin=vmin,vmax=vmax,aspect="auto")
    
        plt.colorbar(im, ax=ax, shrink=1, aspect=20)
        plt.xlabel('x [um]')
        plt.ylabel('r [um]')
        if self.field in self.efield_names:
            plt.title(self.field[1:]+' [TV/m]  t='+str(self.real_time)+" fs")
        elif self.field in self.rho_names:
            plt.title(self.field[1:]+' [pC/cm^3]  t='+str(self.real_time)+" fs") 
        if saveas!=None:
            plt.savefig(saveas)
        plt.show()


class Track:

    '''
    Class Track is responsible for getting particle data ['position', 'momentum', 'charge', 'ID', 'weight'].

    You can call this class using syntax,
    track = Track(h5path = '/path/to/the/file', species = 'species name', moving = True or False)

    To get specific particle data array,
    track._getData(data = 'data type', timestep = timestep)

    Name of the specific data types:
    position --> 'x', 'y', 'z'
    momentum --> 'px', 'py', 'pz'
    charge   --> 'charge'
    ID       --> 'id'
    weight   --> 'weight'

    '''

    def _grp(self,dir,flag=None): #grp stands for group in a HDF5 file
        
        if flag=='ar':
            return np.array(self.h5item.get(dir))
        elif flag=='l':
            return list(self.h5item.get(dir))
        else:
            return self.h5item.get(dir)

    def __init__(self, h5path, species, moving=None):

        self.h5path = h5path + 'TrackParticlesDisordered_electron.h5'
        if os.path.isfile(self.h5path) is False:
            raise IOError('{} file or path does not exist'.format(self.h5path)) 

        self.moving = moving
        self.h5item = gethdf5(self.h5path)
        self.timestep_list = [int(x) for x in self._grp(dir='data',flag='l')]

        self.species_name = self._grp(dir='data/0000000000/particles',flag='l')
        if species is None:
            raise IOError("Mention a Species")
        elif species not in self.species_name:
            raise IOError("Species {} not in simulation dump".format(species))
        else:
            self.species = species

    def _getTime(self,usertime):    
        
        #Finding the simulation_timestep[sim_timesteps] and real time[real_time]
        if usertime is not None:
            int_timesteps = min(self.timestep_list, key=lambda x: (abs(x - usertime), x))
        else:
            int_timesteps = self.timestep_list[-1]
        
        sim_timesteps = str(int_timesteps).zfill(10)   #timestep format in the data dump
        real_time=int(sim_timesteps)*dt/fs    #timestep into femto-seconds unit
        
        return [int_timesteps,sim_timesteps,real_time]        #[integer timestep of simulation time, simulation timestep(which is in string), real time (in femto fs)]
    
    def _getData(self,data, timestep=None):

        time = self._getTime(usertime=timestep)[1]

        self.data_name = self._grp(dir='data/0000000000/particles/'+self.species,flag='l')
        self.pos_name = self._grp(dir='data/0000000000/particles/'+self.species+'/position',flag='l')
        self.mom_name = ['px','py','pz']
        if data is None:
            raise IOError("Mention particle data")
        elif data not in self.data_name and data not in self.pos_name and data not in self.mom_name:
            raise IOError("Particle Data {} not available in simulation dump".format(data))
        else:
            self.data = data

        if self.data in ['x','y','z']:
            data_array = self._grp(dir='data/'+time+'/particles/'+self.species+'/position/'+self.data,flag='ar')
        elif self.data in self.mom_name:
            data_array = self._grp(dir='data/'+time+'/particles/'+self.species+'/momentum/'+self.data[1:],flag='ar')
        else:
            data_array = self._grp(dir='data/'+time+'/particles/'+self.species+'/'+self.data,flag='ar')
        
        return data_array


