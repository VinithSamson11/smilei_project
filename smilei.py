# coding: utf-8

"""
    GENERAL DEFINITIONS FOR SMILEI
"""

import math, os, gc, operator

def _add_metaclass(metaclass):
    """Class decorator for creating a class with a metaclass."""
    # Taken from module "six" for compatibility with python 2 and 3
    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get('__slots__')
        if slots is not None:
            if isinstance(slots, str): slots = [slots]
            for slots_var in slots: orig_vars.pop(slots_var)
        orig_vars.pop('__dict__', None)
        orig_vars.pop('__weakref__', None)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper


class SmileiComponentType(type):
    """Metaclass to all Smilei components"""

    # Constructor of classes
    def __init__(self, name, bases, attrs):
        self._list = []
        self._verify = True
        # Run standard metaclass init
        super(SmileiComponentType, self).__init__(name, bases, attrs)

    # Functions to define the iterator
    def __iter__(self):
        self.current = 0
        return self
    def next(self):
        if self.current >= len(self._list):
            raise StopIteration
        self.current += 1
        return self._list[self.current - 1]
    __next__ = next #python3

    # Function to return one given instance, for example DiagParticleBinning[0]
    # Special case: species can also be indexed by their name: Species["ion1"]
    # Special case: diagnostics can also be indexed by their label: DiagParticleBinning["x-px"]
    def __getitem__(self, key):
        try:
            for obj in self._list:
                if obj.name == key:
                    return obj
        except:
            pass
        return self._list[key]
    
    def has(self, key):
        if type(key) is int and key < len(self._list):
            return True
        for obj in self._list:
            if obj.name == key:
                return True
        return False
    
    # Function to return the number of instances, for example len(Species)
    def __len__(self):
        return len(self._list)

    # Function to display the list of instances
    def __repr__(self):
        if len(self._list)==0:
            return "<Empty list of "+self.__name__+">"
        else:
            l = []
            for obj in self._list: l.append(str(obj))
            return "["+", ".join(l)+"]"

@_add_metaclass(SmileiComponentType)
class SmileiComponent(object):
    """Smilei component generic class"""

    # Function to initialize new components
    def _fillObjectAndAddToList(self, cls, obj, **kwargs):
        # add all kwargs as internal class variables
        if kwargs is not None:
            deprecated = {
                "output_format":"See documentation for radiation reaction",
                "clrw":"See https://smileipic.github.io/Smilei/namelist.html#main-variables",
                "h_chipa_min":"See documentation for radiation reaction",
                "h_chipa_max":"See documentation for radiation reaction",
                "h_dim":"See documentation for radiation reaction",
                "h_computation_method":"See documentation for radiation reaction",
                "integfochi_chipa_min":"See documentation for radiation reaction",
                "integfochi_chipa_max":"See documentation for radiation reaction",
                "integfochi_dim":"See documentation for radiation reaction",
                "xip_chipa_min":"See documentation for radiation reaction",
                "xip_chipa_max":"See documentation for radiation reaction",
                "xip_power":"See documentation for radiation reaction or Breit-Wheeler",
                "xip_threshold":"See documentation for radiation reaction or Breit-Wheeler",
                "xip_chipa_dim":"See documentation for radiation reaction or Breit-Wheeler",
                "xip_chiph_dim":"See documentation for radiation reaction or Breit-Wheeler",
                "compute_table":"See documentation for Breit-Wheeler",
                "T_chiph_min":"See documentation for Breit-Wheeler",
                "T_chiph_max":"See documentation for Breit-Wheeler",
                "T_dim":"See documentation for Breit-Wheeler",
                "xip_chiph_min":"See documentation for Breit-Wheeler",
                "xip_chiph_max":"See documentation for Breit-Wheeler",
            }
            for key, value in kwargs.items():
                if key in deprecated:
                    raise Exception("Deprecated `"+key+"` parameter. "+deprecated[key])
                if key=="_list":
                    print("Python warning: in "+cls.__name__+": cannot have argument named '_list'. Discarding.")
                elif not hasattr(cls, key):
                    raise Exception("ERROR in the namelist: cannot define `"+key+"` in block "+cls.__name__+"()");
                else:
                    setattr(obj, key, value)
        # add the new component to the "_list"
        cls._list.append(obj)

    # Constructor for all SmileiComponents
    def __init__(self, **kwargs):
        self._fillObjectAndAddToList(type(self), self, **kwargs)

    def __repr__(self):
        return "<Smilei "+type(self).__name__+">"


class SmileiSingletonType(SmileiComponentType):
    """Metaclass to all Smilei singletons"""

    def __repr__(self):
        return "<Smilei "+str(self.__name__)+">"

@_add_metaclass(SmileiSingletonType)
class SmileiSingleton(SmileiComponent):
    """Smilei singleton generic class"""

    # Prevent having two instances
    def __new__(cls, **kwargs):
        if len(cls._list) >= 1:
            raise Exception("ERROR in the namelist: cannot define block "+cls.__name__+"() twice")
        return super(SmileiSingleton, cls).__new__(cls)

    # Constructor for all SmileiSingletons
    def __init__(self, **kwargs):
        cls = type(self)
        self._fillObjectAndAddToList(cls, cls, **kwargs)
        # Change all methods to static
        for k,v in cls.__dict__.items():
            if k[0]!='_' and hasattr(v,"__get__"):
               setattr(cls, k, staticmethod(v))

class ParticleData(object):
    """Container for particle data at run-time (for exposing particles in numpy)"""
    _verify = True

class Main(SmileiSingleton):
    """Main parameters"""

    # Default geometry info
    geometry = None
    cell_length = []
    grid_length = []
    number_of_cells = []
    timestep = None
    simulation_time = None
    number_of_timesteps = None
    interpolation_order = 2
    interpolator = "momentum-conserving"
    custom_oversize = 2
    number_of_patches = None
    patch_arrangement = "hilbertian"
    cluster_width = -1
    every_clean_particles_overhead = 100
    timestep = None
    number_of_AM = 2
    number_of_AM_relativistic_field_initialization = 1
    number_of_AM_classical_Poisson_solver = 1
    timestep_over_CFL = None
    cell_sorting = None
    gpu_computing = False                      # Activate the computation on GPU
    
    # PXR tuning
    spectral_solver_order = []
    initial_rotational_cleaning = False

    # Poisson tuning
    solve_poisson = True
    poisson_max_iteration = 50000
    poisson_max_error = 1.e-14

    # Relativistic Poisson tuning
    solve_relativistic_poisson = False
    relativistic_poisson_max_iteration = 50000
    relativistic_poisson_max_error = 1.e-22
    
    # BTIS3 interpolator
    use_BTIS3_interpolation = False

    # Default fields
    maxwell_solver = 'Yee'
    EM_boundary_conditions = [["periodic"]]
    EM_boundary_conditions_k = []
    save_magnectic_fields_for_SM = True
    number_of_pml_cells = [[10]]

    def default_sigma(x):
        return 20. * x**2  
    def default_integrate_sigma(x):
        return 20./3. * x**3  
    def default_kappa(x):
        return 1 + 79. * x**4  
    def default_integrate_kappa(x):
        return x + 79./5. * x**5  

    pml_sigma = [default_sigma, default_sigma, default_sigma, default_integrate_sigma]
    pml_kappa = [default_kappa, default_kappa, default_kappa, default_integrate_kappa]
    time_fields_frozen = 0.
    Laser_Envelope_model = False

    # Default Misc
    reference_angular_frequency_SI = 0.
    print_every = None
    random_seed = None
    print_expected_disk_usage = True

    terminal_mode = True

    def __init__(self, **kwargs):
        # Load all arguments to Main()
        super(Main, self).__init__(**kwargs)

        # Initialize timestep if not defined based on timestep_over_CFL
        if Main.timestep is None:
            if Main.timestep_over_CFL is None:
                raise Exception("timestep and timestep_over_CFL not defined")
            else:
                if Main.cell_length is None:
                    raise Exception("Need cell_length to calculate timestep")

                # Yee solver
                if Main.maxwell_solver == 'Yee':
                    if (Main.geometry=="AMcylindrical"):
                        Main.timestep = Main.timestep_over_CFL / math.sqrt(1./Main.cell_length[0]**2 + ((Main.number_of_AM-1)/Main.cell_length[1])**2 )
                    else:
                        dim = int(Main.geometry[0])
                        if dim<1 or dim>3:
                            raise Exception("timestep_over_CFL not implemented in geometry "+Main.geometry)
                        Main.timestep = Main.timestep_over_CFL / math.sqrt(sum([1./l**2 for l in Main.cell_length]))

                # Grassi
                elif Main.maxwell_solver == 'Grassi':
                    if Main.geometry == '2Dcartesian':
                        Main.timestep = Main.timestep_over_CFL * 0.7071067811*Main.cell_length[0];
                    else:
                        raise Exception("timestep_over_CFL not implemented in geometry "+Main.geometry)

                # GrassiSpL
                elif Main.maxwell_solver == 'GrassiSpL':
                    if Main.geometry == '2Dcartesian':
                        Main.timestep = Main.timestep_over_CFL * 0.6471948469*Main.cell_length[0];
                    else:
                        raise Exception("timestep_over_CFL not implemented in geometry "+Main.geometry)

                # M4
                elif Main.maxwell_solver == 'M4':
                    if Main.geometry == '1Dcartesian':
                        Main.timestep = Main.timestep_over_CFL * Main.cell_length[0];
                    elif Main.geometry == '2Dcartesian':
                        Main.timestep = Main.timestep_over_CFL * min(Main.cell_length[0:2])
                    elif Main.geometry == '3Dcartesian':
                        Main.timestep = Main.timestep_over_CFL * min(Main.cell_length)
                    else:
                        raise Exception("timestep_over_CFL not implemented in geometry "+Main.geometry)

                # None recognized solver
                else:
                    raise Exception("timestep: maxwell_solver not implemented "+Main.maxwell_solver)

        # Constraint on timestep for WT interpolation
        if Main.interpolator.lower() == "wt":
            if Main.geometry == '1Dcartesian':
                if Main.timestep > 0.5 * Main.cell_length[0]:
                    raise Exception("timestep for WT cannot be larger than 0.5*dx")
            elif Main.geometry == '2Dcartesian':
                if Main.timestep > 0.5 * min(Main.cell_length[0:2]):
                    raise Exception("timestep for WT cannot be larger than 0.5*min(dx,dy)")
            elif Main.geometry == '3Dcartesian':
                if Main.timestep > 0.5 * min(Main.cell_length):
                    raise Exception("timestep for WT cannot be larger than 0.5*min(dx,dy,dz)")
            else:
                raise Exception("WT interpolation not implemented in geometry "+Main.geometry)

        # Initialize simulation_time if not defined by the user
        if Main.simulation_time is None:
            if Main.number_of_timesteps is None:
                raise Exception("simulation_time and number_of_timesteps are not defined")
            Main.simulation_time = Main.timestep * Main.number_of_timesteps

        # Initialize grid_length if not defined based on number_of_cells and cell_length
        if (    len(Main.grid_length + Main.number_of_cells) == 0
             or len(Main.grid_length + Main.cell_length) == 0
             or len(Main.number_of_cells + Main.cell_length) == 0
             or len(Main.number_of_cells) * len(Main.grid_length) * len(Main.cell_length) != 0
           ):
                raise Exception("Main: you must define two (and only two) between grid_length, number_of_cells and cell_length")

        if len(Main.grid_length) == 0:
            Main.grid_length = [a*b for a,b in zip(Main.number_of_cells, Main.cell_length)]

        if len(Main.cell_length) == 0:
            Main.cell_length = [a/b for a,b in zip(Main.grid_length, Main.number_of_cells)]

        if len(Main.number_of_cells) == 0:
            Main.number_of_cells = [int(round(float(a)/float(b))) for a,b in zip(Main.grid_length, Main.cell_length)]
            old_grid_length = Main.grid_length
            Main.grid_length = [a*b for a,b in zip(Main.number_of_cells, Main.cell_length)]
            if smilei_mpi_rank == 0:
                different = [abs((a-b)/(a+b))>1e-10 for a,b in zip(Main.grid_length, old_grid_length)]
                if any(different):
                    print("\t[Python WARNING] Main.grid_length="+str(Main.grid_length)+" (was "+str(old_grid_length)+")")

class LoadBalancing(SmileiSingleton):
    """Load balancing parameters"""

    every                = 150
    initial_balance      = True
    cell_load            = 1.0
    frozen_particle_load = 0.1

class MultipleDecomposition(SmileiSingleton):
    """Multiple Decomposition parameters"""
    region_ghost_cells   = 2

# Radiation reaction configuration (continuous and MC algorithms)
class Vectorization(SmileiSingleton):
    """
    Vectorization parameters
    """
    mode                = "off"
    reconfigure_every   = 20
    initial_mode        = "off"


class MovingWindow(SmileiSingleton):
    """Moving window parameters"""

    time_start = 0.
    velocity_x = 1.
    number_of_additional_shifts = 0
    additional_shifts_time = 0.


class Checkpoints(SmileiSingleton):
    """Checkpoints parameters"""

    restart_dir = None
    restart_number = None
    dump_step = 0
    dump_minutes = 0.
    keep_n_dumps = 2
    dump_deflate = 0
    exit_after_dump = True
    file_grouping = 0
    restart_files = []

class CurrentFilter(SmileiSingleton):
    """Current filtering parameters"""
    model = "binomial"
    passes = [0]
    kernelFIR = [0.25,0.5,0.25]

class FieldFilter(SmileiSingleton):
    """Fields filtering parameters"""
    model = "Friedman"
    theta = 0.

class Species(SmileiComponent):
    """Species parameters"""
    name = None
    position_initialization = None
    momentum_initialization = ""
    particles_per_cell = None
    regular_number = []
    c_part_max = 1.0
    mass = None
    charge = None
    charge_density = None
    number_density = None
    mean_velocity = []  # Default value is     0, set in ParticleCreator function in species.cpp
    mean_velocity_AM = []
    temperature = []    # Default value is 1e-10, set in ParticleCreator function in species.cpp
    thermal_boundary_temperature = []
    thermal_boundary_velocity = [0.,0.,0.]
    pusher = "boris"

    # Radiation species parameters
    radiation_model = "none"
    radiation_photon_species = None
    radiation_photon_sampling = 1
    radiation_photon_gamma_threshold = 2.
    radiation_max_emissions = 10

    # Multiphoton Breit-Wheeler parameters
    multiphoton_Breit_Wheeler = [None,None]
    multiphoton_Breit_Wheeler_sampling = [1,1]

    # Particle merging species Parameters
    merging_method = "none"
    merge_every = 0
    merge_min_packet_size = 4
    merge_max_packet_size = 4
    merge_min_particles_per_cell = 4
    merge_min_momentum_cell_length = [1e-10,1e-10,1e-10]
    merge_momentum_cell_size = [16,16,16]
    merge_accumulation_correction = True
    merge_discretization_scale = "linear"
    merge_min_momentum = 1e-5

    time_frozen = 0.0
    radiating = False
    relativistic_field_initialization = False
    boundary_conditions = [["periodic"]]
    ionization_model = "none"
    ionization_electrons = None
    ionization_rate = None
    atomic_number = None
    maximum_charge_state = 0
    is_test = False
    relativistic_field_initialization = False
    keep_interpolated_fields = []

class ParticleInjector(SmileiComponent):
    """Parameters for particle injection at boundaries"""
    name = None
    species = None
    box_side = "xmin"
    position_initialization = "species"
    momentum_initialization = "species"
    mean_velocity = []  # Default value is     0, set in ParticleCreator function
    temperature = []    # Default value is 1e-10, set in ParticleCreator function
    charge_density = None
    number_density = None
    particles_per_cell = None
    regular_number = []
    time_envelope = 1
    
    
class Laser(SmileiComponent):
    """Laser parameters"""
    box_side = "xmin"
    omega = 1.
    chirp_profile = 1.
    time_envelope = 1.
    space_envelope = [1., 0.]
    phase = [0., 0.]
    delay_phase = [0., 0.]
    space_time_profile = None
    space_time_profile_AM = None
    file = None
    _offset = None

class LaserEnvelope(SmileiSingleton):
    """Laser Envelope parameters"""
    omega = 1.
    #time_envelope = 1.
    #space_envelope = [1., 0.]
    envelope_solver = "explicit"
    envelope_profile = None
    Envelope_boundary_conditions = [["reflective"]]
    Env_pml_sigma_parameters = [[0.90,2],[10.0,2],[10.0,2]]
    Env_pml_kappa_parameters = [[1.00,1.00,2],[1.00,1.00,2],[1.00,1.00,2]]
    Env_pml_alpha_parameters = [[0.90,0.90,1],[0.75,0.75,1],[0.75,0.75,1]]
    polarization_phi = 0.
    ellipticity = 0.


class Collisions(SmileiComponent):
    """Collisions parameters"""
    species1 = None
    species2 = None
    coulomb_log = 0.
    coulomb_log_factor = 1.
    every = 1
    debug_every = 0
    time_frozen = 0
    ionizing = False
    nuclear_reaction = None
    nuclear_reaction_multiplier = 0.


#diagnostics
class DiagProbe(SmileiComponent):
    """Probe diagnostic"""
    name = ""
    every = None
    number = []
    origin = []
    corners = []
    vectors = []
    fields = []
    flush_every = 1
    time_integral = False
    datatype = "double"

class DiagParticleBinning(SmileiComponent):
    """Particle Binning diagnostic"""
    name = ""
    deposited_quantity = None
    time_average = 1
    species = None
    axes = []
    every = None
    flush_every = 1

class DiagRadiationSpectrum(SmileiComponent):
    """Radiation Spectrum diagnostic"""
    name = ""
    time_average = 1
    species = None
    photon_energy_axis = None
    axes = []
    every = None
    flush_every = 1

class DiagScreen(SmileiComponent):
    """Screen diagnostic"""
    name = ""
    shape = None
    point = None
    vector = None
    direction = "both"
    deposited_quantity = None
    species = None
    axes = []
    time_average = 1
    every = None
    flush_every = 1

class DiagScalar(SmileiComponent):
    """Scalar diagnostic"""
    every = None
    precision = 10
    vars = []

class DiagFields(SmileiComponent):
    """Field diagnostic"""
    name = ""
    every = None
    fields = []
    time_average = 1
    subgrid = None
    flush_every = 1
    datatype = "double"

class DiagTrackParticles(SmileiComponent):
    """Track diagnostic"""
    name = ""
    species = None
    every = 0
    flush_every = 1
    filter = None
    attributes = ["x", "y", "z", "px", "py", "pz", "w"]

class DiagNewParticles(SmileiComponent):
    """Track diagnostic"""
    name = ""
    species = None
    every = 0
    flush_every = 1
    filter = None
    attributes = ["x", "y", "z", "px", "py", "pz", "w"]

class DiagPerformances(SmileiSingleton):
    """Performances diagnostic"""
    every = 0
    flush_every = 1
    patch_information = True

# external fields
class ExternalField(SmileiComponent):
    """External Field"""
    field = None
    profile = None

# external time fields
class PrescribedField(SmileiComponent):
    """External Time Field"""
    field = None
    profile = None

# external current (antenna)
class Antenna(SmileiComponent):
    """Antenna"""
    field = None
    time_profile  = None
    space_profile = None
    space_time_profile = None

# Particle wall
class PartWall(SmileiComponent):
    """Particle Wall"""
    kind = "none"
    x = None
    y = None
    z = None


# Radiation reaction configuration (continuous and MC algorithms)
class RadiationReaction(SmileiComponent):
    """
    Fine-tuning of synchrotron-like radiation reaction
    (classical continuous, quantum correction, stochastics and MC algorithms)
    """
    # Minimum particle_chi value for the discontinuous radiation
    # Under this value, the discontinuous approach is not applied
    minimum_chi_discontinuous = 1e-2
    # Threshold on particle_chi: if particle_chi < 1E-3 no radiation reaction
    minimum_chi_continuous = 1e-3

    # Path to read or write the tables/databases
    table_path = ""

    # Parameters for computing the tables
    Niel_computation_method = "table"

# MultiphotonBreitWheeler pair creation
class MultiphotonBreitWheeler(SmileiComponent):
    """
    Photon decay into electron-positron pairs
    """
    # Path the tables/databases
    table_path = ""

# Smilei-defined
smilei_mpi_rank = 0
smilei_mpi_size = 1
smilei_rand_max = 2**31-1

# Variable to set to False for the actual run (useful for the test mode)
_test_mode = True
smilei_version='4f145b3-master'
smilei_mpi_size = 16
smilei_rand_max = 2147483647

# Some predefined profiles (see doc)

def constant(value, xvacuum=-float("inf"), yvacuum=-float("inf"), zvacuum=-float("inf")):
    global Main
    if len(Main)==0:
        raise Exception("constant profile has been defined before `Main()`")
    if Main.geometry == "1Dcartesian":
        f = lambda x,t=0: value if x>=xvacuum else 0.
    if (Main.geometry == "2Dcartesian" or Main.geometry == "AMcylindrical"):
        f = lambda x,y,t=0: value if (x>=xvacuum and y>=yvacuum) else 0.
    if Main.geometry == "3Dcartesian":
        f = lambda x,y,z,t=0: value if (x>=xvacuum and y>=yvacuum and z>=zvacuum) else 0.
    f.profileName = "constant"
    f.value   = value
    f.xvacuum = xvacuum
    f.yvacuum = yvacuum
    f.zvacuum = zvacuum
    return f
constant._reserved = True

def trapezoidal(max,
                xvacuum=0., xplateau=None, xslope1=0., xslope2=0.,
                yvacuum=0., yplateau=None, yslope1=0., yslope2=0.,
                zvacuum=0., zplateau=None, zslope1=0., zslope2=0. ):
    global Main
    if len(Main)==0:
        raise Exception("trapezoidal profile has been defined before `Main()`")
    if len(Main.grid_length)>0 and xplateau is None: xplateau = Main.grid_length[0]-xvacuum
    if len(Main.grid_length)>1 and yplateau is None: yplateau = Main.grid_length[1]-yvacuum
    if len(Main.grid_length)>2 and zplateau is None: zplateau = Main.grid_length[2]-zvacuum
    def trapeze(max, vacuum, plateau, slope1, slope2):
        def f(position):
            # vacuum region
            if position < vacuum: return 0.
            # linearly increasing density
            elif position < vacuum+slope1: return max*(position-vacuum) / slope1
            # density plateau
            elif position < vacuum+slope1+plateau: return max
            # linearly decreasing density
            elif position < vacuum+slope1+plateau+slope2:
                return max*(1. - ( position - (vacuum+slope1+plateau) ) / slope2)
            # beyond the plasma
            else: return 0.0
        return f
    if   Main.geometry == "1Dcartesian": dim = 1
    elif (Main.geometry == "2Dcartesian" or Main.geometry == "AMcylindrical"): dim = 2
    elif Main.geometry == "3Dcartesian": dim = 3
    fx = trapeze(max, xvacuum, xplateau, xslope1, xslope2)
    f = fx
    if dim > 1:
        fy = trapeze(1. , yvacuum, yplateau, yslope1, yslope2)
        f = lambda x,y: fx(x)*fy(y)
    if dim > 2:
        fz = trapeze(1. , zvacuum, zplateau, zslope1, zslope2)
        f = lambda x,y,z: fx(x)*fy(y)*fz(z)
    f.profileName = "trapezoidal"
    f.value    = max
    f.xvacuum  = xvacuum
    f.xplateau = xplateau
    f.xslope1  = xslope1
    f.xslope2  = xslope2
    if dim > 1:
        f.yvacuum  = yvacuum
        f.yplateau = yplateau
        f.yslope1  = yslope1
        f.yslope2  = yslope2
    if dim > 2:
        f.zvacuum  = zvacuum
        f.zplateau = zplateau
        f.zslope1  = zslope1
        f.zslope2  = zslope2
    return f

def gaussian(max,
             xvacuum=0., xlength=float("inf"), xfwhm=None, xcenter=None, xorder=2,
             yvacuum=0., ylength=float("inf"), yfwhm=None, ycenter=None, yorder=2,
             zvacuum=0., zlength=float("inf"), zfwhm=None, zcenter=None, zorder=2 ):
    import math
    global Main
    if len(Main)==0:
        raise Exception("gaussian profile has been defined before `Main()`")
    if len(Main.grid_length)>0:
        if xlength is None: xlength = Main.grid_length[0]-xvacuum
        if xfwhm   is None: xfwhm   = (Main.grid_length[0]-xvacuum)/3.
        if xcenter is None: xcenter = xvacuum + (Main.grid_length[0]-xvacuum)/2.
    if len(Main.grid_length)>1:
        if ylength is None: ylength = Main.grid_length[1]-yvacuum
        if yfwhm   is None: yfwhm   = (Main.grid_length[1]-yvacuum)/3.
        if ycenter is None: ycenter = yvacuum + (Main.grid_length[1]-yvacuum)/2.
    if len(Main.grid_length)>2:
        if zlength is None: zlength = Main.grid_length[2]-zvacuum
        if zfwhm   is None: zfwhm   = (Main.grid_length[2]-zvacuum)/3.
        if zcenter is None: zcenter = zvacuum + (Main.grid_length[2]-zvacuum)/2.
    def gauss(max, vacuum, length, sigma, center, order):
        def f(position):
            if order == 0: return max
            # vacuum region
            if position < vacuum: return 0.
            # gaussian
            elif position < vacuum+length: return max*math.exp( -(position-center)**order / sigma )
            # beyond
            else: return 0.0
        return f
    if Main.geometry == "1Dcartesian": dim = 1
    if (Main.geometry == "2Dcartesian" or Main.geometry == "AMcylindrical"): dim = 2
    if Main.geometry == "3Dcartesian": dim = 3
    xsigma = (0.5*xfwhm)**xorder/math.log(2.0)
    fx = gauss(max, xvacuum, xlength, xsigma, xcenter, xorder)
    f = fx
    if dim > 1:
        ysigma = (0.5*yfwhm)**yorder/math.log(2.0)
        fy = gauss(1., yvacuum, ylength, ysigma, ycenter, yorder)
        f = lambda x,y: fx(x)*fy(y)
    if dim > 2:
        zsigma = (0.5*zfwhm)**zorder/math.log(2.0)
        fz = gauss(1., zvacuum, zlength, zsigma, zcenter, zorder)
        f = lambda x,y,z: fx(x)*fy(y)*fz(z)
    f.profileName = "gaussian"
    f.value   = max
    f.xvacuum = xvacuum
    f.xlength = xlength
    f.xsigma  = xsigma
    f.xcenter = xcenter
    f.xorder  = xorder
    if dim > 1:
        f.yvacuum = yvacuum
        f.ylength = ylength
        f.ysigma  = ysigma
        f.ycenter = ycenter
        f.yorder  = yorder
    if dim > 2:
        f.zvacuum = zvacuum
        f.zlength = zlength
        f.zsigma  = zsigma
        f.zcenter = zcenter
        f.zorder  = zorder
    return f


def polygonal(xpoints=[], xvalues=[]):
    global Main
    if len(Main)==0:
        raise Exception("polygonal profile has been defined before `Main()`")
    if len(xpoints)!=len(xvalues):
        raise Exception("polygonal profile requires as many points as values")
    if len(Main.grid_length)>0 and len(xpoints)==0:
        xpoints = [0., Main.grid_length[0]]
        xvalues = [1., 1.]
    N = len(xpoints)
    xpoints = [float(x) for x in xpoints]
    xvalues = [float(x) for x in xvalues]
    xslopes = [0. for i in range(1,N)]
    for i in range(1,N):
        if xpoints[i] == xpoints[i-1]: continue
        xslopes[i-1] = (xvalues[i]-xvalues[i-1])/(xpoints[i]-xpoints[i-1])
    def f(x,y=0.,z=0.):
        if x < xpoints[0]: return 0.0;
        for i in range(1,N):
            if x<xpoints[i]: return xvalues[i-1] + xslopes[i-1] * ( x-xpoints[i-1] )
        return 0.
    f.profileName = "polygonal"
    f.xpoints = xpoints
    f.xvalues = xvalues
    f.xslopes = xslopes
    return f

def cosine(base,
           xamplitude=1., xvacuum=0., xlength=None, xphi=0., xnumber=2,
           yamplitude=1., yvacuum=0., ylength=None, yphi=0., ynumber=2,
           zamplitude=1., zvacuum=0., zlength=None, zphi=0., znumber=2):
    import math
    global Main
    if len(Main)==0:
        raise Exception("cosine profile has been defined before `Main()`")

    if len(Main.grid_length)>0 and xlength is None: xlength = Main.grid_length[0]-xvacuum
    if len(Main.grid_length)>1 and ylength is None: ylength = Main.grid_length[1]-yvacuum
    if len(Main.grid_length)>2 and zlength is None: zlength = Main.grid_length[2]-zvacuum

    def cos(base, amplitude, vacuum, length, phi, number):
        def f(position):
            #vacuum region
            if position < vacuum: return 0.
            # profile region
            elif position < vacuum+length:
                return base + amplitude * math.cos(phi + 2.*math.pi * number * (position-vacuum)/length)
            # beyond
            else: return 0.
        return f
    if Main.geometry == "1Dcartesian": dim = 1
    if (Main.geometry == "2Dcartesian" or Main.geometry == "AMcylindrical"): dim = 2
    if Main.geometry == "3Dcartesian": dim = 3
    fx = cos(base, xamplitude, xvacuum, xlength, xphi, xnumber)
    f = fx
    if dim > 1:
        fy = cos(base, yamplitude, yvacuum, ylength, yphi, ynumber)
        f = lambda x,y: fx(x)*fy(y)
    if dim > 2:
        fz = cos(base, zamplitude, zvacuum, zlength, zphi, znumber)
        f = lambda x,y,z: fx(x)*fy(y)*fz(z)
    f.profileName = "cosine"
    f.base        = base
    f.xamplitude  = xamplitude
    f.xvacuum     = xvacuum
    f.xlength     = xlength
    f.xphi        = xphi
    f.xnumber     = float(xnumber)
    if dim > 1:
        f.yamplitude  = yamplitude
        f.yvacuum     = yvacuum
        f.ylength     = ylength
        f.yphi        = yphi
        f.ynumber     = float(ynumber)
    if dim > 2:
        f.zamplitude  = zamplitude
        f.zvacuum     = zvacuum
        f.zlength     = zlength
        f.zphi        = zphi
        f.znumber     = float(znumber)
    return f

def polynomial(**kwargs):
    global Main
    if len(Main)==0:
        raise Exception("polynomial profile has been defined before `Main()`")
    x0 = 0.
    y0 = 0.
    z0 = 0.
    coeffs = dict()
    for k, a in kwargs.items():
        if   k=="x0":
            x0 = a
        elif k=="y0":
            y0 = a
        elif k=="z0":
            z0 = a
        elif k[:5]=="order":
            if type(a) is not list: a = [a]
            order = int(k[5:])
            coeffs[ order ] = a
            if Main.geometry=="1Dcartesian":
                if len(a)!=1:
                    raise Exception("1D polynomial profile must have one coefficient at order "+str(order))
            elif (Main.geometry=="2Dcartesian" or Main.geometry == "AMcylindrical"):
                if len(a)!=order+1:
                    raise Exception("2D polynomial profile must have "+str(order+1)+" coefficients at order "+str(order))
            elif Main.geometry=="3Dcartesian":
                if len(a)!=(order+1)*(order+2)/2:
                    raise Exception("3D polynomial profile must have "+str((order+1)*(order+2)/2)+" coefficients at order "+str(order))
    if Main.geometry=="1Dcartesian":
        def f(x):
            r = 0.
            xx0 = x-x0
            xx = 1.
            currentOrder = 0
            for order, c in sorted(coeffs.items()):
                while currentOrder<order:
                    currentOrder += 1
                    xx *= xx0
                r += c[0] * xx
            return r
    elif (Main.geometry=="2Dcartesian" or Main.geometry == "AMcylindrical"):
        def f(x,y):
            r = 0.
            xx0 = x-x0
            yy0 = y-y0
            xx = [1.]
            currentOrder = 0
            for order, c in sorted(coeffs.items()):
                while currentOrder<order:
                    currentOrder += 1
                    yy = xx[-1]*yy0
                    xx = [ xxx * xx0 for xxx in xx ] + [yy]
                for i in range(order+1): r += c[i]*xx[i]
            return r
    elif Main.geometry=="3Dcartesian":
        def f(x,y,z):
            r = 0.
            xx0 = x-x0
            yy0 = y-y0
            zz0 = z-z0
            xx = [1.]
            currentOrder = 0
            for order, c in sorted(coeffs.items()):
                while currentOrder<order:
                    currentOrder += 1
                    zz = xx[-1]*zz0
                    yy = [ xxx * yy0 for xxx in xx[-currentOrder-1:] ] + [zz]
                    xx = [ xxx * xx0 for xxx in xx ] + yy
                for i in range(len(c)): r += c[i]*xx[i]
            return r
    else:
        raise Exception("polynomial profiles are not available in this geometry yet")
    f.profileName = "polynomial"
    f.x0 = x0
    f.y0 = y0
    f.z0 = z0
    f.orders = []
    f.coeffs = []
    for order, c in sorted(coeffs.items()):
        f.orders.append( order )
        f.coeffs.append( c     )
    return f



def tconstant(start=0.):
    def f(t):
        return 1. if t>=start else 0.
    f.profileName = "tconstant"
    f.start       = start
    return f
tconstant._reserved = True

def ttrapezoidal(start=0., plateau=None, slope1=0., slope2=0.):
    global Main
    if len(Main)==0:
        raise Exception("ttrapezoidal profile has been defined before `Main()`")
    if plateau is None: plateau = Main.simulation_time - start
    def f(t):
        if t < start: return 0.
        elif t < start+slope1: return (t-start) / slope1
        elif t < start+slope1+plateau: return 1.
        elif t < start+slope1+plateau+slope2:
            return 1. - ( t - (start+slope1+plateau) ) / slope2
        else: return 0.0
    f.profileName = "ttrapezoidal"
    f.start       = start
    f.plateau     = plateau
    f.slope1      = slope1
    f.slope2      = slope2
    return f

def tgaussian(start=0., duration=None, fwhm=None, center=None, order=2):
    import math
    global Main
    if len(Main)==0:
        raise Exception("tgaussian profile has been defined before `Main()`")
    if duration is None: duration = Main.simulation_time-start
    if fwhm     is None: fwhm     = duration/3.
    if center   is None: center   = start + duration/2.
    sigma = (0.5*fwhm)**order/math.log(2.0)
    def f(t):
        if t < start: return 0.
        elif t < start+duration: return math.exp( -(t-center)**order / sigma )
        else: return 0.0
    f.profileName = "tgaussian"
    f.start       = start
    f.duration    = duration
    f.sigma       = sigma
    f.center      = center
    f.order       = order
    return f

def tpolygonal(points=[], values=[]):
    global Main
    if len(Main)==0:
        raise Exception("tpolygonal profile has been defined before `Main()`")
    if len(points)==0:
        points = [0., Main.simulation_time]
        values = [1., 1.]
    N = len(points)
    points = [float(x) for x in points]
    values = [float(x) for x in values]
    slopes = [0. for i in range(1,N)]
    for i in range(1,N):
        if points[i] == points[i-1]: continue
        slopes[i-1] = (values[i]-values[i-1])/(points[i]-points[i-1])
    def f(t):
        if t < points[0]: return 0.0;
        for i in range(1,N):
            if t<points[i]: return values[i-1] + slopes[i-1] * ( t-points[i-1] )
        return 0.
    f.profileName = "tpolygonal"
    f.points      = points
    f.values      = values
    f.slopes      = slopes
    return f

def tcosine(base=0., amplitude=1., start=0., duration=None, phi=0., freq=1.):
    import math
    global Main
    if len(Main)==0:
        raise Exception("tcosine profile has been defined before `Main()`")
    if duration is None: duration = Main.simulation_time-start
    def f(t):
        if t < start: return 0.
        elif t < start+duration:
            return base + amplitude * math.cos(phi + freq*(t-start))
        else: return 0.
    f.profileName = "tcosine"
    f.base        = base
    f.amplitude   = amplitude
    f.start       = start
    f.duration    = duration
    f.phi         = phi
    f.freq        = freq
    return f

def tpolynomial(**kwargs):
    t0 = 0.
    coeffs = dict()
    for k, a in kwargs.items():
        if   k=="t0":
            t0 = a
        elif k[:5]=="order":
            order = int(k[5:])
            try: coeffs[ order ] = a*1.
            except: raise Exception("tpolynomial profile must have one coefficient per order")
    def f(t):
        r = 0.
        tt0 = t-t0
        tt = 1.
        currentOrder = 0
        for order, c in sorted(coeffs.items()):
            while currentOrder<order:
                currentOrder += 1
                tt *= tt0
            r += c * tt
        return r
    f.profileName = "tpolynomial"
    f.t0 = t0
    f.orders = []
    f.coeffs = []
    for order, c in sorted(coeffs.items()):
        f.orders.append( order )
        f.coeffs.append( c     )
    return f

def tsin2plateau(start=0., fwhm=0., plateau=None, slope1=None, slope2=None):
    import math
    global Main
    if len(Main)==0:
        raise Exception("tsin2plateau profile has been defined before `Main()`")
    if plateau is None: plateau = 0 # default is a simple sin2 profile (could be used for a 2D or 3D laserPulse too)
    if slope1 is None: slope1 = fwhm
    if slope2 is None: slope2 = slope1
    def f(t):
        if t < start:
            return 0.
        elif (t < start+slope1) and (slope1!=0.):
            return math.pow( math.sin(0.5*math.pi*(t-start)/slope1) , 2 )
        elif t < start+slope1+plateau:
            return 1.
        elif t < start+slope1+plateau+slope2 and (slope2!=0.):
            return math.pow(  math.cos(0.5*math.pi*(t-start-slope1-plateau)/slope2) , 2 )
        else:
            return 0.
    f.profileName = "tsin2plateau"
    f.start       = start
    #f.fwhm        = fwhm
    f.plateau     = plateau
    f.slope1      = slope1
    f.slope2      = slope2
    return f


def transformPolarization(polarization_phi, ellipticity):
    from math import pi, sqrt, sin, cos, tan, atan2
    e2 = ellipticity**2
    p = (1.-e2)*sin(2.*polarization_phi)/2.
    dephasing = atan2(ellipticity, p)
    amplitude = sqrt(1./(1.+e2))
    c2 = cos(polarization_phi)**2
    s2 = 1. - c2
    amplitudeY = amplitude * sqrt(c2 + e2*s2)
    amplitudeZ = amplitude * sqrt(s2 + e2*c2)
    return [dephasing, amplitudeY, amplitudeZ]

def LaserPlanar1D( box_side="xmin", a0=1., omega=1.,
        polarization_phi=0., ellipticity=0., time_envelope=tconstant(),phase_zero=0.):
    import math
    # Polarization and amplitude
    [dephasing, amplitudeY, amplitudeZ] = transformPolarization(polarization_phi, ellipticity)
    amplitudeY *= a0 * omega
    amplitudeZ *= a0 * omega
    # Create Laser
    Laser(
        box_side        = box_side,
        omega          = omega,
        chirp_profile  = tconstant(),
        time_envelope  = time_envelope,
        space_envelope = [ amplitudeZ, amplitudeY ],
        phase          = [ dephasing-phase_zero, -phase_zero ],
        delay_phase    = [ 0., dephasing ]
    )

def LaserEnvelopePlanar1D( a0=1., omega=1., focus=None, time_envelope=tconstant(),
        envelope_solver = "explicit",Envelope_boundary_conditions = [["reflective"]],
        polarization_phi = 0.,ellipticity = 0.):
    import cmath
    from numpy import vectorize, sqrt

    def space_time_envelope(x,t):
        polarization_amplitude_factor = 1/sqrt(1.+ellipticity**2)
        return (a0*polarization_amplitude_factor) * complex( vectorize(time_envelope)(t) )

    # Create Laser Envelope
    LaserEnvelope(
        omega               = omega,
        envelope_profile    = space_time_envelope,
        envelope_solver     = envelope_solver,
        Envelope_boundary_conditions = Envelope_boundary_conditions,
        polarization_phi    = polarization_phi,
        ellipticity         = ellipticity
    )


def LaserGaussian2D( box_side="xmin", a0=1., omega=1., focus=None, waist=3., incidence_angle=0.,
        polarization_phi=0., ellipticity=0., time_envelope=tconstant(), phase_zero=0.):
    from math import pi, cos, sin, tan, atan, sqrt, exp
    global Main
    if len(Main)==0:
        raise Exception("LaserGaussian2D profile has been defined before `Main()`")
    grid_length = Main.grid_length
    # Polarization and amplitude
    dephasing, amplitudeZ, amplitudeY = transformPolarization(polarization_phi, ellipticity)
    amplitudeY *= a0 * omega
    amplitudeZ *= a0 * omega
    delay_phase = [0., dephasing]
    # Injection on ymin/ymax
    if box_side[0] == "y":
        focus = focus[::-1]
        grid_length = grid_length[::-1]
        amplitudeY = -amplitudeY
    # Injection on max boundary
    if box_side.endswith("max"):
        focus[0] = grid_length[0] - focus[0]
    # Space and phase envelopes
    Zr = omega * waist**2/2.
    if incidence_angle == 0.:
        w  = sqrt(1./(1.+(focus[0]/Zr)**2))
        invWaist2 = (w/waist)**2
        coeff = -omega * focus[0] * w**2 / (2.*Zr**2)
        def spatial(y):
            return sqrt(w) * exp( -invWaist2*(y-focus[1])**2 )
        def phase(y):
            return coeff * (y-focus[1])**2
    else:
        invZr  = sin(incidence_angle) / Zr
        invZr2 = invZr**2
        invZr3 = (cos(incidence_angle) / Zr)**2 / 2.
        invWaist2 = (cos(incidence_angle) / waist)**2
        omega_ = omega * sin(incidence_angle)
        Y1 = focus[1] + focus[0]/tan(incidence_angle)
        Y2 = focus[1] - focus[0]*tan(incidence_angle)
        amplitudeY *= cos(incidence_angle)
        def spatial(y):
            w2 = 1./(1. + invZr2*(y-Y1)**2)
            return sqrt(sqrt(w2)) * exp( -invWaist2*w2*(y-Y2)**2 )
        def phase(y):
            dy = y-Y1
            return omega_*dy*(1.+ invZr3*(y-Y2)**2/(1.+invZr2*dy**2)) - 0.5*atan(invZr*dy)
        # Adjust the phase to match that of a laser that could come from another face
        if Y2 < 0:
            distance_to_boundary = focus[1] / sin(incidence_angle)
        elif Y2 < grid_length[1]:
            distance_to_boundary = focus[0] / cos(incidence_angle)
        else:
            distance_to_boundary = (focus[1] - grid_length[1]) / sin(incidence_angle)
        phase_zero -= omega * distance_to_boundary - 0.5*atan(distance_to_boundary/Zr)
    # Create Laser
    Laser(
        box_side       = box_side,
        omega          = omega,
        chirp_profile  = tconstant(),
        time_envelope  = time_envelope,
        space_envelope = [ lambda y:amplitudeY*spatial(y), lambda y:amplitudeZ*spatial(y) ],
        phase          = [ lambda y:phase(y)-phase_zero+delay_phase[1], lambda y:phase(y)-phase_zero+delay_phase[0] ],
        delay_phase    = delay_phase
    )

def LaserEnvelopeGaussian2D( a0=1., omega=1., focus=None, waist=3., time_envelope=tconstant(),
        envelope_solver = "explicit",Envelope_boundary_conditions = [["reflective"]],
        polarization_phi = 0.,ellipticity = 0.):
    import cmath
    from numpy import exp, sqrt, arctan, vectorize

    def gaussian_beam_with_temporal_profile(x,y,t):
        polarization_amplitude_factor = 1/sqrt(1.+ellipticity**2)
        Zr = omega * waist**2/2.
        w  = sqrt(1./(1.+   ( (x-focus[0])/Zr  )**2 ) )
        coeff = omega * (x-focus[0]) * w**2 / (2.*Zr**2)
        phase = coeff * ( (y-focus[1])**2 )
        exponential_with_total_phase = exp(1j*(phase-arctan( (x-focus[0])/Zr )))
        invWaist2 = (w/waist)**2
        spatial_amplitude = a0 *polarization_amplitude_factor * sqrt(w) * exp( -invWaist2*(y-focus[1])**2)
        space_time_envelope = spatial_amplitude * vectorize(time_envelope)(t)
        return space_time_envelope * exponential_with_total_phase

    # Create Laser Envelope
    LaserEnvelope(
        omega               = omega,
        envelope_profile    = gaussian_beam_with_temporal_profile,
        envelope_solver     = envelope_solver,
        Envelope_boundary_conditions = Envelope_boundary_conditions,
        polarization_phi    = polarization_phi,
        ellipticity         = ellipticity
    )

def LaserGaussian3D( box_side="xmin", a0=1., omega=1., focus=None, waist=3., incidence_angle=[0.,0.],
        polarization_phi=0., ellipticity=0., time_envelope=tconstant(), phase_zero=0.):
    from math import pi, cos, sin, tan, atan, sqrt, exp
    global Main
    if len(Main)==0:
        raise Exception("LaserGaussian3D profile has been defined before `Main()`")
    grid_length = Main.grid_length
    # Polarization and amplitude
    [dephasing, amplitudeZ, amplitudeY] = transformPolarization(polarization_phi, ellipticity)
    amplitudeY *= a0 * omega
    amplitudeZ *= a0 * omega
    # Injection on ymin/ymax or zmin/zmax
    if box_side[0] == "y":
        focus = [focus[1],focus[0],focus[2]]
        grid_length = [grid_length[1],grid_length[0],grid_length[2]]
        amplitudeY = -amplitudeY
    elif box_side[0] == "z":
        focus = [focus[2],focus[0],focus[1]]
        grid_length = [grid_length[2],grid_length[0],grid_length[1]]
    # Injection on max boundary
    if box_side.endswith("max"):
        focus[0] = grid_length[0] - focus[0]
    # Space and phase envelopes
    Zr = omega * waist**2/2.
    if incidence_angle == [0.,0.]:
        w  = sqrt(1./(1.+(focus[0]/Zr)**2))
        invWaist2 = (w/waist)**2
        coeff = -omega * focus[0] * w**2 / (2.*Zr**2)
        def spatial(y,z):
            return w * exp( -invWaist2*((y-focus[1])**2 + (z-focus[2])**2 )  )
        def phase(y,z):
            return coeff * ( (y-focus[1])**2 + (z-focus[2])**2 )
    else:
        invZr = 1./Zr
        invW  = 1./waist
        alpha = omega * Zr
        cy = cos(incidence_angle[0]); sy = sin(incidence_angle[0])
        cz = cos(incidence_angle[1]); sz = sin(incidence_angle[1])
        cycz = cy*cz; cysz = cy*sz; sycz = sy*cz; sysz = sy*sz
        amplitudeY = sysz * amplitudeZ + cy * amplitudeY
        amplitudeZ *= cz
        def spatial(y,z):
            X = invZr * (-focus[0]*cycz + (y-focus[1])*cysz - (z-focus[2])*sy )
            Y = invW  * ( focus[0]*sz   + (y-focus[1])*cz                     )
            Z = invW  * (-focus[0]*sycz + (y-focus[1])*sysz + (z-focus[2])*cy )
            invW2 = 1./(1.+X**2)
            return sqrt(invW2) * exp(-(Y**2+Z**2)*invW2)
        def phase(y,z):
            X = invZr * (-focus[0]*cycz + (y-focus[1])*cysz - (z-focus[2])*sy )
            Y = invZr * ( focus[0]*sz   + (y-focus[1])*cz                     )
            Z = invZr * (-focus[0]*sycz + (y-focus[1])*sysz + (z-focus[2])*cy )
            return alpha * X*(1.+0.5*(Y**2+Z**2)/(1.+X**2)) - atan(X)
        # Adjust the phase to match that of a laser that could come from another face
        faces = (focus[0],focus[1],focus[2],focus[1]-grid_length[1],focus[2]-grid_length[2])
        denominators = (cycz, cysz, -sy, cysz, -sy)
        distance_to_boundary = min([N/D for N,D in zip(faces,denominators) if D != 0 and N/D > 0])
        phase_zero -= omega * distance_to_boundary - atan(distance_to_boundary/Zr)
    # Create Laser
    Laser(
        box_side       = box_side,
        omega          = omega,
        chirp_profile  = tconstant(),
        time_envelope  = time_envelope,
        space_envelope = [ lambda y,z:amplitudeY*spatial(y,z), lambda y,z:amplitudeZ*spatial(y,z) ],
        phase          = [ lambda y,z:phase(y,z)-phase_zero+dephasing, lambda y,z:phase(y,z)-phase_zero ],
        delay_phase    = [ 0., dephasing ]
    )

def LaserEnvelopeGaussian3D( a0=1., omega=1., focus=None, waist=3., time_envelope=tconstant(),
        envelope_solver = "explicit",Envelope_boundary_conditions = [["reflective"]],
        polarization_phi = 0.,ellipticity = 0.):
    import cmath
    from numpy import exp, sqrt, arctan, vectorize

    def gaussian_beam_with_temporal_profile(x,y,z,t):
        polarization_amplitude_factor = 1/sqrt(1.+ellipticity**2)
        Zr = omega * waist**2/2.
        w  = sqrt(1./(1.+   ( (x-focus[0])/Zr  )**2 ) )
        coeff = omega * (x-focus[0]) * w**2 / (2.*Zr**2)
        phase = coeff * ( (y-focus[1])**2 + (z-focus[2])**2 )
        exponential_with_total_phase = exp(1j*(phase-arctan( (x-focus[0])/Zr )))
        invWaist2 = (w/waist)**2
        spatial_amplitude = a0*polarization_amplitude_factor* w * exp( -invWaist2*(  (y-focus[1])**2 + (z-focus[2])**2 )  )
        space_time_envelope = spatial_amplitude * vectorize(time_envelope)(t)
        return space_time_envelope * exponential_with_total_phase

    # Create Laser Envelope
    LaserEnvelope(
        omega               = omega,
        envelope_profile    = gaussian_beam_with_temporal_profile,
        envelope_solver     = envelope_solver,
        Envelope_boundary_conditions = Envelope_boundary_conditions,
        polarization_phi    = polarization_phi,
        ellipticity         = ellipticity
    )


def LaserGaussianAM( box_side="xmin", a0=1., omega=1., focus=None, waist=3.,
        polarization_phi=0., ellipticity=0., time_envelope=tconstant(), phase_zero=0.):
    from math import cos, sin, tan, atan, sqrt, exp
    assert len(focus)==2, "LaserGaussianAM: focus must be a list of length 2."
    # Polarization and amplitude
    [dephasing, amplitudeY, amplitudeZ] = transformPolarization(polarization_phi, ellipticity)
    amplitudeY *= a0 * omega
    amplitudeZ *= a0 * omega
    # Space and phase envelopes
    Zr = omega * waist**2/2.
    w  = sqrt(1./(1.+(focus[0]/Zr)**2))
    invWaist2 = (w/waist)**2
    coeff = -omega * focus[0] * w**2 / (2.*Zr**2)
    def spatial(y):
        return w * exp( -invWaist2*(y-focus[1])**2 )
    def phase(y):
        return coeff * (y-focus[1])**2
    # Create Laser
    Laser(
        box_side        = box_side,
        omega          = omega,
        chirp_profile  = tconstant(),
        time_envelope  = time_envelope,
        space_envelope = [ lambda y:amplitudeZ*spatial(y), lambda y:amplitudeY*spatial(y) ],
        phase          = [ lambda y:phase(y)-phase_zero+dephasing, lambda y:phase(y)-phase_zero ],
        delay_phase    = [ 0., dephasing ]
    )


def LaserEnvelopeGaussianAM( a0=1., omega=1., focus=None, waist=3., time_envelope=tconstant(),
        envelope_solver = "explicit",Envelope_boundary_conditions = [["reflective"]],
        Env_pml_sigma_parameters = [[0.90,2],[10.0,2],[10.0,2]],
        Env_pml_kappa_parameters = [[1.00,1.00,2],[1.00,1.00,2],[1.00,1.00,2]],
        Env_pml_alpha_parameters = [[0.90,0.90,1],[0.75,0.75,1],[0.75,0.75,1]],
        polarization_phi = 0.,ellipticity = 0.):
    import cmath
    from numpy import exp, sqrt, arctan, vectorize

    def gaussian_beam_with_temporal_profile(x,r,t):
        polarization_amplitude_factor = 1/sqrt(1.+ellipticity**2)
        Zr = omega * waist**2/2.
        w  = sqrt(1./(1.+   ( (x-focus[0])/Zr  )**2 ) )
        coeff = omega * (x-focus[0]) * w**2 / (2.*Zr**2)
        phase = coeff * ( r**2 )
        exponential_with_total_phase = exp(1j*(phase-arctan( (x-focus[0])/Zr )))
        invWaist2 = (w/waist)**2
        spatial_amplitude = a0 * polarization_amplitude_factor * w * exp( -invWaist2*(  r**2  ) )
        space_time_envelope = spatial_amplitude * vectorize(time_envelope)(t)
        return space_time_envelope * exponential_with_total_phase

    # Create Laser Envelope
    LaserEnvelope(
        omega               = omega,
        envelope_profile    = gaussian_beam_with_temporal_profile,
        envelope_solver     = envelope_solver,
        Envelope_boundary_conditions = Envelope_boundary_conditions,
        Env_pml_sigma_parameters = Env_pml_sigma_parameters,
        Env_pml_kappa_parameters = Env_pml_kappa_parameters,
        Env_pml_alpha_parameters = Env_pml_alpha_parameters,
        polarization_phi    = polarization_phi,
        ellipticity         = ellipticity
    )

# Define the tools for the propagation of a laser profile
try:
    import numpy as np
    
    _N_LaserOffset = 0
    
    def LaserOffset(box_side="xmin", space_time_profile=[], offset=0., angle=0., extra_envelope=lambda *a:1.,
            fft_time_window=None, fft_time_step=None, keep_n_strongest_modes=100,
            number_of_processes=None, file=None):
        global _N_LaserOffset
        
        file_ = file or ('LaserOffset'+str(_N_LaserOffset)+'.h5')
        
        L = Laser(
            box_side = box_side,
            file = file_,
        )
        
        L._offset = offset
        L._extra_envelope = extra_envelope
        L._profiles = space_time_profile
        L._fft_time_window = fft_time_window or Main.simulation_time
        L._fft_time_step = fft_time_step or Main.timestep
        L._keep_n_strongest_modes = keep_n_strongest_modes
        L._angle = angle
        L._number_of_processes = number_of_processes
        if file:
            if not os.path.exists(file):
                raise Exception("File not found or not accessible: "+file)
            L._propagate = False
        else:
            L._propagate = True
        
        _N_LaserOffset += 1

except:
    
    def LaserOffset(box_side="xmin", space_time_profile=[], offset=0., fft_time_window=None, extra_envelope=lambda *a:1., keep_n_strongest_modes=100, angle=0., number_of_processes=None, file=None):
        L = Laser(
            box_side = box_side,
            file = "none",
            time_envelope = extra_envelope
        )
        print("WARNING: LaserOffset unavailable because numpy was not found")



"""
-----------------------------------------------------------------------
BEGINNING OF THE USER NAMELIST
"""

###### Laser Wakefield Acceleration tutorial in Azimuthal Modes cylindrical geometry
import math 
import numpy as np
import scipy.constants 
from math import *
from scipy import *

##### Physical constants
lambda0             = 0.8e-6                    # reference length, m - here it will be our plasma wavelength
c                   = scipy.constants.c         # lightspeed, m/s
omega0              = 2*math.pi*c/lambda0       # reference angular frequency, rad/s
eps0                = scipy.constants.epsilon_0 # Vacuum permittivity, F/m
e                   = scipy.constants.e         # Elementary charge, C
me                  = scipy.constants.m_e       # Electron mass, kg
ncrit               = eps0*omega0**2*me/e**2    # Plasma critical number density, m-3
c_over_omega0       = lambda0/2./math.pi        # converts from c/omega0 units to m
reference_frequency = omega0                    # reference frequency, s-1
E0                  = me*omega0*c/e             # reference electric field, V/m
electron_mass_MeV   = scipy.constants.physical_constants["electron mass energy equivalent in MeV"][0]


##### Variables used for unit conversions
c_normalized        = 1.                        # speed of light in vacuum in normalized units
um                  = 1.e-6/c_over_omega0       # 1 micron in normalized units
mm                  = 1.e-3/c_over_omega0       # 1 millimetre in normalized units
fs                  = 1.e-15*omega0             # 1 femtosecond in normalized units
mm_mrad             = um                        # 1 millimetre-milliradians in normalized units
pC                  = 1.e-12/e                  # 1 picoCoulomb in normalized units
MeV                 = 1./electron_mass_MeV      # 1 MeV in normalized units


##### Mesh and time evolution
dx                  = 0.0625*um                 # longitudinal resolution
dr                  = 0.0625*um                    # radial resolution

nx                  = 960                      # number of grid points in the longitudinal direction
nr                  = 368                        # number of grid points in the radial direction

Lx                  = nx * dx                   # longitudinal size of the simulation window
Lr                  = nr * dr                   # radial size of the simulation window, which goes from r=0 to r=Lr

npatch_x            = 32
npatch_r            = 16

dt                  = 0.44*dx/c_normalized      # integration timeestep
#Niterations         = 15000

grid_x_min = - Lx + 10*um
grid_r_min = -Lr/2

# --- PLASMA CONSTANTS AND DEFINITIONS---
# Plasma Density
n0 = 1e19 #/ cc
densityramplength = 100 * um  # upramp at beginning of plasma, centered around x=0
# Density down ramp for injection
ddrx = 120 * um  # Downramp centered around this point
ddrlength = 30 * um
dldz = 0.05  # dlambda/dz: fuer 5 prozent setze dldz=0.05
rampc0 = 2*pi*c*np.sqrt(me*eps0)/e  # 33.3e6, in Downramp verwendet
#  Dichten vor (initial) und hinter (final) der Downramp
ni = rampc0**2 * n0 / (rampc0 - (dldz*sqrt(n0)*ddrlength/2)*c_over_omega0**2)**2
nf = rampc0**2 * n0 / (rampc0 + (dldz*sqrt(n0)*ddrlength/2)*c_over_omega0**2)**2
# Plasma frequenz
omegap = sqrt(n0*1e6*e**2/eps0/me) #1e6 is multiplied because we need n0 in m-3 unit

# --- DRIVING LASER PULSE ---
# wavelength
las_lambda = 0.8 * um
#las_omega not used for laser, but for calculating the crtical density at particle setup
las_omega = 2 * pi * c / las_lambda

# peak a0 for a linear pulse - for elliptical pulses this will be reduced appropriately
las_a0_lpequiv = 4

# laser polarisation angle (radians)- 0 is polarised in y dimension
# rotates anti-clockwise as viewed from the laser source
las_polangle = 0

# laser fwhm transverse spotsize in focus
las_fwhm = 6 * um

# pulse duration (fwhm)
las_fwhmt = 20 * fs

#at las_focus_time: pulse maximum arrives at coordinate origin. make sure this is late enough for appropriate injection
las_focus_time = - grid_x_min/1. + las_fwhmt * 2

# laser carrier envelope phase - 0 should mean cycle peak is at the envelope peak (still to be checked)
las_cep = 0
  
# Moving Window
window_vx = sqrt(1 - omegap**2/omega0**2)


# --- DERIVED CONSTANTS --- DO NOT TOUCH ---
# only change things here if you know what you are doing!

# conversion of fwhm to 1/e^2 radii
las_widtht = 0.849322 * las_fwhmt
las_width = 0.849322 * las_fwhm

# gaussian beam equations taken from http://en.wikipedia.org/wiki/Gaussian_beam
# rayleigh range
las_rayleigh = pi * las_width ** 2 / las_lambda
# wave number
las_k = 2.0 * pi / las_lambda

# radius of curvature at the injection boundary
def las_rc(x):
    return x * (1 + (las_rayleigh / (x + 1e-20/c_over_omega0))**2)

# Phasefront shift (delay in space) due to non-infinite las_rc
def las_rc_shift(x,r):
    return r**2 / (2 * las_rc + 1e-20/c_over_omega0)

'''# radius of beam 
def las_w(x):
    return las_width * sqrt(1 + (x / las_rayleigh)**2 )

# gouy phase shift at the injection boundary
def las_gouy(x):
    return atan(x / las_rayleigh)'''

# laser intensity at focus calculated from the defined a0 value
las_intensity_focus = las_a0_lpequiv**2 * 1.368167127e6 / (las_lambda*c_over_omega0)**2


##### Boundary conditions
EM_boundary_conditions  = [["PML","PML"],["PML","PML"],]
#EM_boundary_conditions = [ ["silver-muller","silver-muller"],["buneman","buneman"],]

##### B-TIS3 interpolation to cope with numerical Cherenkov radiation
use_BTIS3_interpolation = False

Main(
    geometry                       = "AMcylindrical",
    number_of_AM                   = 2,
    timestep                       = dt,
    simulation_time                = 1400*fs, 
    cell_length                    = [dx, dr],
    grid_length                    = [Lx, Lr],
    number_of_patches              = [npatch_x, npatch_r],
    EM_boundary_conditions         = EM_boundary_conditions,
    number_of_pml_cells            = [[20,20],[20,20]],     
    solve_poisson                  = False,
    print_every                    = 100,
    use_BTIS3_interpolation        = use_BTIS3_interpolation,
    reference_angular_frequency_SI = omega0,
)

#### Define the plasma

# The physics of interest in this case is near the axis, 
# so no need to radially fill all the window with plasma

n0_normalized  = n0*1.e6/ncrit
ni_normalized  = ni*1.e6/ncrit # convert to meters and then normalize
nf_normalized  = nf*1.e6/ncrit
Radius_plasma  = 140.*um

#longitudinal_profile        = polygonal(xpoints=[x_start_plasma,x_density_transition_peak,x_plateau,x_end_plateau,x_end_plasma], 
#                                        xvalues=[0.,2.*density_plateau_normalized,density_plateau_normalized,density_plateau_normalized,0.])
ref_density_list=[0]

def density_profile(x,r):
    global ref_density_list

    if x>-densityramplength/2:
        density=ni_normalized*(x+densityramplength/2)/densityramplength
        
        if density not in ref_density_list:
            ref_density_list.append(density)
        
        return density
    
    elif x > densityramplength/2:
        density=ni_normalized

        if density not in ref_density_list:
            ref_density_list.append(density)

        return density

    elif x > ddrx:
        density= nf_normalized
        
        if density not in ref_density_list:
            ref_density_list.append(density)

        return density

    elif x > ddrx-ddrlength/2 and x < ddrx+ddrlength/2:
        density=(rampc0**2*n0_normalized)/((rampc0+(dldz*sqrt(n0)*(x-ddrx))*(c_over_omega0**2))**2)

        if density not in ref_density_list:
            ref_density_list.append(density)
        
        return density
    
    else:
        return ref_density_list[-1]


'''def nplasma(x,r):
    profile_r = 0.
    if (r<Radius_plasma):
        profile_r = 1.
    return profile_r*density_profile(x,r)'''

if (use_BTIS3_interpolation == False):
    pusher = "boris"
else:
    pusher = "borisBTIS3"
      
Species( 
    name                    = "electron",
    position_initialization = "regular",
    momentum_initialization = "cold",
    particles_per_cell      = 9.,
    regular_number          = [], # i.e. 2 macro-particles along x, 1 macro-particle along r and 8 along the theta angle between 0 and 2*pi
    c_part_max              = 1.0,
    mass                    = 1.0,
    charge                  = -1.0,
    charge_density          = density_profile,
    mean_velocity           = [0.0, 0.0, 0.0],
    pusher                  = pusher,  
    time_frozen             = 0.0,
    boundary_conditions     = [["remove", "remove"],["reflective", "remove"],],
)

def pos(x):
    return x

#### Define the laser

def las_profile(t):
    
    #las_rc = x * (1 + (las_rayleigh / (x + 1e-20/c_over_omega0))**2)
    #las_rc_shift =  r**2 / (2 * las_rc + 1e-20/c_over_omega0)
    
    time = t #- las_rc_shift
    t0   = las_focus_time #+ pos
    profile = exp(-((time-t0)/las_widtht)**2)
    return profile

# Define a Gaussian beam linearly polarized on the y axis 
# (default polarization angle=0, default polarization=linear)
LaserGaussianAM(
    box_side                = "xmin",
    a0                      = 4,
    omega                   = 1.,
    focus                   = [0., 0.],
    waist                   = las_width,
    polarization_phi        = 0.,
    ellipticity             = 0.,
    time_envelope           = tgaussian(center=las_focus_time,fwhm=las_fwhmt)
)

#### Define a CurrentFilter
# CurrentFilter(
#     model                   = "binomial",
#     passes                  = [2],
# )

#### Define the moving window
MovingWindow(
    time_start              = 392*fs,
    velocity_x              = window_vx
)

#### Define the grid diagnostics

DiagFields(
    every                   = 50*fs,
    fields                  = ['El','Er','Rho']
)

fields_probes               = ['Ex','Ey','Rho','Bz']

if (use_BTIS3_interpolation == True):
    fields_probes.append("BzBTIS3")

# 1D Probe defined on a line parallel to the axis of propagation,
# at a distance sqrt(2)*dr from the axis.
# A Probe placed on the axis would be very noisy
'''DiagProbe(
    every                   = int(50*um/dt),
    origin                  = [0., 1*dr, 1*dr],
    corners                 = [[nx*dx, 0, 0],],
    number                  = [nx],
    fields                  = fields_probes
 )'''
 
# 2D Probe, defined on the plane parallel to the polarization axis of the laser, i.e. the xy plane
DiagProbe(
    every                   = 50*fs,
    origin                  = [0., -nr*dr,0.],
    corners                 = [ [nx*dx,-nr*dr,0.], [0,nr*dr,0.] ],
    number                  = [nx, 2*nr],
    fields                  = fields_probes
)


## TrackParticle diagnostic
def my_filter(particles):
    return (np.sqrt(particles.px**2+particles.py**2+particles.pz**2)>30)

DiagTrackParticles(
 species = "electron",
 every = 50*fs,
 #filter = my_filter,
 attributes = ["x", "y", "z", "px", "py", "pz", "q", "w"]
)

"""
    END OF THE USER NAMELIST
-----------------------------------------------------------------------
"""

gc.collect()
import math
from glob import glob
from re import search

def _mkdir(role, path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            raise Exception("ERROR in the namelist: "+role+" "+path+" cannot be created")
    elif not os.path.isdir(path):
        raise Exception("ERROR in the namelist: "+role+" "+path+" exists but is not a directory")

def _prepare_checkpoint_dir():
    # Checkpoint: prepare dir tree
    if smilei_mpi_rank == 0 and (Checkpoints.dump_step>0 or Checkpoints.dump_minutes>0.):
        checkpoint_dir = "." + os.sep + "checkpoints" + os.sep
        if Checkpoints.file_grouping:
            ngroups = int((smilei_mpi_size-1)/Checkpoints.file_grouping + 1)
            ngroups_chars = int(math.log10(ngroups))+1
            for group in range(ngroups):
                group_dir = checkpoint_dir + '%0*d'%(ngroups_chars,group)
                _mkdir("checkpoint", group_dir)
        else:
            _mkdir("checkpoint", checkpoint_dir)

def _smilei_check():
    """Do checks over the script"""
    
    # Verify classes were not overriden
    for CheckClassName in ["SmileiComponent","Species", "Laser","Collisions",
            "DiagProbe","DiagParticleBinning", "DiagScalar","DiagFields",
            "DiagTrackParticles","DiagNewParticles","DiagPerformances",
            "ExternalField","PrescribedField",
            "SmileiSingleton","Main","Checkpoints","LoadBalancing","MovingWindow",
            "RadiationReaction", "ParticleData", "MultiphotonBreitWheeler",
            "Vectorization", "MultipleDecomposition"]:
        CheckClass = globals()[CheckClassName]
        try:
            if not CheckClass._verify: raise Exception("")
        except:
            raise Exception("ERROR in the namelist: it seems that the name `"+CheckClassName+"` has been overriden")
    
    # Checkpoint: Verify the restart_dir and find possible restart file for each rank
    if len(Checkpoints)==1 and Checkpoints.restart_dir:
        if len(Checkpoints.restart_files) == 0 :
            Checkpoints.restart = True
            pattern = Checkpoints.restart_dir + os.sep + "checkpoints" + os.sep
            if Checkpoints.file_grouping:
                pattern += "*"+ os.sep
            pattern += "dump-*-*.h5"
            # pick those file that match the mpi rank
            files = filter(lambda a: smilei_mpi_rank==int(search(r'dump-[0-9]*-([0-9]*).h5$',a).groups()[-1]), glob(pattern))
            
            if Checkpoints.restart_number is not None:
                # pick those file that match the restart_number
                files = filter(lambda a: Checkpoints.restart_number==int(search(r'dump-([0-9]*)-[0-9]*.h5$',a).groups()[-1]), files)
            
            Checkpoints.restart_files = list(files)
            
            if len(Checkpoints.restart_files) == 0:
                raise Exception(
                "ERROR in the namelist: cannot find valid restart files for processor "+str(smilei_mpi_rank) +
                "\n\t\trestart_dir = '" + Checkpoints.restart_dir +
                "'\n\t\trestart_number = " + str(Checkpoints.restart_number) +
                "\n\t\tmatching pattern: '" + pattern + "'" )
            
        else :
            raise Exception("restart_dir and restart_files are both not empty")
    
    # Verify that constant() and tconstant() were not redefined
    if not hasattr(constant, "_reserved") or not hasattr(tconstant, "_reserved"):
        raise Exception("Names `constant` and `tconstant` cannot be overriden")
    
    # Convert float profiles to constant() or tconstant()
    def toSpaceProfile(input):
        try   : return constant(input*1.)
        except: return input
    def toTimeProfile(input):
        try:
            input*1.
            return tconstant()
        except: return input
    Main.pml_sigma = [ toSpaceProfile(p) for p in (Main.pml_sigma ) ]
    Main.pml_kappa = [ toSpaceProfile(p) for p in (Main.pml_kappa ) ]
    for s in Species:
        s.number_density      = toSpaceProfile(s.number_density)
        s.charge_density  = toSpaceProfile(s.charge_density)
        s.particles_per_cell = toSpaceProfile(s.particles_per_cell)
        s.charge          = toSpaceProfile(s.charge)
        s.mean_velocity   = [ toSpaceProfile(p) for p in s.mean_velocity ]
        s.mean_velocity_AM = [ toSpaceProfile(p) for p in s.mean_velocity_AM ]
        s.temperature     = [ toSpaceProfile(p) for p in s.temperature   ]
    for e in ExternalField:
        e.profile         = toSpaceProfile(e.profile)
    for e in PrescribedField:
        e.profile         = toSpaceProfile(e.profile)
    for a in Antenna:
        a.space_profile   = toSpaceProfile(a.space_profile   )
        a.time_profile    = toTimeProfile (a.time_profile    )
    for l in Laser:
        l.chirp_profile   = toTimeProfile( l.chirp_profile )
        l.time_envelope   = toTimeProfile( l.time_envelope )
        l.space_envelope  = [ toSpaceProfile(p) for p in l.space_envelope ]
        l.phase           = [ toSpaceProfile(p) for p in l.phase          ]
    for s in ParticleInjector:
        s.number_density = toSpaceProfile(s.number_density)
        s.charge_density = toSpaceProfile(s.charge_density)
        s.time_envelope  = toTimeProfile(s.time_envelope)
        s.particles_per_cell = toSpaceProfile(s.particles_per_cell )
        s.mean_velocity   = [ toSpaceProfile(p) for p in s.mean_velocity ]
        s.temperature     = [ toSpaceProfile(p) for p in s.temperature   ]

# this function will be called after initialising the simulation, just before entering the time loop
# if it returns false, the code will call a Py_Finalize();
def _keep_python_running():
    # Verify all temporal profiles, and all profiles that depend on the moving window or on the load balancing
    profiles = []
    for las in Laser:
        profiles += [las.time_envelope]
        profiles += [las.chirp_profile]
        if type(las.space_time_profile) is list:
            profiles += las.space_time_profile
        if type(las.space_time_profile_AM) is list:
            profiles += las.space_time_profile_AM
        if hasattr(las, "_extra_envelope"):
            profiles += [las._extra_envelope]
    profiles += [ant.time_profile for ant in Antenna]
    profiles += [ant.space_time_profile for ant in Antenna]
    profiles += [e.profile for e in PrescribedField]
    if len(MovingWindow)>0 or len(LoadBalancing)>0:
        # Verify if PML are used
        for bcs in Main.EM_boundary_conditions:
            for bc in bcs:
                if (bc == "PML"):
                    return True
        for s in Species:
            profiles += [s.number_density, s.charge_density, s.particles_per_cell, s.charge] + s.mean_velocity + s.mean_velocity_AM + s.temperature
    if len(MovingWindow)>0:
        profiles += [e.profile for e in ExternalField]
    for i in ParticleInjector:
        s = Species[i.species]
        profiles += [
                i.time_envelope,
                i.number_density or s.number_density,
                i.charge_density or s.charge_density,
                i.particles_per_cell or s.particles_per_cell
            ] + i.mean_velocity or s.mean_velocity + i.temperature or s.temperature
    for prof in profiles:
        if callable(prof) and not hasattr(prof,"profileName"):
            return True
    # Verify SDMD grids
    if len(LoadBalancing)>0 and len(MultipleDecomposition)>0:
        return True
    # Verify the tracked species that require a particle selection
    if any([d.filter for d in DiagTrackParticles]) or any([d.filter for d in DiagNewParticles]):
        return True
    # Verify the particle binning having a function for deposited_quantity or axis type
    for d in DiagParticleBinning._list + DiagScreen._list:
        if type(d.deposited_quantity) is not str:
            return True
        for ax in d.axes:
            if type(ax[0]) is not str:
                return True
    # Verify the radiation spectrum having a function for deposited_quantity or axis type
    for d in DiagRadiationSpectrum._list:
        if type(d.photon_energy_axis[0]) is not str:
            return True
        for ax in d.axes:
            if type(ax[0]) is not str:
                return True
    # Verify if ionization from rate is used
    for s in Species:
        if s.ionization_rate is not None:
            return True
    # else False
    return False

# Prevent creating new components (by mistake)
def _noNewComponents(cls, *args, **kwargs):
    print("Please do not create a new "+cls.__name__)
    return None
SmileiComponent.__new__ = staticmethod(_noNewComponents)

# Writes some information in a pickle file for fast post-processing (set scan=False in happi)
def writeInfo():
    def pickable(var):
        if var is None or type(var) in [float, int, complex, str, bytes, bool]:
            return True
        elif type(var) in [list, tuple]:
            return all([pickable(v) for v in var])
        elif type(var) is dict:
            return all([pickable(var[k]) for k in var])
        else:
            try:
                import numpy
                if type(var) is numpy.ndarray and var.size < 10000:
                    return True
            except Exception:
                pass
            return False
    
    import shelve
    singletons = {}
    components = {}
    with shelve.open("info.shelf") as f:
        for name,var in globals().items():
            if type(var) is type(Main):
                singletons[name] = {k:v for k,v in var.__dict__.items() if not k.startswith("_") and pickable(v)}
            elif type(var) is type(Species):
                components[name] = [{k:v for k,v in component.__dict__.items() if not k.startswith("_") and pickable(v)} for component in var]
            elif not name.startswith("_") and pickable(var):
                f[name] = var
        f["_singletons"] = singletons
        f["_components"] = components
Main.cluster_width= 30
Main.omptasks= 0
