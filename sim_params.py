import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi,sqrt
import scipy 
import scipy.constants

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


dx                  = 0.0625*um     # longitudinal resolution
dr                  = 0.0625*um                    # radial resolution

nx                  = 960                      # number of grid points in the longitudinal direction
nr                  = 368                        # number of grid points in the radial direction

Lx                  = nx * dx                   # longitudinal size of the simulation window
Lr                  = nr * dr                   # radial size of the simulation window, which goes from r=0 to r=Lr

npatch_x            = 32
npatch_r            = 16

dt                  = 0.44*dx/c_normalized      # integration timeestep
Niterations         = 15000

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

window_vx = sqrt(1 - omegap**2/omega0**2)

focus_time = 392 #fs