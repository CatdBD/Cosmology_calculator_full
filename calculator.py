__author__ = 'Caffe'
# !/usr/bin/env python

'''
A cosmology calculator written by Robert L. Barone Nugent,
Catherine O. de Burgh-Day and Jaehong Park, 2014.

'''

import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import argparse
import os

# Define the deafult values for the argument parser
DEFAULT_REDSHIFT_RNG = [-1.0, 4.0, 20]
DEFAULT_HUBBLE_CONSTANT = 70.0
DEFAULT_MASS_DENSITY = 0.3
DEFAULT_RADIATION_DENSITY = 0.0
DEFAULT_DARK_ENERGY_DENSITY = 0.7
DEFAULT_SAVEDIR = __file__

def freidman_equation(redshift, mass_density, dark_energy_density):
    '''Return the *dimensionless* friedman equation
    (i.e. without the H_0**2 term)'''
    return np.sqrt(mass_density * (1.0 + redshift) ** 3.0 + \
                   dark_energy_density * (1.0 + redshift))


def tage_integral(redshift, mass_density, dark_energy_density):
    '''Compute the age of the universe'''
    return 1.0 / ((1.0 + redshift) * freidman_equation(redshift, mass_density, dark_energy_density))


def freidman_integral(redshift, mass_density, dark_energy_density):
    '''Compute the proper distance'''
    return 1.0 / freidman_equation(redshift, mass_density, dark_energy_density)


def cosmo(redshift, hubble_constant,
          mass_density=DEFAULT_MASS_DENSITY,
          dark_energy_density=DEFAULT_DARK_ENERGY_DENSITY,
          radiation_density=DEFAULT_RADIATION_DENSITY):
    """
    A simple cosmology calculator.
    Default is concordance cosmology (Flat, negative pressure, radiation free,
    with hubble_constant = 70, mass_density = 0.3, dark_energy_density = 0.7)

    Inputs:
        redshift: the redshift (z) at which to compute the cosmology.
            Unfortunately because the integrator can't take arrays or lists
            this must be a number only.
        hubble_constant: Hubble's Constant (H0)
        mass_density: Omega_M
        dark_energy_density: Omega_Lambda.
    Returns: a Cosmology object
    """
    # These are all constants which you want to very high accuracy

    # units of km
    speed_of_light = 2.99792458e5
    # converts Mpc to km
    Mpc2km = 3.08567758147e+19
    # seconds in a julian year
    seconds_in_a_year = 31557600.0

    curvature = 1. - mass_density - dark_energy_density - radiation_density # curvature term

    # hubble distance
    hubble_distance = speed_of_light / hubble_constant

    # Age at redshift in gigayears
    # The stuff after the integral is all unit conversions
    age_at_redshift = 1.0 / hubble_constant * \
                      integrate.quad(lambda r: tage_integral(r, mass_density, dark_energy_density),
                                     redshift, np.inf)[0] * Mpc2km / (seconds_in_a_year) / (1.e9)

    # or is there curvature?
    if curvature < -1.e-15:
        proper_distance = hubble_distance / np.sqrt(abs(curvature)) * np.sin(np.sqrt(abs(curvature))
        *integrate.quad(lambda r: freidman_integral(r, mass_density, dark_energy_density),0, redshift)[0])
        comoving_volume = ((4*np.pi*hubble_distance**3)/(2*curvature))*((proper_distance/hubble_distance)*
                            np.sqrt(1+curvature*(proper_distance/hubble_distance)**2)\
                            -1/np.sqrt(abs(curvature))*np.arcsin(np.sqrt(abs(curvature))
                            *proper_distance/hubble_distance))/(1.e9)
            # (Gpc**3), true only in an omega_k=1 Universe

    elif curvature > 1.e-15:
        proper_distance = hubble_distance / np.sqrt(curvature) * np.sinh(np.sqrt(curvature)\
        *integrate.quad(lambda r: freidman_integral(r, mass_density, dark_energy_density),0, redshift)[0])
        comoving_volume = ((4*np.pi*hubble_distance**3)/(2*curvature))*((proper_distance/hubble_distance)*
                            np.sqrt(1+curvature*(proper_distance/hubble_distance)**2)
                            -1/np.sqrt(abs(curvature))*np.arcsinh(np.sqrt(abs(curvature))
                            *proper_distance/hubble_distance)  )/(1.e9)
            # (Gpc**3), true only in an curvature=1 Universe

    else:
        proper_distance = hubble_distance*integrate.quad(lambda r:
                            freidman_integral(r, mass_density, dark_energy_density),0, redshift)[0]
        comoving_volume = 4./3.*np.pi * proper_distance**3. /(1.e9) # (Gpc**3), true only in an curvature=1 Universe

    # The rest follows from the proper distance
    luminosity_distance = proper_distance * (1.0 + redshift)
    angular_diameter_distance = proper_distance / (1.0 + redshift)
    distance_modulus = 5. * np.log10(luminosity_distance * 10**5)		# distance modulus

    return [proper_distance, luminosity_distance, angular_diameter_distance,
            age_at_redshift, distance_modulus, comoving_volume]

def compute_cosmologies(arguments):
    '''Takes arguments from the argparser and uses them to
    1) generate an array of redshifts
    2) define a dictionariey to store the resulting cosmology,
       containing empty lists
    3) for each redshift, compute the cosmology
    4) append the resulting information into the relevant entry
       in the dictionary
    5) return a list containing the array of redshifts,
       and the dictionary.'''

    # Array of log-spaced redshifts
    z_lower, z_upper, num_pts = arguments.redshift_range
    z_arr = np.logspace(z_lower, z_upper, num_pts)
    # Make an array of linearly-spaced redshifts
    # z = np.linspace(0.01, 1000, 20)

    # Initialise the dictionaries
    cosmology_dict = {'r': [], 'DL': [], 'DA': [], 'tage': [], 'dm': [], 'comov': []}

    for z in z_arr:
        # Call the cosmology function for different scenarios
        cosmology_results = cosmo(z, arguments.hubble_const)

        # Put the results into dictionaries
        cosmology_dict['r'].append(cosmology_results[0])
        cosmology_dict['DL'].append(cosmology_results[1])
        cosmology_dict['DA'].append(cosmology_results[2])
        cosmology_dict['tage'].append(cosmology_results[3])
        cosmology_dict['dm'].append(cosmology_results[4])
        cosmology_dict['comov'].append(cosmology_results[5])

    # Putting everything after z_arr in square brackets means they
    # will be returned as one list.
    return z_arr, cosmology_dict


def plot_cosmology(output_filename, z_arr, cosmologies, savedir):
    ''' Plots the four parameters for each of the cosmologies
        on a 2x2 plot. '''
    # Unpack the cosmologies

    # set the tick label and axis label size for all plots
    # ** This won't work if you use Ipython becasue rcPramas gets
    # imported when you use matplotlib the first time only in Ipython.
    # You'll need to re-open Ipython every time you chage these **
    matplotlib.rcParams['axes.labelsize'] = 30
    matplotlib.rcParams['xtick.labelsize'] = 30
    matplotlib.rcParams['ytick.labelsize'] = 30

    plt.close()
    # Create the figure instances in a 4x4 plot
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(24, 18))

    # Plot onto the top left axis. 'label' is what is used to make the legend.
    ax1.plot(z_arr, cosmologies['r'], color='r', linewidth=3)
    ax1.set_yscale('log')  # Make the plots logscale
    ax1.set_xscale('log')
    ax1.set_ylabel('Proper distance (Mpc)')
    ax1.set_xlabel('Redshift, z')
    # increase the space between the y axis ticklabels and the axis label
    ax1.yaxis.labelpad = 20

    ax2.plot(z_arr, cosmologies['DA'], color='r', linewidth=3)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_ylabel('Angular diameter distance (Mpc)')
    ax2.set_xlabel('Redshift, z')
    ax2.yaxis.labelpad = 20

    ax3.plot(z_arr, cosmologies['tage'], color='r', linewidth=3)
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_ylim(ax3.get_ylim()[0], 1e4)
    ax3.set_ylabel('Age at redshift z (Gyr)')
    ax3.set_xlabel('Redshift, z')
    ax3.yaxis.labelpad = 20

    ax4.plot(z_arr, cosmologies['DL'], color='r', linewidth=3)
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    ax4.set_ylabel('Luminosity distance (Mpc)')
    ax4.set_xlabel('Redshift, z')
    ax4.yaxis.labelpad = 20

    ax5.plot(z_arr, cosmologies['dm'], color='r', linewidth=3)
    ax5.set_yscale('log')
    ax5.set_xscale('log')
    ax5.set_ylabel('Distance Modulus')
    ax5.set_xlabel('Redshift, z')
    ax5.yaxis.labelpad = 20

    ax6.plot(z_arr, cosmologies['comov'], color='r', linewidth=3)
    ax6.set_yscale('log')
    ax6.set_xscale('log')
    ax6.set_ylabel('Comoving Volume (Gpc$^3$)')
    ax6.set_xlabel('Redshift, z')
    ax6.yaxis.labelpad = 20

    # Tidy up the layout
    plt.tight_layout(rect=(0., 0, 1., 0.9))
    # Save the fig
    plt.savefig(savedir+'/'+output_filename)


def parse_arguments():
    '''Parse the command line arguments of the program'''

    parser = argparse.ArgumentParser(description='Cosmology example')

    parser.add_argument(
        '--plot_name',  # The name of the input variable
        required=True,  # Is it an optional variable?
        metavar='OUTPUT_PLOT_NAME',  # For errormessage printing purposes
        type=str,  # The datatype of the input variable
        # A description of the variable or error message purposes
        help='name of output file for plotted graphs'
    )

    parser.add_argument(
        '--redshift_range',
        required=False,
        metavar='REDSHIFT_RNG',
        type=float,
        nargs='+',
        help='A list containing the range of redshifts (log-spaced) and the number of points, defaults to {}'.format(
            DEFAULT_REDSHIFT_RNG),
        # If the variable is optional, this is the default value
        # (defined at the top of the code)
        default=DEFAULT_REDSHIFT_RNG
    )

    parser.add_argument(
        '--hubble_const',
        required=False,
        metavar='HUBBLE_CONSTANT',
        type=float,
        help='The hubble constant, H_0, defaults to {}'.format(DEFAULT_HUBBLE_CONSTANT),
        default=DEFAULT_HUBBLE_CONSTANT
    )

    parser.add_argument(
        '--omega_m',
        required=False,
        metavar='OMEGA_M',
        type=float,
        help='Mass density, Omeaga_m, defaults to {}'.format(DEFAULT_MASS_DENSITY),
        default=DEFAULT_MASS_DENSITY
    )

    parser.add_argument(
        '--omega_r',
        required=False,
        metavar='OMEGA_R',
        type=float,
        help='Mass density, Omeaga_r, defaults to {}'.format(DEFAULT_RADIATION_DENSITY),
        default=DEFAULT_RADIATION_DENSITY
    )

    parser.add_argument(
        '--omega_L',
        required=False,
        metavar='OMEGA_L',
        type=float,
        help='Dark energy density, Omeaga_L, defaults to {}'.format(DEFAULT_DARK_ENERGY_DENSITY),
        default=DEFAULT_DARK_ENERGY_DENSITY
    )

    parser.add_argument(
        '--savedir',
        required=False,
        metavar='SAVEDIR',
        type=str,
        help='Directory for plots to be saved to. Defaults to {}'.format(DEFAULT_SAVEDIR),
        default=DEFAULT_SAVEDIR
    )

    return parser.parse_args()


def main(arguments):
    '''This is a special function name. Everything in your code should
        run inside main()'''
    # arguments = parse_arguments()
    print 'Computing cosmology for:'
    print 'H_0 = %s, Omega_M = %s, Omega_L = %s' \
          % (arguments.hubble_const, arguments.omega_m, arguments.omega_L)
    z_arr, cosmologies = compute_cosmologies(arguments)
    print 'Plotting results, along with some common cosmologies.'
    savedir = os.path.dirname(os.path.abspath(arguments.savedir))
    print u"Saving results to: '{0:s}'".format(savedir)
    np.save(savedir+'/'+'Cosmology_results.npy', [z_arr,cosmologies])
    plot_cosmology(arguments.plot_name, z_arr, cosmologies, savedir)


if __name__ == '__main__':
    main()
