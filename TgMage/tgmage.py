"""Provide the primary functions."""
import numpy as np
import matplotlib.pyplot as plt
import linecache
import pickle
import sys

from scipy.stats import linregress
from scipy import stats

def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format).

    Replace this function and doc string for your own project.

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution.
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


def thermo_extractor(path, wanted_property=None):
    """
    Placeholder function to show example docstring (NumPy format).

    Replace this function and doc string for your own project.

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution.
    """
    header = linecache.getline(path, 1)  # Extract Header

    header_list = header.split()  # Split the header
    array_of_header = np.arange(len(np.array(header_list)))  # make list mutable and iterable

    if wanted_property is not None:
        wanted_property_index = header_list.index(wanted_property)
        timestep_data_index = header_list.index('Step')
        wanted_property_data = np.genfromtxt(path, usecols=wanted_property_index, skip_header=2)
        timestep_data = np.genfromtxt(path, usecols=timestep_data_index, skip_header=2)
        wanted_property_data_array = np.vstack((timestep_data, wanted_property_data))
        return wanted_property_data_array, header_list

    timestep_data_index = header_list.index('Step')
    thermostyle_data = np.genfromtxt(path, usecols=timestep_data_index, skip_header=2)  # Initialize the thermo_data
    # array by taking time first as first collum

    for ii in array_of_header[1:]:  # Iterate through the rest of the columns
        thermostyle_collum = np.genfromtxt(path, usecols=ii, skip_header=2)
        thermostyle_data = np.vstack((thermostyle_data, thermostyle_collum))  # Add to thermo_data

    return thermostyle_data, header_list  # Return the thermo_data and list of properties inside


def get_first_index_smaller_than(arr, value, warmup):
    T_index = 0
    P_index = 0
    for i in range(len(arr)):
        if arr[i,0] <= warmup:
            T_index = i
        elif arr[i,1] <= value:
            P_index = i
            break
    return T_index, P_index  # If no element is found lower than the specified value.
    

def get_last_index_smaller_than(arr, value, warmup):
    T_index = 0
    P_index = 0
    for i in range(len(arr)-1, -1, -1):
        if arr[i,0] >= warmup:
            T_index = i
        elif arr[i,1] <= value:
            P_index = i
            break
    return T_index, P_index  # If no element is found lower than the specified value.


def Tg_finder(density, IL, p_value=0.05, warmup=None, verbose=True, save=None):
    """
    Placeholder function to show example docstring (NumPy format).

    Replace this function and doc string for your own project.

    Parameters
    ----------
    density : Numpy array 
        Density vs. thermostat of sim with shape of 
        [Temperature, Density] with legth of cooling ramp.

    IL :

    
    p_value : 

    
    warmup=None : 

    
    verbose=True :

    
    save=None :

            
    Returns
    -------
    Tg : float
        Computed Tg by the intersection of linear fits for each
        phase fitted according to the tempereature range defined 
        by p vs. thermostat.  
    """
    # Temperature
    Temperature = density[0]
    Density = density[1]
    
    skip = 3 # number of points to start with
    
    # Get glass fitting range
    G_p_value = np.zeros((len(Temperature)-skip+1,2))
    for t in range(len(Temperature)-skip+1):
        # Get the fits of phase
        m, b, R, trash, std = linregress(Temperature[:skip+t], Density[:skip+t])
        # Get residuals
        Residuals = Density[:skip+t] - m*Temperature[:skip+t]+b
        # Get pvalue
        G_p_value[t][0] = Temperature[skip+t-1]
        G_p_value[t][1] = stats.shapiro(Residuals).pvalue
    
    G_Tstart, G_phase_limit = get_first_index_smaller_than(G_p_value, p_value, Temperature[0]+warmup)
    G_Temperature = Temperature[:np.where(Temperature==G_p_value[G_phase_limit][0])[0][0]+1]
    G_Density = Density[:np.where(Temperature==G_p_value[G_phase_limit][0])[0][0]+1]

    # Diagnostics
    if verbose:
        print(f"\n{IL}, warmup: {warmup}")
        print("\nGlass Diagnostics")
        print("Temperature:", Temperature)
        print("Glass p-values:", G_p_value)
        print("Glass warmup ends: ", G_p_value[G_Tstart][0])
        print(f"Glass phase fitting range [{Temperature[0]}, {G_p_value[G_phase_limit][0]}]")
        print("Glass phase Fitting points: ", np.array([G_Temperature, G_Density]).T)
        
    # Get liquid phase fitting range
    L_p_value = np.zeros((len(Temperature)-skip+1, 2))
    for t in range(len(Temperature)-skip, -1, -1):
        # Get the fits of phase
        m, b, R, trash, std = linregress(Temperature[t:], Density[t:])
        # Get residuals
        Residuals = Density[t:] - m*Temperature[t:]+b
        # Get pvalue
        L_p_value[t][0] = Temperature[t]
        L_p_value[t][1] = stats.shapiro(Residuals).pvalue

    L_Tstart, L_phase_limit = get_last_index_smaller_than(L_p_value, p_value, Temperature[-1]-warmup)
    # L_Temperature = Temperature[np.where(Temperature==L_p_value[L_phase_limit][0])[0][0]+1:]
    # L_Density = Density[np.where(Temperature==L_p_value[L_phase_limit][0])[0][0]+1:]
    L_Temperature = Temperature[np.where(Temperature==L_p_value[L_phase_limit][0])[0][0]:]
    L_Density = Density[np.where(Temperature==L_p_value[L_phase_limit][0])[0][0]:]

    if verbose:
        print("\nLiquid Diagnostics")
        print("Temperature:", Temperature)
        print("Liquid p-values:", L_p_value)
        print("Liquid warmup ends: ", L_p_value[L_Tstart][0])
        print(f"Liquid phase fitting range: [{Temperature[-1]}, {L_p_value[L_phase_limit][0]}]")
        print("Liquid phase fitting points:", np.array([L_Temperature, L_Density]).T)
    
    # Compute Tg
    G_m, G_b, G_R, trash, std = linregress(G_Temperature, G_Density)
    L_m, L_b, L_R, trash, std = linregress(L_Temperature, L_Density)
    Tg = (L_b - G_b)/(G_m - L_m) 
    print("Tg: ", Tg, "K")
    
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=600)
    
    #Title
    # fig.suptitle(f'{IL}, My Method')
    
    # First plot  
    axs[0, 0].scatter(Temperature, density[1], label="Density", s=1, marker="x", color="black")
    axs[0, 0].set_ylabel(r"Density [$g/cm^3$]")
    axs[0, 0].plot(Temperature[:G_phase_limit], [G_m*T+G_b for T in Temperature[:G_phase_limit]], label="Glass Range", lw=2)
    axs[0, 0].plot(Temperature, [G_m*T+G_b for T in Temperature], lw=1, linestyle='dotted', color='C0')
    axs[0, 0].plot(Temperature[L_phase_limit:], [L_m*T+L_b for T in Temperature[L_phase_limit:]], label="Liquid Range", lw=2)
    axs[0, 0].plot(Temperature, [L_m*T+L_b for T in Temperature], linestyle='dotted', color='C1')
    axs[0, 0].set_ylim((density[1].min(), density[1].max()))
    axs[0, 0].set_xlim((Temperature.min(), Temperature.max()))
    axs[0, 0].vlines(Tg, density[1].min(), density[1].max(), label=r'$T_g$='+f"{round(Tg, 2)}K", color="red")
    axs[0, 0].set_xlabel("Temperature [K]")
    axs[0, 0].legend(loc=1)
    
    # Second subplot
    axs[1, 0].scatter(Temperature, Density - [G_m*T+G_b for T in Temperature], marker="x", s=1, color='#dbedf9')
    axs[1, 0].scatter(Temperature, Density - [L_m*T+L_b for T in Temperature], marker="x", s=1, color='#fedcbd')
    axs[1, 0].scatter(Temperature[:G_phase_limit], Density[:G_phase_limit] - [G_m*T+G_b for T in Temperature[:G_phase_limit]], label="Glass", marker="x", s=2)
    axs[1, 0].scatter(Temperature[L_phase_limit:], Density[L_phase_limit:] - [L_m*T+L_b for T in Temperature[L_phase_limit:]], label="Liquid", marker="x", s=2)
    Glass_Residuals = np.array([Density - [G_m*T+G_b for T in Temperature]])
    Liquid_Residuals = np.array([Density - [L_m*T+L_b for T in Temperature]])
    Residuals = np.hstack((Glass_Residuals,Liquid_Residuals))
    axs[1, 0].set_xlim((Temperature.min(), Temperature.max()))
    axs[1, 0].set_ylim((Residuals.min(), Residuals.max()))
    axs[1, 0].set_ylabel("Residuals")
    axs[1, 0].set_xlabel("Temperature [K]")
    axs[1, 0].legend()   

    # Third subplot
    axs[0, 1].scatter(L_p_value[L_phase_limit:,0],L_p_value[L_phase_limit:,1], marker="x", s=5, color="C1")
    axs[0, 1].scatter(L_p_value[:,0], L_p_value[:,1], marker="x", s=1, color="black")
    axs[0, 1].set_ylabel("p-value")
    axs[0, 1].set_xlabel("Temperature [K]")
    axs[0, 1].vlines(L_p_value[L_Tstart][0], -2, 2, label=f"warmup, {warmup}K", color="orange")
    axs[0, 1].set_xlim((Temperature.min(), Temperature.max()))
    axs[0, 1].set_ylim((0,1))

    # Fourth subplot (previously second)
    axs[1, 1].scatter(G_p_value[:G_phase_limit,0], G_p_value[:G_phase_limit,1], marker="x", s=5, color="C0")
    axs[1, 1].scatter(G_p_value[:,0], G_p_value[:,1], marker="x", s=1, color="black")
    axs[1, 1].vlines(G_p_value[G_Tstart][0], -2, 2, label=f"warmup, {warmup}K", color="orange")
    axs[1, 1].set_xlabel("Temperature [K]")
    axs[1, 1].set_ylabel("p-value")
    axs[1, 1].set_xlim((Temperature.min(), Temperature.max()))
    axs[1, 1].set_ylim((0,1))
    
    # Formatting

    # Add labels to the subplots
    axs[0, 0].text(0.50, 0.89, "A", transform=axs[0, 0].transAxes, ha="left", va="bottom", fontsize=16, fontweight="bold")
    axs[0, 0].text(0.10, 0.10, IL, transform=axs[0, 0].transAxes, ha="left", va="bottom", fontsize=12)
    axs[0, 1].text(0.50, 0.89, "B", transform=axs[0, 1].transAxes, ha="left", va="bottom", fontsize=16, fontweight="bold")
    axs[1, 0].text(0.50, 0.89, "C", transform=axs[1, 0].transAxes, ha="left", va="bottom", fontsize=16, fontweight="bold")
    axs[1, 1].text(0.50, 0.89, "D", transform=axs[1, 1].transAxes, ha="left", va="bottom", fontsize=16, fontweight="bold")

    # Tight
    plt.tight_layout()
    
    # Save
    if save is not None:
        plt.savefig(f"{IL}_{warmup}.png", dpi=600)
        plt. close()
    else:
        plt.show()        
        
    return Tg


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
