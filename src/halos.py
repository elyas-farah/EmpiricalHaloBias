import numpy as np
from astropy.io import fits
import pandas as pd


class Halos:
    """
    :halo_df_filename: csv file with the halo data
    :L: The side of the slabs in Mpc/h
    :N_grid: The 2d grids 
    :MASS_CUTOFF_LOW: Low mass cut-off (only select halos above this mass) 
    :MASS_CUTOFF_HIGH: High mass cut-off (only select halos below this mass) 
    """
    def __init__(self, halo_df_filename, 
                         L=1000., 
                         N_grid=512, 
                         MASS_CUTOFF_LOW  = 1e+11, 
                         MASS_CUTOFF_HIGH = 1e+16):
        self.N = N_grid
        self.L = L
        l = (self.L + 1e-10) / N_grid
        halo_df = fits.open(halo_df_filename)        
        halo_df = pd.DataFrame(halo_df[1].data)
        
        select_mass = (halo_df['mass'] > MASS_CUTOFF_LOW) & (halo_df['mass'] < MASS_CUTOFF_HIGH)
        
        self.halo_mass = np.array(halo_df['mass'].values[select_mass])
        self.x_halo = np.array(halo_df['x'].values[select_mass])
        self.y_halo = np.array(halo_df['y'].values[select_mass])
        self.z_halo = np.array(halo_df['z'].values[select_mass])

        self.ix_halo = (self.x_halo / l).astype(int)
        self.iy_halo = (self.y_halo / l).astype(int)
        self.iz_halo = (self.z_halo / l).astype(int)
    
    def get_halo_counts(self, L_start, L_end, axis='z'):
        counts_2d = np.zeros((self.N,self.N))
        if(axis=='z'):
            select_slice = (self.z_halo > L_start) & (self.z_halo < L_end)
            index_select = np.array([self.ix_halo[select_slice], self.iy_halo[select_slice]]).T
        if(axis=='y'):
            select_slice = (self.y_halo > L_start) & (self.y_halo < L_end)
            index_select = np.array([self.ix_halo[select_slice], self.iz_halo[select_slice]]).T
        if(axis=='x'):
            select_slice = (self.x_halo > L_start) & (self.x_halo < L_end)
            index_select = np.array([self.iy_halo[select_slice], self.iz_halo[select_slice]]).T
        unique_indices, counts = np.unique(index_select, axis=0, return_counts=True)
        counts_2d[unique_indices[:,0], unique_indices[:,1]] = counts
        return counts_2d
    
    def get_halo_counts_in_slabs(self, slab_width, axis='z'):
        # Extract eight 2d density slabs from the 3d density
        halo_count_list = []

        N_slabs = int(self.L / slab_width)
        for i in range(N_slabs):
            L_START = i * slab_width
            L_END   = (i+1) * slab_width
    
            halo_counts = self.get_halo_counts(L_START, L_END, axis)
            halo_count_list.append(halo_counts)        
        return halo_count_list