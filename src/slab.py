'''This file includes the Slab object, A section of an N-body simulation cube. To form the slab, a simulation cube is given 
and is divided into slabs along a particular direction (the default is the z-axis). Then, 2D observables are calculated and
stored as instances of this class'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import emcee
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit



class SimSlab():
    '''Instancce:
    delta_m_2D (2D array): the two dimensional projected matter density along the given direction
    N_halos_2D (2D array): The number of halos
    resolution (tuple): the dimension of the slab
    '''

    def __init__(self, delta_m_3D, N_halos_3D, N_slabs, rank, direction, L, volume = None, ncells = 2*4096):
        '''N_slabs: the number of slabs in the simulation cube
        rank: the order of the chosen slab
        direction: the direction perpendicular to the slab'''
        
        if int(N_slabs) != N_slabs:
            raise ValueError("The number of slabs should be an integer, {} is not an integer".format(N_slabs)) 

        if rank > N_slabs:
            raise ValueError("The rank of the slab is larger than the number of slabs, the rank should be smaller than the number of the slabs")


        if delta_m_3D.shape[0]%N_slabs != 0:
            raise ValueError("The resilution of the cube should be divisible by the number of slabs, {} does not divide {}".format(delta_m_3D.shape[0], N_slabs))
        
        if direction not in ["x", "y", "z"]:
            raise ValueError("The direction is either x, y, or z, the given direction is {}".format(direction))
        
        
        height  = int(delta_m_3D.shape[0]/N_slabs)

        if direction == "z":
            delta_m_3D_slab = delta_m_3D[:,:, rank*height:rank*height + height - 1]
            N_halos_3D_slab = N_halos_3D[:,:, rank*height:rank*height + height - 1]
            self.delta_m_2D = np.mean(delta_m_3D_slab, axis = 2)
            self.N_halos_2D = np.sum(N_halos_3D_slab, axis = 2)

        elif direction == "y":
            delta_m_3D_slab = delta_m_3D[:,rank*height:rank*height + height - 1, : ]
            N_halos_3D_slab = N_halos_3D[:,rank*height:rank*height + height - 1, : ]
            self.delta_m_2D = np.mean(delta_m_3D_slab, axis = 1)
            self.N_halos_2D = np.sum(N_halos_3D_slab, axis = 1)

        else:
            delta_m_3D_slab = delta_m_3D[rank*height:rank*height + height - 1,:,:]
            N_halos_3D_slab = N_halos_3D[rank*height:rank*height + height - 1,:,:]
            self.delta_m_2D = np.mean(delta_m_3D_slab, axis = 0)
            self.N_halos_2D = np.sum(N_halos_3D_slab, axis = 0)
        
        self.resolution = (self.N_halos_2D.shape[0], self.N_halos_2D.shape[1])
         
        
        self.thickness = self.resolution[0]/N_slabs
        if volume is None: #only add the volume for the stacking step.
            self.volume = L*L*L/N_slabs
        else:
            self.volume = volume
        self.voxel_volume = self.volume/(self.resolution[0]*self.resolution[1]*self.thickness)


        delta_m_2D_map_vec = self.delta_m_2D.flatten()
        N_halos_2D_map_vec = self.N_halos_2D.flatten()

        order = delta_m_2D_map_vec.argsort()
        delta_m_2D_map_vec = delta_m_2D_map_vec[order]
        N_halos_2D_map_vec = N_halos_2D_map_vec[order]

        
    
    
        
        self.delta_m_2D_bin_mean, self.N_halos_2D_bin_mean, self.N_halos_2D_bin_var = SimSlab.bin_number_counts(delta_m_2D_map_vec, N_halos_2D_map_vec,
                                                                                    ncells=ncells)
        

        
        
        
        
        zero_filter = (N_halos_2D_map_vec > 0)
        
        delta_m_2D_map_vec_filtered = delta_m_2D_map_vec[zero_filter]
        N_halos_2D_map_vec_filtered = N_halos_2D_map_vec[zero_filter]
        
        self.delta_m_2D_bin_mean_filtered, self.N_halos_2D_bin_mean_filtered, self.N_halos_2D_bin_var_filtered = SimSlab.bin_number_counts(delta_m_2D_map_vec_filtered, N_halos_2D_map_vec_filtered,
                                                                                    ncells=ncells)
        
        
        
        
    def bin_number_counts(delta_m, Nh, ncells = 200):
        order = delta_m.argsort()
        delta_m = delta_m[order]
        Nh = Nh[order]

        n_bins = (Nh.size - Nh.size%ncells)

        delta_m_cropped = delta_m[:n_bins]
        Nh_cropped = Nh[:n_bins]
        delta_m_bins = delta_m_cropped.reshape(-1,ncells)
        Nh_bins = Nh_cropped.reshape(-1,ncells)

        delta_m_2D_bin_mean = np.mean(delta_m_bins, axis = 1)
        N_halos_2D_bin_mean = np.mean(Nh_bins, axis = 1)
        N_halos_2D_bin_var  = np.std(Nh_bins, axis = 1)/np.sqrt(ncells)

        if Nh_cropped.size != Nh.size:
            delta_add = np.mean(delta_m[n_bins:])
            Nh_add = np.mean(Nh[n_bins:])
            Nh_add_var = np.std(Nh[n_bins:])/np.sqrt(Nh.size)
            delta_m_2D_bin_mean = np.append(delta_add, delta_m_2D_bin_mean)
            N_halos_2D_bin_mean = np.append(Nh_add, N_halos_2D_bin_mean)
            N_halos_2D_bin_var = np.append(Nh_add_var, N_halos_2D_bin_var)




        return delta_m_2D_bin_mean, N_halos_2D_bin_mean, N_halos_2D_bin_var

    @classmethod
    def from_slabs(cls, slabs_array, L):
        '''A method that takes a set of k number 2D slabs and stack them 
        into one slab 2D slab object. The individual resolution 
        of each slab component should be the same, NxN, and the resultant slab 
        should have a resolution of Nxk.N. It is better to lower the resolution before
        stacking 
        attributes:
        slabs_array: an array (Not numpy) that includes Slab objects which we aim to stack into one
        L: the original comoving length of the imulation cube'''
        
        if not slabs_array:
            raise ValueError("slabs_array must contain at least one Slab object.")
        
        slab_shape = slabs_array[0].resolution
        N_slabs = len(slabs_array)
        for slab in slabs_array:
            if slab.resolution != slab_shape:
                raise ValueError("All slabs must have the same resolution.")
        # Stack slabs along a new axis
        delta_m_2D_grid = slabs_array[0].delta_m_2D
        Nh_2D_grid = slabs_array[0].N_halos_2D
        for slab in slabs_array[1:]:
            delta_m_2D_grid = np.concatenate([delta_m_2D_grid, slab.delta_m_2D], axis=1)
            Nh_2D_grid = np.concatenate([Nh_2D_grid, slab.N_halos_2D], axis=1)



        new_resolution = Nh_2D_grid.shape
        
        
        return cls(
            delta_m_3D=np.expand_dims(delta_m_2D_grid, axis=2),  # Fake 3D input
            N_halos_3D=np.expand_dims(Nh_2D_grid, axis=2),  # Fake 3D input
            N_slabs=N_slabs,
            rank=0,  # Not relevant for stacked slabs
            direction="z",  # Arbitrary choice since it's already 2D
            L=L, volume = L**3
        )


    @classmethod
    def combine_mass_bins(cls, mass_bin_list,L, N_slabs = 8):
        delta_m_2D = mass_bin_list[0].delta_m_2D
        N_halos_2D = 0
        for bin in mass_bin_list:
            N_halos_2D = bin.N_halos_2D + N_halos_2D
        
        print(delta_m_2D.shape, N_halos_2D.shape)
        return cls(
            delta_m_3D=np.expand_dims(delta_m_2D, axis=2),  # Fake 3D input
            N_halos_3D=np.expand_dims(N_halos_2D, axis=2),  # Fake 3D input
            N_slabs=N_slabs,
            rank=0,  # Not relevant for stacked slabs
            direction="z",  # Arbitrary choice since it's already 2D
            L=L, volume = L**3
        )
        

            
            

    

    
   

    def log_L_Poissonian(self, fixed_params_names, free_params_names, fixed_params_values, bias_model,  delta_m_cut, *params):
        '''bias_model(BiasModel): bias model object'''
        all_parameters_list = {name: None for name in bias_model.parameters}

        
        for (i, j) in zip(fixed_params_names, fixed_params_values):
            all_parameters_list[i] = j

        for (i, j) in zip(free_params_names, params[0]):
            all_parameters_list[i] = j
        
        if not bias_model.priors(**all_parameters_list):
            return -np.inf
        delta_m_2D = self.delta_m_2D.flatten()
        N_halos_2D = self.N_halos_2D.flatten()
        if delta_m_cut is None:
            pass
        else:
            mask = np.where(delta_m_2D >= delta_m_cut)
            delta_m_2D = delta_m_2D[mask]
            N_halos_2D = N_halos_2D[mask]
        mu = np.clip(bias_model.function(delta_m_2D, **all_parameters_list), 1e-3, None)
        log_Likelihood = np.sum(N_halos_2D*np.log(mu) - mu)


        return log_Likelihood
    

    def negative_log_L_Poissonian(self, fixed_params_names, free_params_names, fixed_params_values, bias_model, delta_m_cut, *params):

        return -1*SimSlab.log_L_Poissonian(self, fixed_params_names, free_params_names, fixed_params_values, bias_model, delta_m_cut,  *params)
    



    def optimize_max_L_Poissonian(self, initial_guess, fixed_params_names, free_params_names, fixed_params_values, bias_model, delta_m_cut = None):
        # write an error for the consistency between the initial guess and chosen bias model
        '''A fucntion to fit bias paramters model'''
        objective_function = lambda params: SimSlab.negative_log_L_Poissonian(self, fixed_params_names, free_params_names, fixed_params_values, bias_model, delta_m_cut, params)
        result = minimize(fun = objective_function , x0 = initial_guess)
        return result["x"], result["hess_inv"], result["fun"]



    def log_L_Generalized_Poissonian(self, fixed_params_names, free_params_names,  fixed_params_values, bias_model, theta_model, *params, delta_m_cut = None):
        
        parameter_names = bias_model.parameters + theta_model.parameters
        all_parameters_list = {name: None for name in parameter_names}


        for (i, j) in zip(fixed_params_names, fixed_params_values):
            all_parameters_list[i] = j
        for (i, j) in zip(free_params_names, params[0]):
            all_parameters_list[i] = j
        
        if not (bias_model.priors(**all_parameters_list) and theta_model.priors(**all_parameters_list)):
            return -np.inf
        delta_m_2D = self.delta_m_2D
        N_halos_2D = self.N_halos_2D
        if delta_m_cut is None:
            pass
        else:
            mask = np.where(delta_m_2D >= delta_m_cut)
            delta_m_2D = delta_m_2D[mask]
            N_halos_2D = N_halos_2D[mask]

        plt.scatter()
        theta = theta_model.function(delta_m_2D, **all_parameters_list)
        mu = bias_model.function(delta_m_2D, **all_parameters_list)
        mu_clipped = np.clip(mu, 1e-3, None)
        one_minus_theta = np.clip(1 - theta, 1e-3, None) 
        mu_plus_theta = np.clip(mu + theta*(N_halos_2D - mu), 1e-3, None)
        log_Likelihood = np.sum(np.log(mu_clipped) + np.log(one_minus_theta) + (N_halos_2D - 1)*np.log(mu_plus_theta) - (mu + theta*(N_halos_2D - mu)))


        return log_Likelihood


    

    
        
    def negative_log_L_Generalized_Poissonian(self, fixed_params_names, free_params_names,  fixed_params_values, bias_model, theta_model, *params):
        return -1*SimSlab.log_L_Generalized_Poissonian(self, fixed_params_names, free_params_names,  fixed_params_values, bias_model, theta_model, *params)


    
   
        
    def optimize_max_L_Generalized_Poissonian(self, initial_guess, fixed_params_names, free_params_names,  fixed_params_values, bias_model, theta_model, *params):
        # write an error for the consistency between the initial guess and chosen bias model
        objective_function = lambda params: SimSlab.negative_log_L_Generalized_Poissonian(self, fixed_params_names, free_params_names,  fixed_params_values, bias_model, theta_model, params)

        result = minimize(fun = objective_function , x0 = initial_guess)
        return result["x"], result["hess_inv"], result["fun"]

    

   
    
    
    def optimize_MCMC(self, Nwalkers, ndim, low, high, burn_in = 100):
        sampler = emcee.EnsembleSampler(nwalkers = Nwalkers, ndim = ndim, log_prob_fn= self.log_L)
         
        initial_state =  np.random.uniform(low, high, size = (Nwalkers, ndim))
        state = sampler.run_mcmc(initial_state = initial_state, nsteps=1000)
        samples = sampler.get_chain(flat = True, discard = burn_in)
        return samples


            


    

    def plot_slab(self):
        
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 12))

        # First subplot: halo distribution
        bar0 = ax[0].imshow(self.N_halos_2D, vmin = 0., vmax = 10.)
        ax[0].set_title("2D Halo Distribution")
        cbar0 = fig.colorbar(bar0, ax=ax[0], shrink=0.4, aspect=20 )
        cbar0.set_label("Number of Halos")

        # Second subplot: matter distribution
        bar1 = ax[1].imshow(self.delta_m_2D,  vmin = 0., vmax = 1.)
        ax[1].set_title("2D Matter Distribution")
        cbar1 = fig.colorbar(bar1, ax=ax[1], shrink=0.4, aspect=20)
        cbar1.set_label("2D Matter Density ($\mathrm{Mpc}^{-2} \cdot h$)")


        fig.subplots_adjust(wspace=0.7)
        plt.tight_layout()
        plt.show()
        return fig
    

    def plot_Nh_delta_m(self,  Mmin, Mmax, fig_and_ax = None):
        if fig_and_ax is None:
            fig, ax = plt.subplots(figsize = (10, 8))
        else:
            fig, ax = fig_and_ax
        ax.scatter(self.delta_m_2D.flatten(), self.N_halos_2D.flatten(), marker="*", s =6, 
                   label = rf"{np.log10(Mmin):.2f}< $log10(M_{{halos}})$ < {np.log10(Mmax):.2f}")
        ax.set_xlabel(r"$\delta_{m}$")
        ax.set_ylabel(r"$N_{h}$")
        return fig, ax


    # some function routines to prepare the simulation cube for slab construciton

    def halo_catalog_to_grid(x_halo, y_halo, z_halo, l_cube, resol):
        '''A function that distribute haloes from halo catalog on a grid
            arguments:
            x_halo: the position of halos on the x-axis
            y_halo: the position of halos on the y-axis
            z_halo: the position of halos on the z-axis
            l_cube: the length of the simulation cube in Mpc/h 
                    or any unit of length consistent with 
                    the units of position
            resol: the chosen resolution of the simulation cube

            return:
            halos_grid_counts: a grid containing the number of halos in each 
            element of the grid
        '''
        # the length of one voxel
        l = l_cube/resol
        print("the length of one voxel", l, "Mpc/h")
    
        
        halo_pos = np.array([x_halo, y_halo, z_halo]).T
        halo_pos_index = np.floor(halo_pos/l).astype(int)
        
        # distributing halos on a grid
        
        edges = [np.linspace(0, resol, resol + 1) for _ in range(3)]
        halo_grid_counts, _ = np.histogramdd(halo_pos_index, bins=edges)

        return halo_grid_counts
    
    def delta_m_catalogue_to_grid(ix_matter, iy_matter, iz_matter, delta_m, resol):
        delta_m_grid = np.zeros((512, 512, 512))
        delta_m_grid[ix_matter, iy_matter, iz_matter] = delta_m
        if resol != 512:
            delta_m_grid = SimSlab.lower_resolution_delta_m_3D(delta_m_grid, resol=resol)
        return delta_m_grid
    





    def find_A(self): 
        '''A function to find the mean number of halos in one 2D cell. it is simply the mean number of halos in 
        one voxel'''
        number_of_3D_voxels = self.resolution[0]*self.resolution[1]*self.thickness
        n_bar = np.sum(self.N_halos_2D)/number_of_3D_voxels
        A_slab = n_bar * self.thickness
        return A_slab
    

    def lower_resolution_delta_m_3D(grid, resol):
        sh = (resol,grid.shape[0]//resol,
            resol,grid.shape[1]//resol,
            resol,grid.shape[2]//resol)

        return grid.reshape(sh).mean(-1).mean(-2).mean(-3)





    def lower_resolution_2D(self, kernel_resolution):
        "lowers down the resolution of slab object, which is already projected in 2D"
        old_resolution = self.resolution[0]
        kernel_resolution = int(kernel_resolution)
        new_resolution = int(old_resolution/kernel_resolution) 
        new_array_delta_m_2D = np.zeros((new_resolution, new_resolution))
        new_array_N_h_2D = np.zeros((new_resolution, new_resolution))

        i = np.arange(0, old_resolution - 1, kernel_resolution)
        j = np.arange(0, old_resolution - 1, kernel_resolution)


        for ix in i:
            for jy in j:
                new_array_delta_m_2D[int(ix/kernel_resolution), int(jy/kernel_resolution)] = self.delta_m_2D[ix, jy] + self.delta_m_2D[ix + 1, jy] + self.delta_m_2D[ix, jy+1] + self.delta_m_2D[ix + 1, jy + 1] 
                new_array_N_h_2D[int(ix/kernel_resolution), int(jy/kernel_resolution)] = self.N_halos_2D[ix, jy] + self.N_halos_2D[ix + 1, jy] + self.N_halos_2D[ix, jy+1] + self.N_halos_2D[ix + 1, jy + 1] 


        self.delta_m_2D = new_array_delta_m_2D/(kernel_resolution**2)
        self.N_halos_2D = new_array_N_h_2D
        self.resolution = (new_resolution, new_resolution)
        self.voxel_volume = self.voxel_volume* (kernel_resolution**2)




    def delta_m_2D_binning(self, delta_m_2D_bin_edges):
        delta_m_bins  =  []
        N_halos_bins = []

        for k, (delta_m_min, delta_m_max) in enumerate(zip(delta_m_2D_bin_edges[:-1], delta_m_2D_bin_edges[1:]), 0):
            mask_delta_bin = np.where((self.delta_m_2D >= delta_m_min) & (self.delta_m_2D <= delta_m_max), True, False)
            delta_m_2D_masked = self.delta_m_2D[mask_delta_bin]
            N_halos_2D_masked = self.N_halos_2D[mask_delta_bin]

            delta_m_bins.append(delta_m_2D_masked)
            N_halos_bins.append(N_halos_2D_masked)


        return delta_m_bins, N_halos_bins
    

    def interpolated_bias_function(self, ncells, delta_m_cut):

        
        
        mask = np.where(self.delta_m_2D_bin_mean >= delta_m_cut)
        delta_m_2D_mean_masked = self.delta_m_2D_bin_mean[mask]
        N_halos_2D_mean_masked = self.N_halos_2D_bin_mean[mask]
        
       
        
        bias_relation_splined = interp1d(x = delta_m_2D_mean_masked, y = N_halos_2D_mean_masked, kind = 'linear', fill_value = "extrapolate")

        return bias_relation_splined



    def fit_bias_model_to_spline(self, initial_guess, bias_model, bias_relation_splined):
        delta_m_2D_sample = np.linspace(np.min(self.delta_m_2D), np.max(self.delta_m_2D), 100)
        N_halos_2D_sample = bias_relation_splined(delta_m_2D_sample)
        


        def objective_function(xdata, *params):
            all_parameters_list = {name: None for name in bias_model.parameters}
            for (i, j) in zip(bias_model.parameters, params):
                all_parameters_list[i] = j
            
            return bias_model.function(xdata, **all_parameters_list)
        
        
            
        
       

        popt, pcov = curve_fit(objective_function, xdata = delta_m_2D_sample, ydata = N_halos_2D_sample, 
                               p0=initial_guess)
        
        return popt, pcov