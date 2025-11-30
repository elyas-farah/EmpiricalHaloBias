'''This class is meant to facilitate plotting and comparing a combination of likelihood and
bias models by plotting.
Modify the class since the mean of bins and the bins centers is already 
in the Slab object.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


class ModelComparison():

    def __init__(self, slab_object, bias_model, theta_model, parameters_names, parameters_values, Mh_min, Mh_max, delta_m_cut ,ncells = 2*4096):
        '''The input of parameter values should follow the order of 
            parameter names'''
        
        self.slab_object = slab_object
        self.bias_model = bias_model
        self.theta_model = theta_model
        delta_m_2D_map_vec = self.slab_object.delta_m_2D.flatten()
        N_halos_2D_map_vec = self.slab_object.N_halos_2D.flatten()

        order = delta_m_2D_map_vec.argsort()
        delta_m_2D_map_vec = delta_m_2D_map_vec[order]
        N_halos_2D_map_vec = N_halos_2D_map_vec[order]

        delta_m_2D_bins = delta_m_2D_map_vec.reshape(-1,ncells)
        N_halos_2D_map_bins = N_halos_2D_map_vec.reshape(-1,ncells)
        self.Mh_min = Mh_min
        self.Mh_max = Mh_max
        self.delta_m_cut = delta_m_cut





        self.delta_m_2D_bin_mean = np.mean(delta_m_2D_bins, axis = 1)
        self.N_halos_2D_bin_mean = np.mean(N_halos_2D_map_bins, axis = 1)
        self.N_halos_2D_bin_error = np.std(N_halos_2D_map_bins, axis = 1)/ncells
        
        parameters_error_names = []
        for i in parameters_names:
            parameters_error_names.append(i + " error")

        all_parameters_list = {name: None for name in parameters_names}


        
        for (i, j) in zip(parameters_names, parameters_values):
            all_parameters_list[i] = j
        

        self.parameters = all_parameters_list
        
    
    def var_over_mean_simulation(self, ncells = 4096 * 2):
        delta_m_2D_map_vec = self.slab_object.delta_m_2D.flatten()
        N_halos_2D_map_vec = self.slab_object.N_halos_2D.flatten()

        order = delta_m_2D_map_vec.argsort()
        delta_m_2D_map_vec = delta_m_2D_map_vec[order]
        N_halos_2D_map_vec = N_halos_2D_map_vec[order]

        delta_m_2D_bins = delta_m_2D_map_vec.reshape(-1,ncells)
        N_halos_2D_bins = N_halos_2D_map_vec.reshape(-1,ncells)

        delta_m_2D_mean = np.mean(delta_m_2D_bins,axis=1)
        N_halos_2D_mean = np.mean(N_halos_2D_bins,axis=1)
        var = np.var(delta_m_2D_map_vec.reshape(-1,ncells),axis=1)
        var_over_mean = var/N_halos_2D_mean

        errors = (var_over_mean/np.sqrt(ncells)) * (1 + np.sqrt(var)/N_halos_2D_mean)
        return var/delta_m_2D_mean, errors






        
        
        


       

    def var_over_mean_bias_fit(self, ncells = 4096 * 2):
        '''Here, I am considering the slab is already included in some M_halo bin'''

        delta_m_2D_map_vec = self.slab_object.delta_m_2D.flatten()
        N_halos_2D_map_vec = self.slab_object.N_halos_2D.flatten()

        order = delta_m_2D_map_vec.argsort()
        delta_m_2D_map_vec = delta_m_2D_map_vec[order]
        N_halos_2D_map_vec = N_halos_2D_map_vec[order]

        delta_m_2D_bins = delta_m_2D_map_vec.reshape(-1,ncells)
        N_halos_2D_bins = N_halos_2D_map_vec.reshape(-1,ncells)

        
        
        N_halos_2D_model = self.bias_model.function(delta_m_2D_bins, **self.parameters)
        variance_over_mean = np.mean((N_halos_2D_bins - N_halos_2D_model)**2/ N_halos_2D_model, axis = 1)
        errors = np.std((N_halos_2D_bins - N_halos_2D_model)**2/N_halos_2D_model)/np.sqrt(ncells)
        
        return variance_over_mean, errors


    


    def var_over_mean_likelihood_model(self, ncells = 4096 * 2):
        # this is not binning in delta_m_2D since the inference is performed in bins of M_halo
        delta_m_2D_map_vec = self.slab_object.delta_m_2D.flatten()
        N_halos_2D_map_vec = self.slab_object.N_halos_2D.flatten()

        order = delta_m_2D_map_vec.argsort()
        delta_m_2D_map_vec = delta_m_2D_map_vec[order]
        N_halos_2D_map_vec = N_halos_2D_map_vec[order]
        
        delta_m_2D_bins = delta_m_2D_map_vec.reshape(-1,ncells)
        N_halos_2D_bins = N_halos_2D_map_vec.reshape(-1,ncells)


        var_over_mean = np.mean(1/(1 - self.theta_model.function(delta_m_2D_bins, **self.parameters))**2, axis = 1)
        errors = np.std(1/(1 - self.theta_model.function(delta_m_2D_bins, **self.parameters))**2, axis = 1)/np.sqrt(ncells)


        return var_over_mean, errors
    

    def calculate_variance_fraction(self, ncells = 4096 * 2):
        '''In here, the nominator is purely estimated from the bias fit, and the denominator 
        is estimated from the variance. We calculate the residual for different delta_m_2D bins,
        given tha tthe slab is already in a spesific M_halo bin'''
        delta_m_2D_map_vec = self.slab_object.delta_m_2D.flatten()
        N_halos_2D_map_vec = self.slab_object.N_halos_2D.flatten()

        order = delta_m_2D_map_vec.argsort()
        delta_m_2D_map_vec = delta_m_2D_map_vec[order]
        N_halos_2D_map_vec = N_halos_2D_map_vec[order]

        delta_m_2D_bins = delta_m_2D_map_vec.reshape(-1,ncells)
        N_halos_2D_bins = N_halos_2D_map_vec.reshape(-1,ncells)


        N_halos_2D_model = self.bias_model.function(delta_m_2D_bins, **self.parameters)

        var_over_mean = 1/(1 - self.theta_model.function(delta_m_2D_bins, **self.parameters))**2
        
        
        nominator = (N_halos_2D_bins - N_halos_2D_model)**2
        denominator = N_halos_2D_model*var_over_mean

        variance_fraction = np.mean(nominator/denominator, axis = 1)
        errors = np.std(nominator/denominator, axis = 1)/np.sqrt(ncells)


        return variance_fraction, errors
    

    def calculate_mean_fraction(self, ncells = 4096 * 2):
        '''In here, the nominator is purely estimated from the bias fit, and the denominator 
        is estimated from the variance. We calculate the residual for different delta_m_2D bins,
        given tha tthe slab is already in a spesific M_halo bin'''
        delta_m_2D_map_vec = self.slab_object.delta_m_2D.flatten()
        N_halos_2D_map_vec = self.slab_object.N_halos_2D.flatten()

        order = delta_m_2D_map_vec.argsort()
        delta_m_2D_map_vec = delta_m_2D_map_vec[order]
        N_halos_2D_map_vec = N_halos_2D_map_vec[order]

        delta_m_2D_bins = delta_m_2D_map_vec.reshape(-1,ncells)
        N_halos_2D_bins = N_halos_2D_map_vec.reshape(-1,ncells)


        N_halos_2D_model = self.bias_model.function(delta_m_2D_bins, **self.parameters)

        nominator = (N_halos_2D_bins - N_halos_2D_model)
        denominator = N_halos_2D_model

        mean_fraction = np.mean(nominator/denominator, axis = 1)
        errors = np.std(nominator/denominator, axis = 1)/np.sqrt(ncells)

        return mean_fraction, errors
    
    
    
    def calculate_fits_in_bins(self):
        '''This funciton calculates all variance/mean combination for a particular ModelComparison object'''
        simulation, simulation_errors = self.var_over_mean_simulation()
        bias_fit, bias_fit_errors = self.var_over_mean_bias_fit()
        likelihood_fit, likelihood_fit_error = self.var_over_mean_likelihood_model()


        variance_fraction, variance_fraction_errors = self.calculate_variance_fraction()
        mean_fraction, mean_fraction_errors = self.calculate_mean_fraction()



        quantities = {"simulation": simulation, "bias fit": bias_fit, "likelihood fit": likelihood_fit, 
                      "variance fraction": variance_fraction, "mean fraction": mean_fraction}
        quantities_error = {"simulation": simulation_errors, "bias fit": bias_fit_errors, "likelihood fit": likelihood_fit_error, 
                            "variance fraction": variance_fraction_errors, "mean fraction": mean_fraction_errors}
        
        return quantities, quantities_error
    


    def plot_fits(self, marker = None, color = "red", ax_counts = None, ax_variance_fraction = None, ax_mean_fraction = None, plotting_other_models= False):
        quantities, quantities_errors = self.calculate_fits_in_bins()
        delta_m_2D_bin_centers = self.delta_m_2D_bin_mean
        N_halos_2D_model = self.bias_model.function(delta_m_2D_bin_centers, **self.parameters)
        if ax_counts is None:    
            fig, (ax_counts, ax_mean_fraction, ax_variance_fraction) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [2, 2, 2]}, sharex=False, figsize = (12, 9))
        
        if not plotting_other_models:
            ax_counts.errorbar(delta_m_2D_bin_centers, self.N_halos_2D_bin_mean, yerr = self.N_halos_2D_bin_error,
                          marker = "o", ms = 3, color = "black", label = "binned halo counts")

        
             
        ax_counts.plot(delta_m_2D_bin_centers, N_halos_2D_model, ls = "dashed", marker = marker, color = color, label = self.bias_model.name + " fit")
        
        # Get the current x axis limits
        x_min, x_max = ax_counts.get_xlim()
        
        # Make sure x_min is actually the minimum value by taking the minimum of the current xlim and data
        x_min = min(x_min, min(delta_m_2D_bin_centers))
        
        # Ensure the x_min is well below the minimum of your data points 
        # (giving a bit of extra space to ensure no white space at the edge)
        x_min = x_min - 0.05 * abs(x_max - x_min) 
        
        # Set the updated limits
        ax_counts.set_xlim(x_min, x_max)
        # Now apply the shading from the updated x_min to delta_m_cut
        ax_counts.axvspan(x_min, self.delta_m_cut, alpha=0.3, color='black', label='No halos region')

        ax_counts.set_ylabel(r"$N_{h}$", fontsize = 14)
        ax_counts.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        ax_counts.set_title(rf"{np.log10(self.Mh_min):.2f}< $log10(M_{{halos}})$ < {np.log10(self.Mh_max):.2f}")
        
        '''if include_simualtion:
            ax.errorbar(delta_m_2D_bin_centers, quantities["simulation"], yerr = quantities_errors["simulation"], 
                        marker = "o", color= "black", label = "Simulation")
        
        if include_bias_fit:
            ax.errorbar(delta_m_2D_bin_centers, quantities["bias fit"], yerr = quantities_errors["bias fit"], 
                        marker = "o", color= "red", label = "Bias fit")
        
        if include_likelihood_model_fit:

            ax.errorbar(delta_m_2D_bin_centers, quantities["likelihood fit"], yerr = quantities_errors["likelihood fit"], 
                        marker = "o", color= "blue", label = "Likelihood fit")
        ax.set_ylabel("Var/Mean", fontsize = 14)
        ax.set_xlabel(r"$\delta_{m}^{2D}$", fontsize = 14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        ax.axhline(1, ls = "dashed", color = "black")'''

        # For consistency, update the x_min for other subplots too
        ax_variance_fraction.set_xlim(x_min, x_max)
        


        ax_variance_fraction.errorbar(delta_m_2D_bin_centers, quantities["variance fraction"], yerr = quantities_errors["variance fraction"],
                        color = color)
        ax_variance_fraction.axvspan(x_min, self.delta_m_cut, alpha=0.3, color='black', label='No halos region')
        
        ax_variance_fraction.set_ylim(.5, 1.5)
        ax_variance_fraction.set_ylabel("variance \nfraction", fontsize = 14)
        ax_variance_fraction.axhline(1, ls = "dashed", color = "black")
        ax_variance_fraction.set_xlabel(r"$\delta_{m}^{2D}$", fontsize = 14)

        # For consistency, update the x_min for other subplots too
        
        ax_mean_fraction.set_xlim(x_min, x_max)
        
        ax_mean_fraction.errorbar(delta_m_2D_bin_centers, quantities["mean fraction"], yerr = quantities_errors["mean fraction"],
                         color = color)
        ax_mean_fraction.axvspan(x_min, self.delta_m_cut, alpha=0.3, color='black', label='No halos region')
        
        ax_mean_fraction.set_ylim(-.2, .2)
        ax_mean_fraction.set_ylabel("mean \nfraction", fontsize = 14)
        ax_mean_fraction.axhline(0, ls = "dashed", color = "black")
        plt.subplots_adjust(hspace=0.3)
        if ax_counts is None:
            return fig, ax_counts, ax_mean_fraction, ax_variance_fraction 
        else:
            
            return  ax_counts, ax_mean_fraction, ax_variance_fraction, fig
            
    
    def construct_var_mean_matrices(list_of_models):
        '''This function takes a list of models, where each model represent a particular halo mass bin. It constructs 
        matrices that include the variance over mean of various combination and residue. These matrices are used in plotting.
        Each row of a matrix represent the values calculated in a specific halo mass bin. Each column in the matrix represent 
        values calculated in a specific matter pertubation bin.'''
        n_halo_mass_bins = len(list_of_models)
        delta_m_2D_bin_edges = list_of_models[0].delta_m_2D_bin_mean
        n_delta_m_2D_bins = delta_m_2D_bin_edges.size

        var_over_mean_bias_fit_matrix = np.zeros((n_halo_mass_bins, n_delta_m_2D_bins))
        var_over_mean_likelihood_fit_matrix = np.zeros((n_halo_mass_bins, n_delta_m_2D_bins))
        var_over_mean_simulation_matrix = np.zeros((n_halo_mass_bins, n_delta_m_2D_bins))
        variance_fraction_matrix = np.zeros((n_halo_mass_bins, n_delta_m_2D_bins))
        mean_fraction_matrix = np.zeros((n_halo_mass_bins, n_delta_m_2D_bins))


        var_over_mean_bias_fit_error_matrix = np.zeros((n_halo_mass_bins, n_delta_m_2D_bins))
        var_over_mean_likelihood_fit_error_matrix = np.zeros((n_halo_mass_bins, n_delta_m_2D_bins))
        var_over_mean_simulation_error_matrix = np.zeros((n_halo_mass_bins, n_delta_m_2D_bins))
        variance_fraction_error_matrix = np.zeros((n_halo_mass_bins, n_delta_m_2D_bins))
        mean_fraction_error_matrix = np.zeros((n_halo_mass_bins, n_delta_m_2D_bins))


        for i, model in enumerate(list_of_models, 0):
            quantities, quatities_error = model.calculate_fits_in_bins()

            
            
            var_over_mean_bias_fit_matrix[i] = quantities["bias fit"]
            var_over_mean_bias_fit_error_matrix[i] = quatities_error["bias fit"]

            var_over_mean_likelihood_fit_matrix[i] = quantities["likelihood fit"]
            var_over_mean_likelihood_fit_error_matrix[i] = quatities_error["likelihood fit"]

            var_over_mean_simulation_matrix[i] = quantities["simulation"]
            var_over_mean_simulation_error_matrix[i] = quatities_error["simulation"]

            variance_fraction_matrix[i] = quantities["variance fraction"]
            variance_fraction_error_matrix[i] = quatities_error["variance fraction"]

            mean_fraction_matrix[i] = quantities["mean fraction"]
            mean_fraction_error_matrix[i] = quatities_error["mean fraction"]
        

        quantities_matrices = {"simulation": var_over_mean_simulation_matrix, "bias fit": var_over_mean_bias_fit_matrix, "likelihood fit": var_over_mean_likelihood_fit_matrix, 
                      "variance fraction": variance_fraction_matrix, "mean fraction": mean_fraction_matrix}

        quantities_error_matrices = {"simulation": var_over_mean_simulation_error_matrix, "bias fit": var_over_mean_bias_fit_error_matrix, "likelihood fit": var_over_mean_likelihood_fit_error_matrix, 
                      "variance fraction": variance_fraction_error_matrix, "mean fraction": mean_fraction_error_matrix}
        

        return quantities_matrices, quantities_error_matrices
    


    def plot_matrix(x_axis_values, matrix, matrix_error, ax_list, color = "red", rows_or_columns = "rows", linestyle = ".", label = "Simulation"):
        '''A funciton that takes a quantity matrix'''
        if rows_or_columns == "rows":
            pass
        else:
            matrix = matrix.T
            matrix_error = matrix_error.T
        
        for i, ax in enumerate(ax_list, 0):
            ax.errorbar(x_axis_values, matrix[i], yerr = matrix_error[i], 
                        marker = "o", color= color, ls = linestyle,  label = label)
    
        



    
    
    def plot_fits_in_bins(list_of_bins, M_halo_bin_edges, legend_margin = 2, fig_and_axes = None, return_legend = False, binning_type = "HaloMass", color = "red",include_simualtion = True, include_bias_fit = True, include_likelihood_model_fit = True):
                    n_halo_mass_bins = len(list_of_bins)
                    delta_m_2D_bin_edges = list_of_bins[0].delta_m_2D_bin_edges
                    n_delta_m_2D_bins = delta_m_2D_bin_edges.size - 1
                    M_halo_bin_center = (M_halo_bin_edges[1:] + M_halo_bin_edges[:-1])/2 # this need to change
                     # this need to change
                    delta_m_2D_bin_center = (delta_m_2D_bin_edges[1:] + delta_m_2D_bin_edges[:-1])/2


                    quantities_matrices, quantities_error_matrices = ModelComparison.construct_var_mean_matrices(list_of_bins)
                    

                    if binning_type == "HaloMass":
                        rows_or_columns = "rows"
                        n_panels = n_halo_mass_bins
                        graph_title = r"$\log_{10}(M_{halo})$"
                        x_axis_value = delta_m_2D_bin_center
                        x_axis_label = r"$\delta_{m}^{2D}$"
                        title_values = log_M_halo_bin_center
                    
                    elif binning_type == "MatterPerturbation":
                        n_panels = n_delta_m_2D_bins
                        rows_or_columns = "columns"
                        graph_title = r"$\delta_{m}^{2D}$"
                        x_axis_value = log_M_halo_bin_center
                        x_axis_label = r"$\log_{10}(M_{halo})$"
                        title_values = delta_m_2D_bin_center
                    

                    else:
                        raise ValueError("binning_type can either be HaloMass or MatterPerturbation")
                        
                        

                    if fig_and_axes is None:
                        fig, axes = plt.subplots(
                        n_panels * 3, 1,  # 3 rows per panel (main plot, variance fraction, mean fraction)
                        gridspec_kw={'height_ratios': [2, 2, 2] * n_panels},  # Repeat height ratios for each panel
                        sharex=True, 
                        figsize=(12 + legend_margin, 8 * n_panels )  # Adjust figure height based on the number of panels
                    )
                    else:
                        fig, axes= fig_and_axes
                



                    mean_fraction_plots = axes[np.arange(0, n_panels*3, 3)]
                    variance_fraction_plots = axes[np.arange(1, n_panels*3, 3)]
                    variance_over_mean_plots = axes[np.arange(2, n_panels*3, 3)]

                    for ax, bin in zip(mean_fraction_plots, title_values):
                        ax.set_title(graph_title + " = " + str(np.round(bin, 2)),fontsize = 10)


                    legend_entries_list = []
                    if include_simualtion:
                        ModelComparison.plot_matrix(x_axis_values=x_axis_value, ax_list = variance_over_mean_plots,matrix=quantities_matrices["simulation"], 
                                                    matrix_error=quantities_error_matrices["simulation"], color=color, label="Simulation", 
                                                    rows_or_columns=rows_or_columns, linestyle="-")
                        legend_entry = mlines.Line2D([], [], color='black', marker='o', linestyle='-', label='Simulation')
                        legend_entries_list.append(legend_entry)
                        
                    
                    if include_bias_fit:
                        ModelComparison.plot_matrix(x_axis_values=x_axis_value, ax_list = variance_over_mean_plots, matrix=quantities_matrices["bias fit"], 
                                                    matrix_error=quantities_error_matrices["bias fit"], color=color, label="Bias fit", 
                                                    rows_or_columns=rows_or_columns, linestyle="--")
                        legend_entry = mlines.Line2D([], [], color='black', marker='o', linestyle='--', label='Bias fit')
                        legend_entries_list.append(legend_entry)

                        
                        
                    
                    if include_likelihood_model_fit:
                        ModelComparison.plot_matrix(x_axis_values=x_axis_value, ax_list = variance_over_mean_plots, matrix=quantities_matrices["likelihood fit"], 
                                                    matrix_error=quantities_error_matrices["likelihood fit"], color=color, label="likelihood", 
                                                    rows_or_columns=rows_or_columns, linestyle="dotted")
                        legend_entry = mlines.Line2D([], [], color='black', marker='o', linestyle='dotted', label='likelihood fit')
                        legend_entries_list.append(legend_entry)
                        
                    if return_legend:
                        pass
                    else: 
                        axes[2].legend(handles=legend_entries_list, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
                    
                    ModelComparison.plot_matrix(x_axis_values=x_axis_value, ax_list = variance_fraction_plots, matrix=quantities_matrices["variance fraction"], 
                                                    matrix_error=quantities_error_matrices["variance fraction"], color=color, label="variance fraction", 
                                                    rows_or_columns= rows_or_columns, linestyle= "-")
                    
                    
                    ModelComparison.plot_matrix(x_axis_values=x_axis_value, ax_list = mean_fraction_plots, matrix=quantities_matrices["mean fraction"], 
                                                    matrix_error=quantities_error_matrices["mean fraction"], color=color, label="mean fraction", 
                                                    rows_or_columns=rows_or_columns, linestyle= "-")
                    
                    for (i, j, k) in zip(mean_fraction_plots, variance_fraction_plots, variance_over_mean_plots):
                        i.set_ylim(-0.5, 0.5)
                        i.axhline(0, linestyle = "--", color = "black")

                        j.set_ylim(0, 2)
                        j.axhline(1, linestyle = "--", color = "black")

                        k.axhline(1, linestyle = "--", color = "black")



                        


                    if return_legend:
                        return fig, axes, legend_entries_list
                    else:
                        return fig, axes
            



    def compare_models(list_of_models, log_M_halo_bin_edges, list_of_colors = None, binning_type = "HaloMass", legend_margin = 2):
            '''list_of_models is a list of binned models, where each list of models contain a list of ModelComparison objects,
            binning in halo mass binning scheme. This function is meant to compare a compbination of likelihood and bias models
            list_of_colors are the colors which we shall plot each of the model.'''

            if list_of_colors is None:
                range = np.arange(0, 10, (10 - 0)/len(list_of_models))
                list_of_colors = []
                for i in range:
                    list_of_colors.append(plt.cm.plasma(i / 10))
            



            legend_entries_list = []
            for i, (model, color) in enumerate(zip(list_of_models, list_of_colors), 0):
                if i == 0:
                    fig, axes, features_legend = ModelComparison.plot_fits_in_bins(model, legend_margin= legend_margin, log_M_halo_bin_edges=log_M_halo_bin_edges, 
                                                              binning_type = binning_type, color= color, return_legend=True)
                else:
                    fig, axes = ModelComparison.plot_fits_in_bins(model, log_M_halo_bin_edges=log_M_halo_bin_edges, legend_margin=legend_margin,
                                                                binning_type = binning_type, color= color, fig_and_axes= (fig, axes), return_legend=False)
                
                
                
                legend_entry = mpatches.Patch(color=color, label= model[0].bias_model.name + " + " + model[0].theta_model.name)
                legend_entries_list.append(legend_entry)
                
                
            
            axes[1].legend(handles = features_legend + legend_entries_list, loc = "upper right", bbox_to_anchor=(1.30, 1))


            return fig, axes