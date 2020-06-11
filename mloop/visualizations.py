'''
Module of classes used to create visualizations of data produced by the experiment and learners.
'''
from __future__ import absolute_import, division, print_function
__metaclass__ = type

import mloop.utilities as mlu
import mloop.learners as mll
import mloop.controllers as mlc
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl

figure_counter = 0
cmap = plt.get_cmap('hsv')
run_label = 'Run number'
cost_label = 'Cost'
generation_label = 'Generation number'
scale_param_label = 'Min (0) to max (1) parameters'
param_label = 'Parameter'
log_length_scale_label = 'Log of length scale'
noise_label = 'Noise level'
legend_loc = 2

def show_all_default_visualizations(controller, show_plots=True):
    '''
    Plots all visualizations available for a controller, and it's internal learners.
    
    Args:
        controller (Controller): The controller to extract plots from
        
    Keyword Args:
        show_plots (Controller): Determine whether to run plt.show() at the end or not. For debugging. 
    '''
    log = logging.getLogger(__name__)
    configure_plots()
    log.debug('Creating controller visualizations.')
    create_controller_visualizations(controller.total_archive_filename,
                                    file_type=controller.controller_archive_file_type)
    
    if isinstance(controller, mlc.DifferentialEvolutionController):
        log.debug('Creating differential evolution visualizations.')
        create_differential_evolution_learner_visualizations(controller.learner.total_archive_filename, 
                                                             file_type=controller.learner.learner_archive_file_type)
        
    if isinstance(controller, mlc.GaussianProcessController):
        log.debug('Creating gaussian process visualizations.')
        create_gaussian_process_learner_visualizations(controller.ml_learner.total_archive_filename, 
                                                       file_type=controller.ml_learner.learner_archive_file_type)
        
    log.info('Showing visualizations, close all to end MLOOP.')
    if show_plots:
        plt.show()

def show_all_default_visualizations_from_archive(controller_filename, learner_filename, controller_type, show_plots=True):
    log = logging.getLogger(__name__)
    configure_plots()
    log.debug('Creating controller visualizations.')
    controller_file_type = controller_filename.split(".")[-1]
    learner_file_type = learner_filename.split(".")[-1]
    create_controller_visualizations(controller_filename, file_type=controller_file_type)

    if controller_type == 'neural_net':
        log.debug('Creating neural net visualizations.')
        create_neural_net_learner_visualizations(
            learner_filename,
            file_type=learner_file_type)
    else:
        log.error('show_all_default_visualizations not implemented for type: ' + controller_type)
        raise ValueError

    log.info('Showing visualizations, close all to end MLOOP.')
    if show_plots:
        plt.show()

def _color_from_controller_name(controller_name):
    '''
    Gives a color (as a number betweeen zero an one) corresponding to each controller name string.
    '''
    global cmap
    return cmap(float(mlc.controller_dict[controller_name])/float(mlc.number_of_controllers))

def _color_list_from_num_of_params(num_of_params):
    '''
    Gives a list of colors based on the number of parameters. 
    '''
    global cmap
    return [cmap(float(x)/num_of_params) for x in range(num_of_params)]

def configure_plots():
    '''
    Configure the setting for the plots.
    '''
    mpl.rcParams['lines.linewidth'] = 2.0
    mpl.rcParams['lines.markersize'] = 6.0
    mpl.rcParams['font.size'] = 16.0
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['legend.framealpha'] = 0.5
    mpl.rcParams['legend.numpoints'] = 1
    mpl.rcParams['legend.scatterpoints'] = 1
    mpl.rcParams['legend.fontsize']= 'medium'
    
def create_controller_visualizations(filename,
                                    file_type='pkl',
                                    plot_cost_vs_run=True,
                                    plot_parameters_vs_run=True,
                                    plot_parameters_vs_cost=True):
    '''
    Runs the plots for a controller file.
    
    Args:
        filename (Optional [string]): Filename for the controller archive. 
    
    Keyword Args:
        file_type (Optional [string]): File type 'pkl' pickle, 'mat' matlab or 'txt' text.
        plot_cost_vs_run (Optional [bool]): If True plot cost versus run number, else do not. Default True. 
        plot_parameters_vs_run (Optional [bool]): If True plot parameters versus run number, else do not. Default True. 
        plot_parameters_vs_cost (Optional [bool]): If True plot parameters versus cost number, else do not. Default True. 
    '''
    visualization = ControllerVisualizer(filename,file_type=file_type)
    if plot_cost_vs_run:
        visualization.plot_cost_vs_run()
    if plot_parameters_vs_run:
        visualization.plot_parameters_vs_run()
    if plot_parameters_vs_cost:
       visualization.plot_parameters_vs_cost()

class ControllerVisualizer():
    '''
    ControllerVisualizer creates figures from a Controller Archive. 
    
    Args:
        filename (String): Filename of the GaussianProcessLearner archive.
    
    Keyword Args:
        file_type (String): Can be 'mat' for matlab, 'pkl' for pickle or 'txt' for text. Default 'pkl'. 
    
    '''
    def __init__(self, filename, 
                 file_type ='pkl', 
                 **kwargs):
        
        self.log = logging.getLogger(__name__)
        
        self.filename = str(filename)
        self.file_type = str(file_type)
        if not mlu.check_file_type_supported(self.file_type):
            self.log.error('GP training file type not supported' + repr(self.file_type))
        controller_dict = mlu.get_dict_from_file(self.filename, self.file_type)
            
        self.archive_type = controller_dict['archive_type']
        if 'archive_type' in controller_dict and not (controller_dict['archive_type'] == 'controller'):
            self.log.error('The archive appears to be the wrong type.')
            raise ValueError
        
        self.num_in_costs = int(controller_dict['num_in_costs'])
        self.num_out_params = int(controller_dict['num_out_params'])
        self.out_params = np.array(controller_dict['out_params'])
        self.out_type = [x.strip() for x in list(controller_dict['out_type'])]
        self.in_costs = np.squeeze(np.array(controller_dict['in_costs']))
        self.in_uncers = np.squeeze(np.array(controller_dict['in_uncers']))
        self.in_bads = np.squeeze(list(controller_dict['in_bads']))
        self.best_index = int(controller_dict['best_index'])
        self.num_params = int(controller_dict['num_params'])
        self.min_boundary = np.squeeze(np.array(controller_dict['min_boundary']))
        self.max_boundary = np.squeeze(np.array(controller_dict['max_boundary']))
        
        if np.all(np.isfinite(self.min_boundary)) and np.all(np.isfinite(self.min_boundary)):
            self.finite_flag = True
            self.param_scaler = lambda p: (p-self.min_boundary)/(self.max_boundary - self.min_boundary)
            self.scaled_params = np.array([self.param_scaler(self.out_params[ind,:]) for ind in range(self.num_out_params)])
        else:
            self.finite_flag = False
        
        self.unique_types = set(self.out_type)
        self.cost_colors = [_color_from_controller_name(x) for x in self.out_type]
        self.in_numbers = np.arange(1,self.num_in_costs+1)
        self.out_numbers = np.arange(1,self.num_out_params+1)
        self.param_colors = _color_list_from_num_of_params(self.num_params)
        
    def plot_cost_vs_run(self):
        '''
        Create a plot of the costs versus run number.
        '''
        global figure_counter, run_label, cost_label, legend_loc
        figure_counter += 1
        plt.figure(figure_counter)
        
        # Only plot points for which a cost was actually measured. This may not
        # be the case for all parameter sets if the optimization is still in
        # progress, or ended by a keyboard interupt, etc..
        in_numbers = self.in_numbers[:self.num_in_costs]
        in_costs = self.in_costs[:self.num_in_costs]
        in_uncers = self.in_uncers[:self.num_in_costs]
        cost_colors = self.cost_colors[:self.num_in_costs]
        
        plt.scatter(in_numbers, in_costs+in_uncers, marker='_', color='k')
        plt.scatter(in_numbers, in_costs-in_uncers, marker='_', color='k')
        plt.scatter(in_numbers, in_costs,marker='o', c=cost_colors, s=5*mpl.rcParams['lines.markersize'])
        plt.xlabel(run_label)
        plt.ylabel(cost_label)
        plt.title('Controller: Cost vs run number.')
        artists = []
        for ut in self.unique_types:
            artists.append(plt.Line2D((0,1),(0,0), color=_color_from_controller_name(ut), marker='o', linestyle=''))
        plt.legend(artists,self.unique_types,loc=legend_loc)
        
    def plot_parameters_vs_run(self):
        '''
        Create a plot of the parameters versus run number.
        '''
        global figure_counter, run_label, scale_param_label, legend_loc
        figure_counter += 1
        plt.figure(figure_counter)
        if self.finite_flag:
            for ind in range(self.num_params):
                plt.plot(self.out_numbers,self.scaled_params[:,ind],'o',color=self.param_colors[ind])
                plt.ylabel(scale_param_label)
                plt.ylim((0,1))
        else:
            for ind in range(self.num_params):
                plt.plot(self.out_numbers,self.out_params[:,ind],'o',color=self.param_colors[ind])
                plt.ylabel(run_label)
        plt.xlabel(run_label)
        
        plt.title('Controller: Parameters vs run number.')
        artists=[]
        for ind in range(self.num_params):
            artists.append(plt.Line2D((0,1),(0,0), color=self.param_colors[ind],marker='o',linestyle=''))
        plt.legend(artists,[str(x) for x in range(1,self.num_params+1)],loc=legend_loc)
        
    def plot_parameters_vs_cost(self):
        '''
        Create a plot of the parameters versus run number.
        '''
        # Only plot points for which a cost was actually measured. This may not
        # be the case for all parameter sets if the optimization is still in
        # progress, or ended by a keyboard interupt, etc..
        in_costs = self.in_costs[:self.num_in_costs]
        in_uncers = self.in_uncers[:self.num_in_costs]
        
        global figure_counter, run_label, run_label, scale_param_label, legend_loc
        figure_counter += 1
        plt.figure(figure_counter)
        
        if self.finite_flag:
            scaled_params = self.scaled_params[:self.num_in_costs,:]
            for ind in range(self.num_params):
                plt.plot(scaled_params[:,ind], in_costs + in_uncers,'_',color=self.param_colors[ind])
                plt.plot(scaled_params[:,ind], in_costs - in_uncers,'_',color=self.param_colors[ind])
                plt.plot(scaled_params[:,ind], in_costs,'o',color=self.param_colors[ind])
                plt.xlabel(scale_param_label)
                plt.xlim((0,1))
        else:
            out_params = self.out_params[:self.num_in_costs, :]
            for ind in range(self.num_params):
                plt.plot(out_params[:,ind], in_costs + in_uncers,'_',color=self.param_colors[ind])
                plt.plot(out_params[:,ind], in_costs - in_uncers,'_',color=self.param_colors[ind])
                plt.plot(out_params[:,ind], in_costs,'o',color=self.param_colors[ind])
                plt.xlabel(run_label)
        plt.ylabel(cost_label)
        plt.title('Controller: Cost vs parameters.')
        artists=[]
        for ind in range(self.num_params):
            artists.append(plt.Line2D((0,1),(0,0), color=self.param_colors[ind],marker='o',linestyle=''))
        plt.legend(artists,[str(x) for x in range(1,self.num_params+1)], loc=legend_loc)

def create_differential_evolution_learner_visualizations(filename,
                                                         file_type='pkl',
                                                         plot_params_vs_generations=True,
                                                         plot_costs_vs_generations=True):
    '''
    Runs the plots from a differential evolution learner file.
    
    Args:
        filename (Optional [string]): Filename for the differential evolution archive. Must provide datetime or filename. Default None.
        
    Keyword Args:
        file_type (Optional [string]): File type 'pkl' pickle, 'mat' matlab or 'txt' text.
        plot_params_generations (Optional [bool]): If True plot parameters vs generations, else do not. Default True. 
        plot_costs_generations (Optional [bool]): If True plot costs vs generations, else do not. Default True. 
    '''
    visualization = DifferentialEvolutionVisualizer(filename, file_type=file_type)
    if plot_params_vs_generations:
        visualization.plot_params_vs_generations()
    if plot_costs_vs_generations:
        visualization.plot_costs_vs_generations()

class DifferentialEvolutionVisualizer():
    '''
    DifferentialEvolutionVisualizer creates figures from a differential evolution archive. 
    
    Args:
        filename (String): Filename of the DifferentialEvolutionVisualizer archive.
    
    Keyword Args:
        file_type (String): Can be 'mat' for matlab, 'pkl' for pickle or 'txt' for text. Default 'pkl'. 
    
    '''
    def __init__(self, filename, 
                 file_type ='pkl', 
                 **kwargs):
        
        self.log = logging.getLogger(__name__)
        
        self.filename = str(filename)
        self.file_type = str(file_type)
        if not mlu.check_file_type_supported(self.file_type):
            self.log.error('GP training file type not supported' + repr(self.file_type))
        learner_dict = mlu.get_dict_from_file(self.filename, self.file_type)
        
        if 'archive_type' in learner_dict and not (learner_dict['archive_type'] == 'differential_evolution'):
            self.log.error('The archive appears to be the wrong type.' + repr(learner_dict['archive_type']))
            raise ValueError
        self.archive_type = learner_dict['archive_type']
        
        self.num_generations = int(learner_dict['generation_count'])
        self.num_population_members = int(learner_dict['num_population_members'])
        self.num_params = int(learner_dict['num_params'])
        self.min_boundary = np.squeeze(np.array(learner_dict['min_boundary']))
        self.max_boundary = np.squeeze(np.array(learner_dict['max_boundary']))
        self.params_generations = np.array(learner_dict['params_generations'])
        self.costs_generations = np.array(learner_dict['costs_generations'])
          
        self.finite_flag = True
        self.param_scaler = lambda p: (p-self.min_boundary)/(self.max_boundary - self.min_boundary)
        self.scaled_params_generations = np.array([[self.param_scaler(self.params_generations[inda,indb,:]) for indb in range(self.num_population_members)] for inda in range(self.num_generations)])
        
        self.gen_numbers = np.arange(1,self.num_generations+1)
        self.param_colors = _color_list_from_num_of_params(self.num_params)
        self.gen_plot = np.array([np.full(self.num_population_members, ind, dtype=int) for ind in self.gen_numbers]).flatten()
        
    def plot_costs_vs_generations(self):
        '''
        Create a plot of the costs versus run number.
        '''
        if self.costs_generations.size == 0:
            self.log.warning('Unable to plot DE: costs vs generations as the initial generation did not complete.')
            return
        
        global figure_counter, cost_label, generation_label
        figure_counter += 1
        plt.figure(figure_counter)
        plt.plot(self.gen_plot,self.costs_generations.flatten(),marker='o',linestyle='',color='k')
        plt.xlabel(generation_label)
        plt.ylabel(cost_label)
        plt.title('Differential evolution: Cost vs generation number.')
        
    def plot_params_vs_generations(self):
        '''
        Create a plot of the parameters versus run number.
        '''
        if self.params_generations.size == 0:
            self.log.warning('Unable to plot DE: params vs generations as the initial generation did not complete.')
            return
        
        global figure_counter, generation_label, scale_param_label, legend_loc
        figure_counter += 1
        plt.figure(figure_counter)
        
        for ind in range(self.num_params):
            plt.plot(self.gen_plot,self.params_generations[:,:,ind].flatten(),marker='o',linestyle='',color=self.param_colors[ind])
            plt.ylim((0,1))
        plt.xlabel(generation_label)
        plt.ylabel(scale_param_label)
        
        plt.title('Differential evolution: Params vs generation number.')
        artists=[]
        for ind in range(self.num_params):
            artists.append(plt.Line2D((0,1),(0,0), color=self.param_colors[ind],marker='o',linestyle=''))
        plt.legend(artists,[str(x) for x in range(1,self.num_params+1)],loc=legend_loc)
        
def create_gaussian_process_learner_visualizations(filename,
                                                   file_type='pkl',
                                                   plot_cross_sections=True,
                                                   plot_hyperparameters_vs_run=True):
    '''
    Runs the plots from a gaussian process learner file.
    
    Args:
        filename (Optional [string]): Filename for the gaussian process archive. Must provide datetime or filename. Default None.
        
    Keyword Args:
        file_type (Optional [string]): File type 'pkl' pickle, 'mat' matlab or 'txt' text.
        plot_cross_sections (Optional [bool]): If True plot predict landscape cross sections, else do not. Default True. 
    '''
    visualization = GaussianProcessVisualizer(filename, file_type=file_type)
    if plot_cross_sections:
        visualization.plot_cross_sections()
    if plot_hyperparameters_vs_run:
        visualization.plot_hyperparameters_vs_run()
    
class GaussianProcessVisualizer(mll.GaussianProcessLearner):
    '''
    GaussianProcessVisualizer extends of GaussianProcessLearner, designed not to be used as a learner, but to instead post process a GaussianProcessLearner archive file and produce useful data for visualization of the state of the learner. Fixes the Gaussian process hyperparameters to what was last found during the run.
    
    Args:
        filename (String): Filename of the GaussianProcessLearner archive.
    
    Keyword Args:
        file_type (String): Can be 'mat' for matlab, 'pkl' for pickle or 'txt' for text. Default 'pkl'. 
      
    '''
    
    def __init__(self, filename, file_type = 'pkl', **kwargs):
        
        super(GaussianProcessVisualizer, self).__init__(gp_training_filename = filename,
                                                        gp_training_file_type = file_type,
                                                        update_hyperparameters = False,
                                                        **kwargs)
        
        self.log = logging.getLogger(__name__)
        
        #Trust region
        self.has_trust_region = bool(np.array(self.training_dict['has_trust_region']))
        self.trust_region = np.squeeze(np.array(self.training_dict['trust_region'], dtype=float))
        
        self.create_gaussian_process()
        self.fit_gaussian_process()
        
        self.log_length_scale_history = np.log10(np.array(self.length_scale_history, dtype=float))
        self.noise_level_history = np.array(self.noise_level_history) 
        self.fit_numbers = np.arange(1,self.fit_count+1)
        
        if np.all(np.isfinite(self.min_boundary)) and np.all(np.isfinite(self.min_boundary)):
            self.finite_flag = True
            self.param_scaler = lambda p: (p-self.min_boundary)/self.diff_boundary
        else:
            self.finite_flag = False
        
        self.param_colors = _color_list_from_num_of_params(self.num_params)
        if self.has_trust_region:
            self.scaled_trust_min = self.param_scaler(np.maximum(self.best_params - self.trust_region, self.min_boundary))
            self.scaled_trust_max = self.param_scaler(np.minimum(self.best_params + self.trust_region, self.max_boundary))
        
    def run(self):
        '''
        Overides the GaussianProcessLearner multiprocessor run routine. Does nothing but makes a warning.
        '''
        self.log.warning('You should not have executed start() from the GaussianProcessVisualizer. It is not intended to be used as a independent process. Ending.')
    
      
    def return_cross_sections(self, points=100, cross_section_center=None):
        '''
        Finds the predicted global minima, then returns a list of vectors of parameters values, costs and uncertainties, corresponding to the 1D cross sections along each parameter axis through the predicted global minima.
        
        Keyword Args:
            points (int): the number of points to sample along each cross section. Default value is 100. 
            cross_section_center (array): parameter array where the centre of the cross section should be taken. If None, the parameters for the best returned cost are used.  
        
        Returns:
            a tuple (cross_arrays, cost_arrays, uncer_arrays)
            cross_parameter_arrays (list): a list of arrays for each cross section, with the values of the varied parameter going from the minimum to maximum value.
            cost_arrays (list): a list of arrays for the costs evaluated along each cross section about the minimum. 
            uncertainty_arrays (list): a list of uncertainties 
            
        '''
        points = int(points)
        if points <= 0:
            self.log.error('Points provided must be larger than zero:' + repr(points))
            raise ValueError
        
        if cross_section_center is None:
            cross_section_center = self.best_params
        else:
            cross_section_center = np.array(cross_section_center)
            if not self.check_in_boundary(cross_section_center):
                self.log.error('cross_section_center not in boundaries:' + repr(cross_section_center))
                raise ValueError
        
        cross_parameter_arrays = [ np.linspace(min_p, max_p, points) for (min_p,max_p) in zip(self.min_boundary,self.max_boundary)]
        cost_arrays = []
        uncertainty_arrays = []
        for ind in range(self.num_params):
            sample_parameters = np.array([cross_section_center for _ in range(points)])
            sample_parameters[:, ind] = cross_parameter_arrays[ind]
            (costs, uncers) = self.gaussian_process.predict(sample_parameters,return_std=True)
            cost_arrays.append(costs)
            uncertainty_arrays.append(uncers)
        cross_parameter_arrays = np.array(cross_parameter_arrays)/self.cost_scaler.scale_
        cost_arrays = self.cost_scaler.inverse_transform(np.array(cost_arrays))
        uncertainty_arrays = np.array(uncertainty_arrays)
        return (cross_parameter_arrays,cost_arrays,uncertainty_arrays) 
    
    def plot_cross_sections(self):
        '''
        Produce a figure of the cross section about best cost and parameters
        '''
        global figure_counter, legend_loc
        figure_counter += 1
        plt.figure(figure_counter)
        points = 100
        (_,cost_arrays,uncertainty_arrays) = self.return_cross_sections(points=points)
        rel_params = np.linspace(0,1,points)
        for ind in range(self.num_params):
            plt.plot(rel_params,cost_arrays[ind,:] + uncertainty_arrays[ind,:],'--',color=self.param_colors[ind])
            plt.plot(rel_params,cost_arrays[ind,:] - uncertainty_arrays[ind,:],'--',color=self.param_colors[ind])
            plt.plot(rel_params,cost_arrays[ind,:],'-',color=self.param_colors[ind])
        if self.has_trust_region:
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            ytrust = ymin + 0.1*(ymax - ymin)
            for ind in range(self.num_params):
                plt.plot([self.scaled_trust_min[ind],self.scaled_trust_max[ind]],[ytrust,ytrust],'s', color=self.param_colors[ind])
        plt.xlabel(scale_param_label)
        plt.xlim((0,1))
        plt.ylabel(cost_label)
        plt.title('GP Learner: Predicted landscape' + ('with trust regions.' if self.has_trust_region else '.'))
        artists = []
        for ind in range(self.num_params):
            artists.append(plt.Line2D((0,1),(0,0), color=self.param_colors[ind], linestyle='-'))
        plt.legend(artists,[str(x) for x in range(1,self.num_params+1)],loc=legend_loc)    
    
    '''
    Method is currently not supported. Of questionable usefulness. Not yet deleted.
        
    def plot_all_minima_vs_cost(self):
        
        #Produce figure of the all the local minima versus cost.
        
        if not self.has_all_minima:
            self.find_all_minima()
        global figure_counter, legend_loc
        figure_counter += 1
        plt.figure(figure_counter)
        self.minima_num = self.all_minima_costs.size
        scaled_minima_params = np.array([self.param_scaler(self.all_minima_parameters[ind,:]) for ind in range(self.minima_num)])
        global run_label, run_label, scale_param_label
        if self.finite_flag:
            for ind in range(self.num_params):
                plt.plot(scaled_minima_params[:,ind],self.all_minima_costs+self.all_minima_uncers,'_',color=self.param_colors[ind])
                plt.plot(scaled_minima_params[:,ind],self.all_minima_costs-self.all_minima_uncers,'_',color=self.param_colors[ind])
                plt.plot(scaled_minima_params[:,ind],self.all_minima_costs,'o',color=self.param_colors[ind])
                plt.xlabel(scale_param_label)
        else:
            for ind in range(self.num_params):
                plt.plot(self.all_minima_parameters[:,ind],self.all_minima_costs+self.all_minima_uncers,'_',color=self.param_colors[ind])
                plt.plot(self.all_minima_parameters[:,ind],self.all_minima_costs-self.all_minima_uncers,'_',color=self.param_colors[ind])
                plt.plot(self.all_minima_parameters[:,ind],self.all_minima_costs,'o',color=self.param_colors[ind])
                plt.xlabel(run_label)
        plt.xlabel(scale_param_label)
        plt.xlim((0,1))
        plt.ylabel(cost_label)
        plt.title('GP Learner: Cost vs parameters.')
        artists = []
        for ind in range(self.num_params):
            artists.append(plt.Line2D((0,1),(0,0), color=self.param_colors[ind],marker='o',linestyle=''))
        plt.legend(artists, [str(x) for x in range(1,self.num_params+1)], loc=legend_loc)
    '''
    
    def plot_hyperparameters_vs_run(self):
        global figure_counter, run_label, legend_loc, log_length_scale_label, noise_label
        figure_counter += 1
        plt.figure(figure_counter)
        if isinstance(self.length_scale, float):
            scale_num = 1
        else:
            scale_num = self.length_scale.size
        for ind in range(scale_num):
            if scale_num == 1:
                plt.plot(self.fit_numbers,self.log_length_scale_history,'o',color=self.param_colors[ind])
            else:
                plt.plot(self.fit_numbers,self.log_length_scale_history[:,ind],'o',color=self.param_colors[ind])
        plt.xlabel(run_label)
        plt.ylabel(log_length_scale_label)
        plt.title('GP Learner: Log of lengths scales vs fit number.')
        if scale_num!=1:
            artists=[]
            for ind in range(self.num_params):
                artists.append(plt.Line2D((0,1),(0,0), color=self.param_colors[ind],marker='o',linestyle=''))
            plt.legend(artists,[str(x) for x in range(1,self.num_params+1)],loc=legend_loc)
            
        if self.cost_has_noise:
            figure_counter += 1
            plt.figure(figure_counter)
            plt.figure(figure_counter)
            plt.plot(self.fit_numbers,self.noise_level_history,'o',color='k')
            plt.xlabel(run_label)
            plt.ylabel(noise_label)
            plt.title('GP Learner: Noise level vs fit number.')
            
def create_neural_net_learner_visualizations(filename,
                                             file_type='pkl',
                                             plot_cross_sections=True):
    '''
    Creates plots from a neural nets learner file.
    
    Args:
        filename (Optional [string]): Filename for the neural net archive. Must provide datetime or filename. Default None.
        
    Keyword Args:
        file_type (Optional [string]): File type 'pkl' pickle, 'mat' matlab or 'txt' text.
        plot_cross_sections (Optional [bool]): If True plot predict landscape cross sections, else do not. Default True. 
    '''
    visualization = NeuralNetVisualizer(filename, file_type=file_type)
    if plot_cross_sections:
        visualization.do_cross_sections()
    visualization.plot_surface()
    visualization.plot_density_surface()
    visualization.plot_losses()

            
class NeuralNetVisualizer(mll.NeuralNetLearner):
    '''
    NeuralNetVisualizer extends of NeuralNetLearner, designed not to be used as a learner, but to instead post process a NeuralNetLearner archive file and produce useful data for visualization of the state of the learner. 
    
    Args:
        filename (String): Filename of the GaussianProcessLearner archive.
    
    Keyword Args:
        file_type (String): Can be 'mat' for matlab, 'pkl' for pickle or 'txt' for text. Default 'pkl'. 
      
    '''
    
    def __init__(self, filename, file_type = 'pkl', **kwargs):
        
        
        
        super(NeuralNetVisualizer, self).__init__(nn_training_filename = filename,
                                                  nn_training_file_type = file_type,
                                                  update_hyperparameters = False,
                                                  **kwargs)
        
        self.log = logging.getLogger(__name__)
        
        #Trust region
        self.has_trust_region = bool(np.array(self.training_dict['has_trust_region']))
        self.trust_region = np.squeeze(np.array(self.training_dict['trust_region'], dtype=float))
        
        self.import_neural_net()
        
        if np.all(np.isfinite(self.min_boundary)) and np.all(np.isfinite(self.min_boundary)):
            self.finite_flag = True
            self.param_scaler = lambda p: (p-self.min_boundary)/self.diff_boundary
        else:
            self.finite_flag = False
        
        self.param_colors = _color_list_from_num_of_params(self.num_params)
        if self.has_trust_region:
            self.scaled_trust_min = self.param_scaler(np.maximum(self.best_params - self.trust_region, self.min_boundary))
            self.scaled_trust_max = self.param_scaler(np.minimum(self.best_params + self.trust_region, self.max_boundary))
        
    def run(self):
        '''
        Overides the GaussianProcessLearner multiprocessor run routine. Does nothing but makes a warning.
        '''
        self.log.warning('You should not have executed start() from the GaussianProcessVisualizer. It is not intended to be used as a independent process. Ending.')
    
      
    def return_cross_sections(self, points=100, cross_section_center=None):
        '''
        Finds the predicted global minima, then returns a list of vectors of parameters values, costs and uncertainties, corresponding to the 1D cross sections along each parameter axis through the predicted global minima.
        
        Keyword Args:
            points (int): the number of points to sample along each cross section. Default value is 100. 
            cross_section_center (array): parameter array where the centre of the cross section should be taken. If None, the parameters for the best returned cost are used.  
        
        Returns:
            a tuple (cross_arrays, cost_arrays, uncer_arrays)
            cross_parameter_arrays (list): a list of arrays for each cross section, with the values of the varied parameter going from the minimum to maximum value.
            cost_arrays (list): a list of arrays for the costs evaluated along each cross section about the minimum. 
            uncertainty_arrays (list): a list of uncertainties 
            
        '''
        points = int(points)
        if points <= 0:
            self.log.error('Points provided must be larger than zero:' + repr(points))
            raise ValueError
        
        if cross_section_center is None:
            cross_section_center = self.best_params
        else:
            cross_section_center = np.array(cross_section_center)
            if not self.check_in_boundary(cross_section_center):
                self.log.error('cross_section_center not in boundaries:' + repr(cross_section_center))
                raise ValueError
        
        res = []
        for net_index in range(self.num_nets):
            cross_parameter_arrays = [ np.linspace(min_p, max_p, points) for (min_p,max_p) in zip(self.min_boundary,self.max_boundary)]
            cost_arrays = []
            for ind in range(self.num_params):
                sample_parameters = np.array([cross_section_center for _ in range(points)])
                sample_parameters[:, ind] = cross_parameter_arrays[ind]
                costs = self.predict_costs_from_param_array(sample_parameters, net_index)
                cost_arrays.append(costs)
            if self.cost_scaler.scale_:
                cross_parameter_arrays = np.array(cross_parameter_arrays)/self.cost_scaler.scale_
            else:
                cross_parameter_arrays = np.array(cross_parameter_arrays)
            cost_arrays = self.cost_scaler.inverse_transform(np.array(cost_arrays))
            res.append((cross_parameter_arrays, cost_arrays))
        return res

    def do_cross_sections(self):
        '''
        Produce a figure of the cross section about best cost and parameters
        '''
        points = 100
        rel_params = np.linspace(0,1,points)
        all_cost_arrays = [a for _,a in self.return_cross_sections(points=points)]
        for net_index, cost_arrays in enumerate(all_cost_arrays):
            def prepare_plot():
                global figure_counter
                figure_counter += 1
                fig = plt.figure(figure_counter)
                axes = plt.gca()
                for ind in range(self.num_params):
                    axes.plot(rel_params,cost_arrays[ind,:],'-',color=self.param_colors[ind],label=str(ind))
                if self.has_trust_region:
                    ymin, ymax = axes.get_ylim()
                    ytrust = ymin + 0.1*(ymax - ymin)
                    for ind in range(self.num_params):
                        axes.plot([self.scaled_trust_min[ind],self.scaled_trust_max[ind]],[ytrust,ytrust],'s', color=self.param_colors[ind])
                axes.set_xlabel(scale_param_label)
                axes.set_xlim((0,1))
                axes.set_ylabel(cost_label)
                axes.set_title('NN Learner: Predicted landscape' + ('with trust regions.' if self.has_trust_region else '.') + ' (' + str(net_index) + ')')
                return fig
            prepare_plot()
            artists = []
            for ind in range(self.num_params):
                artists.append(plt.Line2D((0,1),(0,0), color=self.param_colors[ind], linestyle='-'))
            plt.legend(artists,[str(x) for x in range(1,self.num_params+1)],loc=legend_loc)
        if self.num_nets > 1:
            # And now create a plot showing the average, max and min for each cross section.
            def prepare_plot():
                global figure_counter
                figure_counter += 1
                fig = plt.figure(figure_counter)
                axes = plt.gca()
                for ind in range(self.num_params):
                    this_param_cost_array = np.array(all_cost_arrays)[:,ind,:]
                    mn = np.mean(this_param_cost_array, axis=0)
                    m = np.min(this_param_cost_array, axis=0)
                    M = np.max(this_param_cost_array, axis=0)
                    axes.plot(rel_params,mn,'-',color=self.param_colors[ind],label=str(ind))
                    axes.plot(rel_params,m,'--',color=self.param_colors[ind],label=str(ind))
                    axes.plot(rel_params,M,'--',color=self.param_colors[ind],label=str(ind))
                axes.set_xlabel(scale_param_label)
                axes.set_xlim((0,1))
                axes.set_ylabel(cost_label)
                axes.set_title('NN Learner: Average predicted landscape')
                return fig
            prepare_plot()
            for ind in range(self.num_params):
                artists.append(plt.Line2D((0,1),(0,0), color=self.param_colors[ind], linestyle='-'))
            plt.legend(artists,[str(x) for x in range(1,self.num_params+1)],loc=legend_loc)

    def plot_surface(self):
        '''
        Produce a figure of the cost surface (only works when there are 2 parameters)
        '''
        if self.num_params != 2:
            return
        global figure_counter
        figure_counter += 1
        fig = plt.figure(figure_counter)
        ax = fig.add_subplot(111, projection='3d')

        points = 50
        param_set = [ np.linspace(min_p, max_p, points) for (min_p,max_p) in zip(self.min_boundary,self.max_boundary)]
        params = [(x,y) for x in param_set[0] for y in param_set[1]]
        costs = self.predict_costs_from_param_array(params)
        ax.scatter([param[0] for param in params], [param[1] for param in params], costs)
        ax.set_zlim(top=500,bottom=0)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('cost')

        ax.scatter(self.all_params[:,0], self.all_params[:,1], self.all_costs, c='r')

    def plot_density_surface(self):
        '''
        Produce a density plot of the cost surface (only works when there are 2 parameters)
        '''
        if self.num_params != 2:
            return
        global figure_counter
        figure_counter += 1
        fig = plt.figure(figure_counter)

        points = 50
        xs, ys = np.meshgrid(
                np.linspace(self.min_boundary[0], self.max_boundary[0], points),
                np.linspace(self.min_boundary[1], self.max_boundary[1], points))
        zs_list = self.predict_costs_from_param_array(list(zip(xs.flatten(),ys.flatten())))
        zs = np.array(zs_list).reshape(points,points)
        plt.pcolormesh(xs,ys,zs)
        plt.scatter(self.all_params[:,0], self.all_params[:,1], c=self.all_costs, vmin=np.min(zs), vmax=np.max(zs), s=100)
        plt.colorbar()
        plt.xlabel("Param 0")
        plt.ylabel("Param 1")

    def plot_losses(self):
        '''
        Produce a figure of the loss as a function of training run.
        '''
        global figure_counter
        figure_counter += 1
        fig = plt.figure(figure_counter)

        losses = self.get_losses()
        plt.scatter(range(len(losses)), losses)
        plt.xlabel("Run")
        plt.ylabel("Training cost")
        plt.title('Loss vs training run.')
