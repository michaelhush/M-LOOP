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
        show_plots (bool): Determine whether to run plt.show() at the end or
            not. For debugging. Default True.
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

def show_all_default_visualizations_from_archive(controller_filename,
                                                 learner_filename,
                                                 controller_type=None,
                                                 show_plots=True,
                                                 controller_visualization_args={},
                                                 learner_visualization_args={},
                                                 learner_visualizer_init_args={}):
    '''
    Plots all visualizations available for a controller and it's learner from their archives.
    
    Args:
        controller_filename (str): The filename, inlcuding path, of the
            controller archive.
        learner_filename (str): The filename, inlcuding path, of the learner
            archive.
        
    Keyword Args:
        controller_type (str): The value of controller_type type used in the
            optimization corresponding to the learner learner archive, e.g.
            'gaussian_process', 'neural_net', or 'differential_evolution'. If
            set to None then controller_type will be determined automatically.
            Default None.
        show_plots (bool): Determine whether to run plt.show() at the end or
            not. For debugging. Default True.
        controller_visualization_args (dict): Keyword arguments to pass to the
            controller visualizer's create_visualizations() method. Default {}.
        learner_visualization_args (dict): Keyword arguments to pass to the
            learner visualizer's create_visualizations() method. Default {}.
        learner_visualizer_init_args (dict): Keyword arguments to pass to the
            learner visualizer's __init__() method. Default {}.
    '''
    log = logging.getLogger(__name__)
    configure_plots()
    
    # Create visualizations for the controller archive.
    log.debug('Creating controller visualizations.')
    create_controller_visualizations(
        controller_filename,
        **controller_visualization_args,
    )
    
    # Create visualizations for the learner archive.
    create_learner_visualizations(
        learner_filename,
        learner_visualization_args=learner_visualization_args,
        learner_visualizer_init_args=learner_visualizer_init_args,
    )

    log.info('Showing visualizations, close all to end MLOOP.')
    if show_plots:
        plt.show()

def create_learner_visualizer_from_archive(filename, controller_type=None, **kwargs):
    '''
    Create an instance of the appropriate visualizer class for a learner archive.
    
    Args:
        filename (String): Filename of the learner archive.
    
    Keyword Args:
        controller_type (String): The type of controller used during the
            optimization that created the provided learner archive. Options
            include 'gaussian_process', 'neural_net', and
            'differential_evolution'. If set to None, then controller_type will
            be determined automatically from the archive. Default None.
        **kwargs: Additional keyword arguments are passed to the visualizer's
            __init__() method.

    Returns:
        visualizer: An instance of the appropriate visualizer class for plotting
            data from filename.
    '''
    # Automatically determine controller_type if necessary.
    if controller_type is None:
        controller_type = mlu.get_controller_type_from_learner_archive(filename)
        
    # Create an instance of the appropriate visualizer class for the archive.
    log = logging.getLogger(__name__)
    if controller_type == 'neural_net':
        log.debug('Creating neural net visualizer.')
        visualizer = NeuralNetVisualizer(filename, **kwargs)
    elif controller_type == 'gaussian_process':
        log.debug('Creating gaussian process visualizer.')
        visualizer = GaussianProcessVisualizer(filename, **kwargs)
    elif controller_type == 'differential_evolution':
        log.debug('Creating differential evolution visualizer.')
        visualizer = DifferentialEvolutionVisualizer(filename, **kwargs)
    else:
        message = ('create_learner_visualizer_from_archive() not implemented '
                   'for type: {type_}.').format(type_=controller_type)
        log.error(message)
        raise ValueError(message)
    
    return visualizer

def create_learner_visualizations(filename,
                                  learner_visualization_args={},
                                  learner_visualizer_init_args={}):
    '''
    Runs the plots for a learner archive file.
    
    Args:
        filename (str): Filename for the learner archive. 
    
    Keyword Args:
        learner_visualization_args (dict): Keyword arguments to pass to the
            learner visualizer's create_visualizations() method. Default {}.
        learner_visualizer_init_args (dict): Keyword arguments to pass to the
            learner visualizer's __init__() method. Default {}.
    '''
    visualizer = create_learner_visualizer_from_archive(
        filename,
        **learner_visualizer_init_args,
    )
    visualizer.create_visualizations(**learner_visualization_args)

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

def _ensure_parameter_subset_valid(visualizer, parameter_subset):
    '''
    Make sure indices in parameter_subset are acceptable.
    
    Args:
        visualizer (ControllerVisualizer-like): An instance of one of the
            visualization classes defined in this module, which should have the
            attributes param_numbers and log.
        parameter_subset (list-like): The indices corresponding to a subset of
            the optimization parameters. The indices should be 0-based, i.e. the
            first parameter is identified with index 0. Generally the values of
            the indices in parameter_subset should be integers between 0 and the
            number of parameters minus one, inclusively.
    '''
    for ind in parameter_subset:
        if ind not in visualizer.param_numbers:
            message = '{ind} is not a valid parameter index.'.format(ind=ind)
            visualizer.log.error(message)
            raise ValueError(message)

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
                                    file_type=None,
                                    **kwargs):
    '''
    Runs the plots for a controller file.
    
    Args:
        filename (String): Filename of the controller archive.
    
    Keyword Args:
        file_type (String): Can be 'mat' for matlab, 'pkl' for pickle or 'txt'
            for text. If set to None, then the type will be determined from the
            extension in filename. Default None.
        **kwargs: Additional keyword arguments are passed to the visualizer's
            create_visualizations() method.
    '''
    visualization = ControllerVisualizer(filename,file_type=file_type)
    visualization.create_visualizations(**kwargs)

class ControllerVisualizer():
    '''
    ControllerVisualizer creates figures from a Controller Archive. 
    
    Args:
        filename (String): Filename of the controller archive.
    
    Keyword Args:
        file_type (String): Can be 'mat' for matlab, 'pkl' for pickle or 'txt'
            for text. If set to None, then the type will be determined from the
            extension in filename. Default None.
    
    '''
    def __init__(self, filename,
                 file_type=None,
                 **kwargs):
        
        self.log = logging.getLogger(__name__)
        
        self.filename = str(filename)
        # Automatically determine file_type if necessary.
        if file_type is None:
            file_type = mlu.get_file_type(self.filename)
        self.file_type = str(file_type)
        if not mlu.check_file_type_supported(self.file_type):
            self.log.error('File type not supported: ' + repr(self.file_type))
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
        
        if np.all(np.isfinite(self.min_boundary)) and np.all(np.isfinite(self.max_boundary)):
            self.finite_flag = True
            self.param_scaler = lambda p: (p-self.min_boundary)/(self.max_boundary - self.min_boundary)
            self.scaled_params = np.array([self.param_scaler(self.out_params[ind,:]) for ind in range(self.num_out_params)])
        else:
            self.finite_flag = False
        
        self.unique_types = set(self.out_type)
        self.cost_colors = [_color_from_controller_name(x) for x in self.out_type]
        self.in_numbers = np.arange(1,self.num_in_costs+1)
        self.out_numbers = np.arange(1,self.num_out_params+1)
        self.param_numbers = np.arange(self.num_params)
    
    def create_visualizations(self,
                              plot_cost_vs_run=True,
                              plot_parameters_vs_run=True,
                              plot_parameters_vs_cost=True):
        '''
        Runs the plots for a controller file.
        
        Keyword Args:
            plot_cost_vs_run (Optional [bool]): If True plot cost versus run
                number, else do not. Default True. 
            plot_parameters_vs_run (Optional [bool]): If True plot parameters
                versus run number, else do not. Default True. 
            plot_parameters_vs_cost (Optional [bool]): If True plot parameters
                versus cost number, else do not. Default True. 
        '''
        if plot_cost_vs_run:
            self.plot_cost_vs_run()
        if plot_parameters_vs_run:
            self.plot_parameters_vs_run()
        if plot_parameters_vs_cost:
            self.plot_parameters_vs_cost()
        
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
    
    def _ensure_parameter_subset_valid(self, parameter_subset):
        _ensure_parameter_subset_valid(self, parameter_subset)
        
    def plot_parameters_vs_run(self, parameter_subset=None):
        '''
        Create a plot of the parameters versus run number.
    
        Args:
            parameter_subset (list-like): The indices of parameters to plot. The
                indices should be 0-based, i.e. the first parameter is
                identified with index 0. Generally the values of the indices in
                parameter_subset should be between 0 and the number of
                parameters minus one, inclusively. If set to `None`, then all
                parameters will be plotted. Default None.
        '''
        # Get default value for parameter_subset if necessary.
        if parameter_subset is None:
            parameter_subset = self.param_numbers
        
        # Make sure that the provided parameter_subset is acceptable.
        self._ensure_parameter_subset_valid(parameter_subset)
        
        # Generate set of distinct colors for plotting.
        num_params = len(parameter_subset)
        param_colors = _color_list_from_num_of_params(num_params)
            
        global figure_counter, run_label, scale_param_label, legend_loc
        figure_counter += 1
        plt.figure(figure_counter)
        if self.finite_flag:
            for ind in range(num_params):
                param_index = parameter_subset[ind]
                color = param_colors[ind]
                plt.plot(self.out_numbers,self.scaled_params[:,param_index],'o',color=color)
                plt.ylabel(scale_param_label)
                plt.ylim((0,1))
        else:
            for ind in range(num_params):
                param_index = parameter_subset[ind]
                color = param_colors[ind]
                plt.plot(self.out_numbers,self.out_params[:,param_index],'o',color=color)
                plt.ylabel(run_label)
        plt.xlabel(run_label)
        
        plt.title('Controller: Parameters vs run number.')
        artists=[]
        for ind in range(num_params):
            color = param_colors[ind]
            artists.append(plt.Line2D((0,1),(0,0), color=color,marker='o',linestyle=''))
        plt.legend(artists,[str(x) for x in parameter_subset],loc=legend_loc)
        
    def plot_parameters_vs_cost(self, parameter_subset=None):
        '''
        Create a plot of the parameters versus run number.
    
        Args:
            parameter_subset (list-like): The indices of parameters to plot. The
                indices should be 0-based, i.e. the first parameter is
                identified with index 0. Generally the values of the indices in
                parameter_subset should be between 0 and the number of
                parameters minus one, inclusively. If set to `None`, then all
                parameters will be plotted. Default None.
        '''
        # Only plot points for which a cost was actually measured. This may not
        # be the case for all parameter sets if the optimization is still in
        # progress, or ended by a keyboard interupt, etc..
        in_costs = self.in_costs[:self.num_in_costs]
        in_uncers = self.in_uncers[:self.num_in_costs]
        
        # Get default value for parameter_subset if necessary.
        if parameter_subset is None:
            parameter_subset = self.param_numbers
        
        # Make sure that the provided parameter_subset is acceptable.
        self._ensure_parameter_subset_valid(parameter_subset)
        
        # Generate set of distinct colors for plotting.
        num_params = len(parameter_subset)
        param_colors = _color_list_from_num_of_params(num_params)
        
        global figure_counter, run_label, run_label, scale_param_label, legend_loc
        figure_counter += 1
        plt.figure(figure_counter)
        
        if self.finite_flag:
            scaled_params = self.scaled_params[:self.num_in_costs,:]
            for ind in range(num_params):
                param_index = parameter_subset[ind]
                color = param_colors[ind]
                plt.plot(scaled_params[:,param_index], in_costs + in_uncers,'_',color=color)
                plt.plot(scaled_params[:,param_index], in_costs - in_uncers,'_',color=color)
                plt.plot(scaled_params[:,param_index], in_costs,'o',color=color)
                plt.xlabel(scale_param_label)
                plt.xlim((0,1))
        else:
            out_params = self.out_params[:self.num_in_costs, :]
            for ind in range(num_params):
                param_index = parameter_subset[ind]
                color = param_colors[ind]
                plt.plot(out_params[:,param_index], in_costs + in_uncers,'_',color=color)
                plt.plot(out_params[:,param_index], in_costs - in_uncers,'_',color=color)
                plt.plot(out_params[:,param_index], in_costs,'o',color=color)
                plt.xlabel(run_label)
        plt.ylabel(cost_label)
        plt.title('Controller: Cost vs parameters.')
        artists=[]
        for ind in range(num_params):
            color = param_colors[ind]
            artists.append(plt.Line2D((0,1),(0,0), color=color,marker='o',linestyle=''))
        plt.legend(artists,[str(x) for x in parameter_subset], loc=legend_loc)

def create_differential_evolution_learner_visualizations(filename,
                                                         file_type=None,
                                                         **kwargs):
    '''
    Runs the plots from a differential evolution learner file.
    
    Args:
        filename (String): Filename for the differential evolution learner
            archive.
        
    Keyword Args:
        file_type (String): Can be 'mat' for matlab, 'pkl' for pickle or 'txt'
            for text. If set to None, then the type will be determined from the
            extension in filename. Default None.
        **kwargs: Additional keyword arguments are passed to the visualizer's
            create_visualizations() method.
    '''
    visualization = DifferentialEvolutionVisualizer(filename, file_type=file_type)
    visualization.create_visualizations(**kwargs)

class DifferentialEvolutionVisualizer():
    '''
    DifferentialEvolutionVisualizer creates figures from a differential evolution archive. 
    
    Args:
        filename (String): Filename of the DifferentialEvolutionVisualizer archive.
    
    Keyword Args:
        file_type (String): Can be 'mat' for matlab, 'pkl' for pickle or 'txt'
            for text. If set to None, then the type will be determined from the
            extension in filename. Default None.
    
    '''
    def __init__(self, filename,
                 file_type=None,
                 **kwargs):
        
        self.log = logging.getLogger(__name__)
        
        self.filename = str(filename)
        # Automatically determine file_type if necessary.
        if file_type is None:
            file_type = mlu.get_file_type(self.filename)
        self.file_type = str(file_type)
        if not mlu.check_file_type_supported(self.file_type):
            self.log.error('File type not supported: ' + repr(self.file_type))
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
        self.param_numbers = np.arange(self.num_params)
        
        self.gen_numbers = np.arange(1,self.num_generations+1)
        self.param_colors = _color_list_from_num_of_params(self.num_params)
        self.gen_plot = np.array([np.full(self.num_population_members, ind, dtype=int) for ind in self.gen_numbers]).flatten()
        
    def create_visualizations(self,
                              plot_params_vs_generations=True,
                              plot_costs_vs_generations=True):
        '''
        Runs the plots from a differential evolution learner file.
            
        Keyword Args:
            plot_params_generations (Optional [bool]): If True plot parameters
                vs generations, else do not. Default True. 
            plot_costs_generations (Optional [bool]): If True plot costs vs
                generations, else do not. Default True. 
        '''
        if plot_params_vs_generations:
            self.plot_params_vs_generations()
        if plot_costs_vs_generations:
            self.plot_costs_vs_generations()
        
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
    
    def _ensure_parameter_subset_valid(self, parameter_subset):
        _ensure_parameter_subset_valid(self, parameter_subset)
        
    def plot_params_vs_generations(self, parameter_subset=None):
        '''
        Create a plot of the parameters versus run number.
    
        Args:
            parameter_subset (list-like): The indices of parameters to plot. The
                indices should be 0-based, i.e. the first parameter is
                identified with index 0. Generally the values of the indices in
                parameter_subset should be between 0 and the number of
                parameters minus one, inclusively. If set to `None`, then all
                parameters will be plotted. Default None.
        '''
        # Get default value for parameter_subset if necessary.
        if parameter_subset is None:
            parameter_subset = self.param_numbers
        
        # Make sure that the provided parameter_subset is acceptable.
        self._ensure_parameter_subset_valid(parameter_subset)
        
        # Generate set of distinct colors for plotting.
        num_params = len(parameter_subset)
        param_colors = _color_list_from_num_of_params(num_params)
        
        if self.params_generations.size == 0:
            self.log.warning('Unable to plot DE: params vs generations as the initial generation did not complete.')
            return
        
        global figure_counter, generation_label, scale_param_label, legend_loc
        figure_counter += 1
        plt.figure(figure_counter)
        
        artists=[]
        for ind in range(num_params):
            param_index = parameter_subset[ind]
            color = param_colors[ind]
            plt.plot(self.gen_plot,self.params_generations[:,:,param_index].flatten(),marker='o',linestyle='',color=color)
            artists.append(plt.Line2D((0,1),(0,0), color=color,marker='o',linestyle=''))
            plt.ylim((0,1))
        
        plt.title('Differential evolution: Params vs generation number.') 
        plt.xlabel(generation_label)
        plt.ylabel(scale_param_label)           
        plt.legend(artists,[str(x) for x in parameter_subset],loc=legend_loc)
        
def create_gaussian_process_learner_visualizations(filename,
                                                   file_type=None,
                                                   **kwargs):
    '''
    Runs the plots from a gaussian process learner file.
    
    Args:
        filename (String): Filename for the gaussian process learner archive.
        
    Keyword Args:
        file_type (String): Can be 'mat' for matlab, 'pkl' for pickle or 'txt'
            for text. If set to None, then the type will be determined from the
            extension in filename. Default None.
        **kwargs: Additional keyword arguments are passed to the visualizer's
            create_visualizations() method.
    '''
    visualization = GaussianProcessVisualizer(filename, file_type=file_type)
    visualization.create_visualizations(**kwargs)
    
class GaussianProcessVisualizer(mll.GaussianProcessLearner):
    '''
    GaussianProcessVisualizer extends of GaussianProcessLearner, designed not to be used as a learner, but to instead post process a GaussianProcessLearner archive file and produce useful data for visualization of the state of the learner. Fixes the Gaussian process hyperparameters to what was last found during the run.
    
    Args:
        filename (String): Filename of the GaussianProcessLearner archive.
    
    Keyword Args:
        file_type (String): Can be 'mat' for matlab, 'pkl' for pickle or 'txt'
            for text. If set to None, then the type will be determined from the
            extension in filename. Default None.
      
    '''
    
    def __init__(self, filename, file_type=None, **kwargs):
        
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
        
        self.param_numbers = np.arange(self.num_params)
        self.log_length_scale_history = np.log10(np.array(self.length_scale_history, dtype=float))
        self.noise_level_history = np.array(self.noise_level_history) 
        self.fit_numbers = np.arange(1,self.fit_count+1)
        
        if np.all(np.isfinite(self.min_boundary)) and np.all(np.isfinite(self.max_boundary)):
            self.finite_flag = True
            self.param_scaler = lambda p: (p-self.min_boundary)/self.diff_boundary
        else:
            self.finite_flag = False
        
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
    
    def create_visualizations(self,
                              plot_cross_sections=True,
                              plot_hyperparameters_vs_run=True):
        '''
        Runs the plots from a gaussian process learner file.
            
        Keyword Args:
            plot_cross_sections (Optional [bool]): If True plot predicted
                landscape cross sections, else do not. Default True. 
            plot_hyperparameters_vs_run (Optional [bool]): If True plot fitted
                hyperparameters as a function of run number, else do not.
                Default True.
        '''
        if plot_cross_sections:
            self.plot_cross_sections()
        if plot_hyperparameters_vs_run:
            self.plot_hyperparameters_vs_run()
    
    def _ensure_parameter_subset_valid(self, parameter_subset):
        _ensure_parameter_subset_valid(self, parameter_subset)
    
    def plot_cross_sections(self, parameter_subset=None):
        '''
        Produce a figure of the cross section about best cost and parameters.
    
        Args:
            parameter_subset (list-like): The indices of parameters to plot. The
                indices should be 0-based, i.e. the first parameter is
                identified with index 0. Generally the values of the indices in
                parameter_subset should be between 0 and the number of
                parameters minus one, inclusively. If set to `None`, then all
                parameters will be plotted. Default None.
        '''
        # Get default value for parameter_subset if necessary.
        if parameter_subset is None:
            parameter_subset = self.param_numbers
        
        # Make sure that the provided parameter_subset is acceptable.
        self._ensure_parameter_subset_valid(parameter_subset)
        
        # Generate set of distinct colors for plotting.
        num_params = len(parameter_subset)
        param_colors = _color_list_from_num_of_params(num_params)
        
        global figure_counter, legend_loc
        figure_counter += 1
        plt.figure(figure_counter)
        points = 100
        (_,cost_arrays,uncertainty_arrays) = self.return_cross_sections(points=points)
        rel_params = np.linspace(0,1,points)
        for ind in range(num_params):
            param_index = parameter_subset[ind]
            color = param_colors[ind]
            plt.plot(rel_params,cost_arrays[param_index,:] + uncertainty_arrays[param_index,:],'--',color=color)
            plt.plot(rel_params,cost_arrays[param_index,:] - uncertainty_arrays[param_index,:],'--',color=color)
            plt.plot(rel_params,cost_arrays[param_index,:],'-',color=color)
        if self.has_trust_region:
            axes = plt.gca()
            ymin, ymax = axes.get_ylim()
            ytrust = ymin + 0.1*(ymax - ymin)
            for ind in range(num_params):
                param_index = parameter_subset[ind]
                color = param_colors[ind]
                plt.plot([self.scaled_trust_min[param_index],self.scaled_trust_max[param_index]],[ytrust,ytrust],'s', color=color)
        plt.xlabel(scale_param_label)
        plt.xlim((0,1))
        plt.ylabel(cost_label)
        plt.title('GP Learner: Predicted landscape' + ('with trust regions.' if self.has_trust_region else '.'))
        artists = []
        for ind in range(num_params):
            color = param_colors[ind]
            artists.append(plt.Line2D((0,1),(0,0), color=color, linestyle='-'))
        plt.legend(artists,[str(x) for x in parameter_subset],loc=legend_loc)    
    
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
        plt.legend(artists, [str(x) for x in range(self.num_params)], loc=legend_loc)
    '''
    
    def plot_hyperparameters_vs_run(self, parameter_subset=None):
        '''
        Produce a figure of the hyperparameters as a function of run number.
        
        This method also plots the fitted noise level as a function of run
        number if the cost has noise.
    
        Args:
            parameter_subset (list-like): The indices of parameters to plot. The
                indices should be 0-based, i.e. the first parameter is
                identified with index 0. Generally the values of the indices in
                parameter_subset should be between 0 and the number of
                parameters minus one, inclusively. If set to `None`, then all
                parameters will be plotted. Default None.
        '''
        # Get default value for parameter_subset if necessary.
        if parameter_subset is None:
            parameter_subset = self.param_numbers
        
        # Make sure that the provided parameter_subset is acceptable.
        self._ensure_parameter_subset_valid(parameter_subset)
        
        # Generate set of distinct colors for plotting.
        num_params = len(parameter_subset)
        param_colors = _color_list_from_num_of_params(num_params)
        
        global figure_counter, run_label, legend_loc, log_length_scale_label, noise_label
        figure_counter += 1
        plt.figure(figure_counter)
        
        artists=[]
        for ind in range(num_params):
            param_index = parameter_subset[ind]
            color = param_colors[ind]
            if self.num_params == 1:
                plt.plot(self.fit_numbers,self.log_length_scale_history,'o',color=color)
            else:
                plt.plot(self.fit_numbers,self.log_length_scale_history[:,param_index],'o',color=color)
            artists.append(plt.Line2D((0,1),(0,0), color=color,marker='o',linestyle=''))
            
        plt.legend(artists,[str(x) for x in parameter_subset],loc=legend_loc)
        plt.xlabel(run_label)
        plt.ylabel(log_length_scale_label)
        plt.title('GP Learner: Log of lengths scales vs fit number.')
        
        # Make plot of noise level vs run number if cost has noise. 
        if self.cost_has_noise:
            figure_counter += 1
            plt.figure(figure_counter)
            plt.figure(figure_counter)
            plt.plot(self.fit_numbers,self.noise_level_history,'o',color='k')
            plt.xlabel(run_label)
            plt.ylabel(noise_label)
            plt.title('GP Learner: Noise level vs fit number.')
            
def create_neural_net_learner_visualizations(filename,
                                             file_type=None,
                                             **kwargs):
    '''
    Creates plots from a neural net's learner file.
    
    Args:
        filename (String): Filename for the neural net learner archive.
        
    Keyword Args:
        file_type (String): Can be 'mat' for matlab, 'pkl' for pickle or 'txt'
            for text. If set to None, then the type will be determined from the
            extension in filename. Default None.
        **kwargs: Additional keyword arguments are passed to the visualizer's
            create_visualizations() method.
    '''
    visualization = NeuralNetVisualizer(filename, file_type=file_type)
    visualization.create_visualizations(**kwargs)

            
class NeuralNetVisualizer(mll.NeuralNetLearner):
    '''
    NeuralNetVisualizer extends of NeuralNetLearner, designed not to be used as a learner, but to instead post process a NeuralNetLearner archive file and produce useful data for visualization of the state of the learner. 
    
    Args:
        filename (String): Filename of the NeuralNetLearner archive.
    
    Keyword Args:
        file_type (String): Can be 'mat' for matlab, 'pkl' for pickle or 'txt'
            for text. If set to None, then the type will be determined from the
            extension in filename. Default None.
    '''
    
    def __init__(self, filename, file_type = None, **kwargs):
        
        
        
        super(NeuralNetVisualizer, self).__init__(nn_training_filename = filename,
                                                  nn_training_file_type = file_type,
                                                  update_hyperparameters = False,
                                                  **kwargs)
        
        self.log = logging.getLogger(__name__)
        
        #Trust region
        self.has_trust_region = bool(np.array(self.training_dict['has_trust_region']))
        self.trust_region = np.squeeze(np.array(self.training_dict['trust_region'], dtype=float))
        
        self.import_neural_net()
        
        if np.all(np.isfinite(self.min_boundary)) and np.all(np.isfinite(self.max_boundary)):
            self.finite_flag = True
            self.param_scaler = lambda p: (p-self.min_boundary)/self.diff_boundary
        else:
            self.finite_flag = False
        
        if self.has_trust_region:
            self.scaled_trust_min = self.param_scaler(np.maximum(self.best_params - self.trust_region, self.min_boundary))
            self.scaled_trust_max = self.param_scaler(np.minimum(self.best_params + self.trust_region, self.max_boundary))
            
        self.param_numbers = np.arange(self.num_params)
        
    def run(self):
        '''
        Overides the GaussianProcessLearner multiprocessor run routine. Does nothing but makes a warning.
        '''
        self.log.warning('You should not have executed start() from the GaussianProcessVisualizer. It is not intended to be used as a independent process. Ending.')
    
    def create_visualizations(self, plot_cross_sections=True):
        '''
        Creates plots from a neural net's learner file.
            
        Keyword Args:
            plot_cross_sections (Optional [bool]): If True plot predicted
                landscape cross sections, else do not. Default True. 
        '''
        if plot_cross_sections:
            self.do_cross_sections()
        self.plot_surface()
        self.plot_density_surface()
        self.plot_losses()
    
      
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
    
    def _ensure_parameter_subset_valid(self, parameter_subset):
        _ensure_parameter_subset_valid(self, parameter_subset)

    def do_cross_sections(self, parameter_subset=None):
        '''
        Produce a figure of the cross section about best cost and parameters.
    
        Args:
            parameter_subset (list-like): The indices of parameters to plot. The
                indices should be 0-based, i.e. the first parameter is
                identified with index 0. Generally the values of the indices in
                parameter_subset should be between 0 and the number of
                parameters minus one, inclusively. If set to `None`, then all
                parameters will be plotted. Default None.
        '''
        # Get default value for parameter_subset if necessary.
        if parameter_subset is None:
            parameter_subset = self.param_numbers
        
        # Make sure that the provided parameter_subset is acceptable.
        self._ensure_parameter_subset_valid(parameter_subset)
        
        # Generate set of distinct colors for plotting.
        num_params = len(parameter_subset)
        param_colors = _color_list_from_num_of_params(num_params)
        
        points = 100
        rel_params = np.linspace(0,1,points)
        all_cost_arrays = [a for _,a in self.return_cross_sections(points=points)]
        for net_index, cost_arrays in enumerate(all_cost_arrays):
            def prepare_plot():
                global figure_counter
                figure_counter += 1
                fig = plt.figure(figure_counter)
                axes = plt.gca()
                for ind in range(num_params):
                    param_index = parameter_subset[ind]
                    color = param_colors[ind]
                    axes.plot(rel_params,cost_arrays[param_index,:],'-',color=color,label=str(param_index))
                if self.has_trust_region:
                    ymin, ymax = axes.get_ylim()
                    ytrust = ymin + 0.1*(ymax - ymin)
                    for ind in range(num_params):
                        param_index = parameter_subset[ind]
                        color = param_colors[ind]
                        axes.plot([self.scaled_trust_min[param_index],self.scaled_trust_max[param_index]],[ytrust,ytrust],'s', color=color)
                axes.set_xlabel(scale_param_label)
                axes.set_xlim((0,1))
                axes.set_ylabel(cost_label)
                axes.set_title('NN Learner: Predicted landscape' + (' with trust regions.' if self.has_trust_region else '.') + ' (' + str(net_index) + ')')
                return fig
            prepare_plot()
            artists = []
            for ind in range(num_params):
                color = param_colors[ind]
                artists.append(plt.Line2D((0,1),(0,0), color=color, linestyle='-'))
            plt.legend(artists,[str(x) for x in parameter_subset],loc=legend_loc)
        if self.num_nets > 1:
            # And now create a plot showing the average, max and min for each cross section.
            def prepare_plot():
                global figure_counter
                figure_counter += 1
                fig = plt.figure(figure_counter)
                axes = plt.gca()
                for ind in range(num_params):
                    param_index = parameter_subset[ind]
                    color = param_colors[ind]
                    this_param_cost_array = np.array(all_cost_arrays)[:,param_index,:]
                    mn = np.mean(this_param_cost_array, axis=0)
                    m = np.min(this_param_cost_array, axis=0)
                    M = np.max(this_param_cost_array, axis=0)
                    axes.plot(rel_params,mn,'-',color=color,label=str(param_index))
                    axes.plot(rel_params,m,'--',color=color,label=str(param_index))
                    axes.plot(rel_params,M,'--',color=color,label=str(param_index))
                axes.set_xlabel(scale_param_label)
                axes.set_xlim((0,1))
                axes.set_ylabel(cost_label)
                axes.set_title('NN Learner: Average predicted landscape')
                return fig
            prepare_plot()
            for ind in range(num_params):
                color = param_colors[ind]
                artists.append(plt.Line2D((0,1),(0,0), color=color, linestyle='-'))
            plt.legend(artists,[str(x) for x in parameter_subset],loc=legend_loc)

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
