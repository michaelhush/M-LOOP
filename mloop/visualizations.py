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
import warnings

figure_counter = 0
cmap = plt.get_cmap('hsv')
run_label = 'Run number'
fit_label = 'Fit number'
cost_label = 'Cost'
generation_label = 'Generation number'
scale_param_label = 'Min (0) to max (1) parameters'
param_label = 'Parameter'
log_length_scale_label = 'Log of length scale'
noise_label = 'Noise level'
_DEFAULT_LEGEND_LOC = 2
legend_loc = _DEFAULT_LEGEND_LOC

def set_legend_location(loc=None):
    '''
    Set the location of the legend in future figures.

    Note that this function doesn't change the location of legends in existing
    figures. It only changes where legends will appear in figures generated
    after the call to this function. If called without arguments, the legend
    location for future figures will revert to its default value.

    Keyword Args:
        loc (Optional str, int, or pair of floats): The value to use for loc in
            the calls to matplotlib's legend(). Can be e.g. 2, 'upper right',
            (1, 0). See matplotlib's documentation for more options and
            additional information. If set to None then the legend location will
            be set back to its default value. Default None.
    '''
    # Set default value for loc if necessary.
    if loc is None:
        loc = _DEFAULT_LEGEND_LOC

    # Update the global used for setting the legend location.
    global legend_loc
    legend_loc = loc

def show_all_default_visualizations(controller,
                                    show_plots=True,
                                    max_parameters_per_plot=None):
    '''
    Plots all visualizations available for a controller, and it's internal learners.

    Args:
        controller (Controller): The controller to extract plots from

    Keyword Args:
        show_plots (Optional, bool): Determine whether to run plt.show() at the
            end or not. For debugging. Default True.
        max_parameters_per_plot (Optional, int): The maximum number of
            parameters to include in plots that display the values of
            parameters. If the number of parameters is larger than
            parameters_per_plot, then the parameters will be divided into groups
            and each group will be plotted in its own figure. If set to None,
            then all parameters will be included in the same plot regardless of
            how many there are. Default None.
    '''
    log = logging.getLogger(__name__)
    configure_plots()
    log.debug('Creating controller visualizations.')
    create_controller_visualizations(
        controller.total_archive_filename,
        max_parameters_per_plot=max_parameters_per_plot,
    )

    # For machine learning controllers, the controller.learner is actually the
    # learner for the trainer while controller.ml_learner is the machine
    # learning controller. For other controllers, controller.learner is the
    # actual learner.
    try:
        learner_archive_filename = controller.ml_learner.total_archive_filename
    except AttributeError:
        learner_archive_filename = controller.learner.total_archive_filename

    log.debug('Creating learner visualizations.')
    create_learner_visualizations(
        learner_archive_filename,
        max_parameters_per_plot=max_parameters_per_plot,
    )

    log.info('Showing visualizations, close all to end M-LOOP.')
    if show_plots:
        plt.show()

def show_all_default_visualizations_from_archive(controller_filename,
                                                 learner_filename,
                                                 controller_type=None,
                                                 show_plots=True,
                                                 max_parameters_per_plot=None,
                                                 controller_visualization_kwargs=None,
                                                 learner_visualization_kwargs=None,
                                                 learner_visualizer_init_kwargs=None):
    '''
    Plots all visualizations available for a controller and its learner from their archives.

    Args:
        controller_filename (str): The filename, including path, of the
            controller archive.
        learner_filename (str): The filename, including path, of the learner
            archive.

    Keyword Args:
        controller_type (str): The value of controller_type type used in the
            optimization corresponding to the learner learner archive, e.g.
            'gaussian_process', 'neural_net', or 'differential_evolution'. If
            set to None then controller_type will be determined automatically.
            Default None.
        show_plots (bool): Determine whether to run plt.show() at the end or
            not. For debugging. Default True.
        max_parameters_per_plot (Optional [int]): The maximum number of
            parameters to include in plots that display the values of
            parameters. If the number of parameters is larger than
            parameters_per_plot, then the parameters will be divided into groups
            and each group will be plotted in its own figure. If set to None,
            then all parameters will be included in the same plot regardless of
            how many there are. If a value for max_parameters_per_plot is
            included in controller_visualization_kwargs, then the value in that
            dictionary will override this setting. The same applies to
            learner_visualization_kwargs. Default None.
        controller_visualization_kwargs (dict): Keyword arguments to pass to the
            controller visualizer's create_visualizations() method. If set to
            None, no additional keyword arguments will be passed. Default None.
        learner_visualization_kwargs (dict): Keyword arguments to pass to the
            learner visualizer's create_visualizations() method.  If set to
            None, no additional keyword arguments will be passed. Default None.
        learner_visualizer_init_kwargs (dict): Keyword arguments to pass to the
            learner visualizer's __init__() method. If set to None, no
            additional keyword arguments will be passed. Default None.
    '''
    # Set default value for controller_visualization_kwargs if necessary.
    if controller_visualization_kwargs is None:
        controller_visualization_kwargs = {}

    # Update controller_visualization_kwargs with max_parameters_per_plot if
    # necessary.
    if 'max_parameters_per_plot' not in controller_visualization_kwargs:
        controller_visualization_kwargs['max_parameters_per_plot'] = max_parameters_per_plot

    log = logging.getLogger(__name__)
    configure_plots()

    # Create visualizations for the controller archive.
    log.debug('Creating controller visualizations.')
    create_controller_visualizations(
        controller_filename,
        **controller_visualization_kwargs,
    )

    # Create visualizations for the learner archive.
    create_learner_visualizations(
        learner_filename,
        max_parameters_per_plot=max_parameters_per_plot,
        learner_visualization_kwargs=learner_visualization_kwargs,
        learner_visualizer_init_kwargs=learner_visualizer_init_kwargs,
    )

    log.info('Showing visualizations, close all to end M-LOOP.')
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
                                  max_parameters_per_plot=None,
                                  learner_visualization_kwargs=None,
                                  learner_visualizer_init_kwargs=None):
    '''
    Runs the plots for a learner archive file.

    Args:
        filename (str): Filename for the learner archive.

    Keyword Args:
        max_parameters_per_plot (Optional [int]): The maximum number of
            parameters to include in plots that display the values of
            parameters. If the number of parameters is larger than
            parameters_per_plot, then the parameters will be divided into groups
            and each group will be plotted in its own figure. If set to None,
            then all parameters will be included in the same plot regardless of
            how many there are. If a value for max_parameters_per_plot is
            included in learner_visualization_kwargs, then the value in that
            dictionary will override this setting. Default None.
        learner_visualization_kwargs (dict): Keyword arguments to pass to the
            learner visualizer's create_visualizations() method.  If set to
            None, no additional keyword arguments will be passed. Default None.
        learner_visualizer_init_kwargs (dict): Keyword arguments to pass to the
            learner visualizer's __init__() method. If set to None, no
            additional keyword arguments will be passed. Default None.
    '''
    # Set default values as necessary.
    if learner_visualization_kwargs is None:
        learner_visualization_kwargs = {}
    if learner_visualizer_init_kwargs is None:
        learner_visualizer_init_kwargs = {}

    # Update controller_visualization_kwargs with max_parameters_per_plot if
    # necessary.
    if 'max_parameters_per_plot' not in learner_visualization_kwargs:
        learner_visualization_kwargs['max_parameters_per_plot'] = max_parameters_per_plot

    # Create a visualizer and have it make the plots.
    visualizer = create_learner_visualizer_from_archive(
        filename,
        **learner_visualizer_init_kwargs,
    )
    visualizer.create_visualizations(**learner_visualization_kwargs)

def _color_from_controller_name(controller_name):
    '''
    Gives a color (as a number between zero an one) corresponding to each controller name string.
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

    Note that the data from the training archive, if one was provided to the
    learner at the beginning of the optimization, is NOT included in the
    controller archive generated during the optimization. Therefore any data
    from the training archive is not included in the plots generated by this
    class. This is in contrast to some of the learner visualizer classes.

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
        self.param_names = mlu._param_names_from_file_dict(controller_dict)

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
                              plot_parameters_vs_cost=True,
                              max_parameters_per_plot=None):
        '''
        Runs the plots for a controller file.

        Keyword Args:
            plot_cost_vs_run (Optional [bool]): If True plot cost versus run
                number, else do not. Default True.
            plot_parameters_vs_run (Optional [bool]): If True plot parameters
                versus run number, else do not. Default True.
            plot_parameters_vs_cost (Optional [bool]): If True plot parameters
                versus cost number, else do not. Default True.
            max_parameters_per_plot (Optional [int]): The maximum number of
                parameters to include in plots that display the values of
                parameters. If the number of parameters is larger than
                parameters_per_plot, then the parameters will be divided into
                groups and each group will be plotted in its own figure. If set
                to None, then all parameters will be included in the same plot
                regardless of how many there are. Default None.
        '''
        parameter_chunks = mlu.chunk_list(
            self.param_numbers,
            max_parameters_per_plot,
        )

        if plot_cost_vs_run:
            self.plot_cost_vs_run()

        if plot_parameters_vs_run:
            for parameter_chunk in parameter_chunks:
                self.plot_parameters_vs_run(parameter_subset=parameter_chunk)

        if plot_parameters_vs_cost:
            for parameter_chunk in parameter_chunks:
                self.plot_parameters_vs_cost(parameter_subset=parameter_chunk)

    def plot_cost_vs_run(self):
        '''
        Create a plot of the costs versus run number.

        Note that the data from the training archive, if one was provided to the
        learner at the beginning of the optimization, will NOT be plotted here.
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

        Note that the data from the training archive, if one was provided to the
        learner at the beginning of the optimization, will NOT be plotted here.

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
        legend_labels = mlu._generate_legend_labels(
            parameter_subset,
            self.param_names,
        )
        plt.legend(artists, legend_labels ,loc=legend_loc)

    def plot_parameters_vs_cost(self, parameter_subset=None):
        '''
        Create a plot of the parameters versus run number.

        Note that the data from the training archive, if one was provided to the
        learner at the beginning of the optimization, will NOT be plotted here.

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
        legend_labels = mlu._generate_legend_labels(
            parameter_subset,
            self.param_names,
        )
        plt.legend(artists, legend_labels ,loc=legend_loc)

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
        self.param_names = mlu._param_names_from_file_dict(learner_dict)
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
                              plot_costs_vs_generations=True,
                              max_parameters_per_plot=None):
        '''
        Runs the plots from a differential evolution learner file.

        Keyword Args:
            plot_params_generations (Optional [bool]): If True plot parameters
                vs generations, else do not. Default True.
            plot_costs_generations (Optional [bool]): If True plot costs vs
                generations, else do not. Default True.
            max_parameters_per_plot (Optional [int]): The maximum number of
                parameters to include in plots that display the values of
                parameters. If the number of parameters is larger than
                parameters_per_plot, then the parameters will be divided into
                groups and each group will be plotted in its own figure. If set
                to None, then all parameters will be included in the same plot
                regardless of how many there are. Default None.
        '''
        parameter_chunks = mlu.chunk_list(
            self.param_numbers,
            max_parameters_per_plot,
        )

        if plot_params_vs_generations:
            for parameter_chunk in parameter_chunks:
                self.plot_params_vs_generations(
                    parameter_subset=parameter_chunk,
                )

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
        legend_labels = mlu._generate_legend_labels(
            parameter_subset,
            self.param_names,
        )
        plt.legend(artists, legend_labels ,loc=legend_loc)

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

    If a training archive was provided at the start of the optimization as
    `gp_training_filename` and that training archive was generated by a Gaussian
    process optimization, then some of its data is saved in the new learner
    archive generated during the optimization. That implies that some of the
    data, such as fitted hyperparameter values, from the training archive will
    be included in the plots generated by this class.

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
                                                        gp_training_override_kwargs=True,
                                                        update_hyperparameters = False,
                                                        learner_archive_filename=None,
                                                        **kwargs)

        self.log = logging.getLogger(__name__)
        training_dict = self.training_dict

        # Optimization options not loaded by parent class.
        self.param_names = mlu._param_names_from_file_dict(training_dict)
        self.cost_has_noise = bool(training_dict['cost_has_noise'])
        #Trust region
        self.has_trust_region = bool(np.array(training_dict['has_trust_region']))
        self.trust_region = np.squeeze(np.array(training_dict['trust_region'], dtype=float))
        # Try to extract options not present in archives from M-LOOP <= 3.1.1
        if 'length_scale_bounds' in training_dict:
            self.length_scale_bounds = mlu.safe_cast_to_array(training_dict['length_scale_bounds'])
        if 'noise_level_bounds' in training_dict:
            self.noise_level_bounds = mlu.safe_cast_to_array(training_dict['noise_level_bounds'])

        self.fit_gaussian_process()

        self.param_numbers = np.arange(self.num_params)
        self.log_length_scale_history = np.log10(np.array(self.length_scale_history, dtype=float))
        self.noise_level_history = np.array(self.noise_level_history)

        if np.all(np.isfinite(self.min_boundary)) and np.all(np.isfinite(self.max_boundary)):
            self.finite_flag = True
            self.param_scaler = lambda p: (p-self.min_boundary)/self.diff_boundary
        else:
            self.finite_flag = False

        if self.has_trust_region:
            self.scaled_trust_min = self.param_scaler(np.maximum(self.best_params - self.trust_region, self.min_boundary))
            self.scaled_trust_max = self.param_scaler(np.minimum(self.best_params + self.trust_region, self.max_boundary))

        # Record value of update_hyperparameters used for optimization. Note that
        # self.update_hyperparameters is always set to False here above
        # regardless of its value during the optimization.
        self.used_update_hyperparameters = training_dict['update_hyperparameters']

    def run(self):
        '''
        Overides the GaussianProcessLearner multiprocessor run routine. Does nothing but makes a warning.
        '''
        self.log.warning('You should not have executed start() from the GaussianProcessVisualizer. It is not intended to be used as a independent process. Ending.')


    def return_cross_sections(self, points=100, cross_section_center=None):
        '''
        Generate 1D cross sections along each parameter axis.

        The cross sections are returned as a list of vectors of parameters
        values, costs, and uncertainties, corresponding to the 1D cross sections
        along each parameter axis. The cross sections pass through
        `cross_section_center`, which will default to the parameters that gave
        the best measured cost.

        Keyword Args:
            points (int): the number of points to sample along each cross section. Default value is 100.
            cross_section_center (array): parameter array where the centre of
                the cross section should be taken. If None, the parameters for
                the best measured cost are used. Default `None`.

        Returns:
            a tuple (cross_arrays, cost_arrays, uncer_arrays)
            cross_parameter_arrays (list): a list of arrays for each cross section, with the values of the varied parameter going from the minimum to maximum value.
            cost_arrays (list): a list of arrays for the costs evaluated along
                each cross section through `cross_section_center`.
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
        scaled_cost_arrays = []
        scaled_uncertainty_arrays = []
        for ind in range(self.num_params):
            sample_parameters = np.array([cross_section_center for _ in range(points)])
            sample_parameters[:, ind] = cross_parameter_arrays[ind]
            (costs, uncers) = self.gaussian_process.predict(sample_parameters,return_std=True)
            scaled_cost_arrays.append(costs)
            scaled_uncertainty_arrays.append(uncers)
        cross_parameter_arrays = np.array(cross_parameter_arrays)
        cost_arrays = self.cost_scaler.inverse_transform(np.array(scaled_cost_arrays))
        uncertainty_arrays = np.array(scaled_uncertainty_arrays) * self.cost_scaler.scale_
        return (cross_parameter_arrays,cost_arrays,uncertainty_arrays)

    def create_visualizations(self,
                              plot_cross_sections=True,
                              plot_hyperparameters_vs_fit=True,
                              plot_noise_level_vs_fit=True,
                              max_parameters_per_plot=None,
                              **kwargs):
        '''
        Runs the plots from a gaussian process learner file.

        Keyword Args:
            plot_cross_sections (Optional [bool]): If `True` plot predicted
                landscape cross sections, else do not. Default `True`.
            plot_hyperparameters_vs_fit (Optional [bool]): If `True` plot fitted
                hyperparameters as a function of fit number, else do not.
                Default `True`.
            plot_noise_level_vs_fit (Optional [bool]): If `True` plot the fitted
                noise level as a function of fit number, else do not. If there
                is no fitted noise level (i.e. `cost_has_noise` was set to
                `False`), then this plot will not be made regardless of the
                value passed for `plot_noise_level_vs_fit`. Default `True`.
            max_parameters_per_plot (Optional [int]): The maximum number of
                parameters to include in plots that display the values of
                parameters. If the number of parameters is larger than
                `parameters_per_plot`, then the parameters will be divided into
                groups and each group will be plotted in its own figure. If set
                to `None`, then all parameters will be included in the same plot
                regardless of how many there are. Default `None`.
        '''
        # Check for deprecated argument names.
        if 'plot_hyperparameters_vs_run' in kwargs:
            msg = ("create_visualizations() argument "
                   "plot_hyperparameters_vs_run is deprecated; "
                   "use plot_hyperparameters_vs_fit instead.")
            warnings.warn(msg)
            plot_hyperparameters_vs_fit = kwargs['plot_hyperparameters_vs_run']
        if 'plot_noise_level_vs_run' in kwargs:
            msg = ("create_visualizations() argument "
                   "plot_noise_level_vs_run is deprecated; "
                   "use plot_noise_level_vs_fit instead.")
            warnings.warn(msg)
            plot_noise_level_vs_fit = kwargs['plot_noise_level_vs_run']

        # Determine which parameters belong on plots together.
        parameter_chunks = mlu.chunk_list(
            self.param_numbers,
            max_parameters_per_plot,
        )

        # Generate the requested plots.
        if plot_cross_sections:
            for parameter_chunk in parameter_chunks:
                self.plot_cross_sections(
                    parameter_subset=parameter_chunk,
                )

        if plot_hyperparameters_vs_fit:
            for parameter_chunk in parameter_chunks:
                self.plot_hyperparameters_vs_fit(
                    parameter_subset=parameter_chunk,
                )

        if plot_noise_level_vs_fit:
            self.plot_noise_level_vs_fit()

    def _ensure_parameter_subset_valid(self, parameter_subset):
        _ensure_parameter_subset_valid(self, parameter_subset)

    def plot_cross_sections(self, parameter_subset=None):
        '''
        Produce a figure of the cross section about best cost and parameters.

        This method will produce plots showing cross sections of the predicted
        cost landscape along each parameter axis through the point in parameter
        space which gave the best measured cost. In other words, one parameter
        will be varied from its minimum allowed value to its maximum allowed
        value with the other parameters fixed at the values that they had in the
        set of parameters that gave the best measured cost. At each point the
        predicted cost will be plotted. That process will be repeated for each
        parameter in `parameter_subset`. The x axes will be scaled to the range
        0 to 1, corresponding to the minimum and maximum bound respectively for
        each parameter, so that curves from different cross sections can be
        overlaid nicely.

        The expected value of the cost will be plotted as a solid line.
        Additionally, dashed lines representing the 1-sigma uncertainty in the
        predicted cost will be plotted as well. This uncertainty includes
        contributions from the uncertainty in the model due to taking only a
        finite number of observations. Additionally, if `cost_has_noise` was set
        to `True` then the fitted noise level will be added in quadrature with
        the model uncertainty. Note that as more data points are taken the
        uncertainty in the model generally decreases, but the predicted noise
        level will typically converge to a nonzero value. That implies that the
        predicted cost uncertainty generally won't tend towards zero if
        `cost_has_noise` is set to `True`, even if many observations are made.

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
        plt.title('GP Learner: Predicted landscape' + (' with trust regions.' if self.has_trust_region else '.'))
        artists = []
        for ind in range(num_params):
            color = param_colors[ind]
            artists.append(plt.Line2D((0,1),(0,0), color=color, linestyle='-'))
        legend_labels = mlu._generate_legend_labels(
            parameter_subset,
            self.param_names,
        )
        plt.legend(artists, legend_labels ,loc=legend_loc)

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

    def plot_hyperparameters_vs_run(self, *args, **kwargs):
        '''
        Deprecated. Use `plot_hyperparameters_vs_fit()` instead.
        '''
        msg = ("plot_hyperparameters_vs_run() is deprecated; "
               "use plot_hyperparameters_vs_fit() instead.")
        warnings.warn(msg)
        self.plot_hyperparameters_vs_fit(*args, **kwargs)

    def plot_hyperparameters_vs_fit(self, parameter_subset=None):
        '''
        Produce a figure of the hyperparameters as a function of fit number.

        Only one fit is performed per generation, and multiple parameter sets
        are run each generation. Therefore the number of fits is generally less
        than the number of runs.

        The plot generated will include the data from the training archive if
        one was provided as `gp_training_filename` and the training archive was
        generated by a Gaussian process optimization.

        Args:
            parameter_subset (list-like): The indices of parameters to plot. The
                indices should be 0-based, i.e. the first parameter is
                identified with index 0. Generally the values of the indices in
                `parameter_subset` should be between 0 and the number of
                parameters minus one, inclusively. If set to `None`, then all
                parameters will be plotted. Default `None`.
        '''
        # Get default value for parameter_subset if necessary.
        if parameter_subset is None:
            parameter_subset = self.param_numbers

        # Make sure that the provided parameter_subset is acceptable.
        self._ensure_parameter_subset_valid(parameter_subset)

        # Get the indices corresponding to the number of fits. If
        # update_hyperparameters was set to False, then we'll say that there
        # were zero fits of the hyperparameters.
        if self.used_update_hyperparameters:
            log_length_scale_history = self.log_length_scale_history
            fit_numbers = np.arange(1, len(log_length_scale_history)+1)
        else:
            fit_numbers = [0]
            log_length_scale_history = np.log10(np.array([self.length_scale], dtype=float))

        # Generate set of distinct colors for plotting.
        num_params = len(parameter_subset)
        param_colors = _color_list_from_num_of_params(num_params)

        global figure_counter, fit_label, legend_loc, log_length_scale_label
        figure_counter += 1
        plt.figure(figure_counter)

        if type(self.length_scale) is float:
            # First treat the case of an isotropic kernel with one length scale
            # shared by all parameters.
            plt.plot(fit_numbers, log_length_scale_history,'o',color=param_colors[0])
            plt.title('GP Learner: Log of length scale vs fit number.')
        else:
            # Now treat case of non-isotropic kernels with one length scale per
            # parameter.
            artists=[]
            for ind in range(num_params):
                param_index = parameter_subset[ind]
                color = param_colors[ind]
                plt.plot(fit_numbers, log_length_scale_history[:,param_index],'o',color=color)
                artists.append(plt.Line2D((0,1),(0,0), color=color,marker='o',linestyle=''))

            legend_labels = mlu._generate_legend_labels(
                parameter_subset,
                self.param_names,
            )
            plt.legend(artists, legend_labels ,loc=legend_loc)
            plt.title('GP Learner: Log of length scales vs fit number.')

        plt.xlabel(fit_label)
        plt.ylabel(log_length_scale_label)

    def plot_noise_level_vs_run(self, *args, **kwargs):
        '''
        Deprecated. Use `plot_noise_level_vs_fit()` instead.
        '''
        msg = ("plot_noise_level_vs_run() is deprecated; "
               "use plot_noise_level_vs_fit() instead.")
        warnings.warn(msg)
        self.plot_noise_level_vs_fit(*args, **kwargs)

    def plot_noise_level_vs_fit(self):
        '''
        This method plots the fitted noise level as a function of fit number.

        The `noise_level` approximates the variance of values that would be
        measured if the cost were repeatedly measured for the same set of
        parameters. Note that this is the variance in those costs; not the
        standard deviation.

        This plot is only relevant to optimizations for which `cost_has_noise`
        is `True`. If `cost_has_noise` is `False` then this method does nothing
        and silently returns.

        Only one fit is performed per generation, and multiple parameter sets
        are run each generation. Therefore the number of fits is generally less
        than the number of runs.

        The plot generated will include the data from the training archive if
        one was provided as `gp_training_filename` and the training archive was
        generated by a Gaussian process optimization.
        '''
        # Make plot of noise level vs run number if cost has noise.
        if self.cost_has_noise:
            global figure_counter, fit_label, noise_label

            if self.used_update_hyperparameters:
                noise_level_history = self.noise_level_history
                fit_numbers = np.arange(1, len(noise_level_history)+1)
            else:
                # As in self.plot_hyperparameters_vs_run(), if
                # update_hyperparameters was set to False, we'll say there were
                # zero fits and plot the only value.
                fit_numbers = [0]
                noise_level_history = [self.noise_level]

            figure_counter += 1
            plt.figure(figure_counter)
            plt.plot(fit_numbers, noise_level_history,'o',color='k')
            plt.xlabel(fit_label)
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
                                                  learner_archive_filename=None,
                                                  **kwargs)

        self.log = logging.getLogger(__name__)
        training_dict = self.training_dict

        # Archive data not loaded by parent class
        self.param_names = mlu._param_names_from_file_dict(training_dict)
        #Trust region
        self.has_trust_region = bool(np.array(training_dict['has_trust_region']))
        self.trust_region = np.squeeze(np.array(training_dict['trust_region'], dtype=float))
        self.nn_training_file_dir = self.training_file_dir
        # Cost scaler
        self.cost_scaler_init_index = training_dict['cost_scaler_init_index']
        if not self.cost_scaler_init_index is None:
            self._init_cost_scaler()
        # update_hyperparameters wasn't used or saved by M-LOOP versions 3.1.1
        # and below, but effectively was set to False. Default to that value for
        # archives that don't have an entry for it.
        update_hyperparameters = training_dict.get(
            'update_hyperparameters',
            False,
        )
        self.update_hyperparameters = bool(update_hyperparameters)

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

    def create_visualizations(self,
                              plot_cross_sections=True,
                              max_parameters_per_plot=None):
        '''
        Creates plots from a neural net's learner file.

        Keyword Args:
            plot_cross_sections (Optional [bool]): If True plot predicted
                landscape cross sections, else do not. Default True.
            max_parameters_per_plot (Optional [int]): The maximum number of
                parameters to include in plots that display the values of
                parameters. If the number of parameters is larger than
                parameters_per_plot, then the parameters will be divided into
                groups and each group will be plotted in its own figure. If set
                to None, then all parameters will be included in the same plot
                regardless of how many there are. Default None.
        '''
        parameter_chunks = mlu.chunk_list(
            self.param_numbers,
            max_parameters_per_plot,
        )

        if plot_cross_sections:
            for parameter_chunk in parameter_chunks:
                self.do_cross_sections(parameter_subset=parameter_chunk)

        self.plot_surface()
        self.plot_density_surface()
        self.plot_losses()
        self.plot_regularization_history()

    def return_cross_sections(self, points=100, cross_section_center=None):
        '''
        Generate 1D cross sections along each parameter axis.

        The cross sections are returned as a list of vectors of parameters
        values and costs, corresponding to the 1D cross sections along each
        parameter axis. The cross sections pass through `cross_section_center`,
        which will default to the parameters that gave the best measured cost.
        One such pair of list of parameter vectors and corresponding predicted
        costs are returned for each net.

        Keyword Args:
            points (int): the number of points to sample along each cross section. Default value is 100.
            cross_section_center (array): parameter array where the centre of
                the cross section should be taken. If None, the parameters for
                the best measured cost are used. Default `None`.

        Returns:
            a list of tuple (cross_arrays, cost_arrays), one tuple for each net.
            cross_parameter_arrays (list): a list of arrays for each cross section, with the values of the varied parameter going from the minimum to maximum value.
            cost_arrays (list): a list of arrays for the costs evaluated along
                each cross section through `cross_section_center`.
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
            scaled_cost_arrays = []
            for ind in range(self.num_params):
                sample_parameters = np.array([cross_section_center for _ in range(points)])
                sample_parameters[:, ind] = cross_parameter_arrays[ind]
                costs = self.predict_costs_from_param_array(sample_parameters, net_index)
                scaled_cost_arrays.append(costs)
            cross_parameter_arrays = np.array(cross_parameter_arrays)
            cost_arrays = self.cost_scaler.inverse_transform(np.array(scaled_cost_arrays))
            res.append((cross_parameter_arrays, cost_arrays))
        return res

    def _ensure_parameter_subset_valid(self, parameter_subset):
        _ensure_parameter_subset_valid(self, parameter_subset)

    def do_cross_sections(self, parameter_subset=None,
                          plot_individual_cross_sections=True):
        '''
        Produce figures of the cross section about best cost and parameters.

        This method will produce plots showing cross sections of the predicted
        cost landscape along each parameter axis through the point in parameter
        space which gave the best measured cost. In other words, one parameter
        will be varied from its minimum allowed value to its maximum allowed
        value with the other parameters fixed at the values that they had in the
        set of parameters that gave the best measured cost. At each point the
        predicted cost will be plotted. That process will be repeated for each
        parameter in `parameter_subset`. The x axes will be scaled to the range
        0 to 1, corresponding to the minimum and maximum bound respectively for
        each parameter, so that curves from different cross sections can be
        overlaid nicely.

        One plot will be created which includes a solid line that shows the mean
        of the cost landscapes predicted by each net, as well as two dashed
        lines showing the minimum and maximum of the costs predicted by the
        nets for those parameter values. If `plot_individual_cross_sections` is
        set to `True` then additional plots will be created, one for each net,
        which show each net's predicted cost landscape cross sections.

        Args:
            parameter_subset (list-like): The indices of parameters to plot. The
                indices should be 0-based, i.e. the first parameter is
                identified with index 0. Generally the values of the indices in
                parameter_subset should be between 0 and the number of
                parameters minus one, inclusively. If set to `None`, then all
                parameters will be plotted. Default None.
            plot_individual_cross_sections (bool): Whether or not to create
                extra plots to show each net's predicted cross sections in its
                own figure. The plot of the average/min/max of the different
                nets' predicted cross sections in one figure will be generated
                regardless of this setting. Default `True`.
        '''
        # Get default value for parameter_subset if necessary.
        if parameter_subset is None:
            parameter_subset = self.param_numbers

        # Make sure that the provided parameter_subset is acceptable.
        self._ensure_parameter_subset_valid(parameter_subset)

        # Generate set of distinct colors for plotting.
        num_params = len(parameter_subset)
        param_colors = _color_list_from_num_of_params(num_params)

        # Generate labels for legends.
        legend_labels = mlu._generate_legend_labels(
            parameter_subset,
            self.param_names,
        )

        points = 100
        rel_params = np.linspace(0,1,points)
        all_cost_arrays = [a for _,a in self.return_cross_sections(points=points)]
        if plot_individual_cross_sections:
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
                plt.legend(artists, legend_labels ,loc=legend_loc)
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
            artists = []
            for ind in range(num_params):
                color = param_colors[ind]
                artists.append(plt.Line2D((0,1),(0,0), color=color, linestyle='-'))
            plt.legend(artists, legend_labels ,loc=legend_loc)

    def plot_surface(self):
        '''
        Produce a figure of the cost surface (only works when there are 2 parameters)
        '''
        if self.num_params != 2:
            return
        global figure_counter
        figure_counter += 1
        fig = plt.figure(figure_counter)
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')

        points = 50
        param_set = [ np.linspace(min_p, max_p, points) for (min_p,max_p) in zip(self.min_boundary,self.max_boundary)]
        params = [(x,y) for x in param_set[0] for y in param_set[1]]
        costs = self.predict_costs_from_param_array(params)
        ax.scatter([param[0] for param in params], [param[1] for param in params], costs)
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
        Produce a figure of the loss as a function of epoch for each net.

        The loss is the mean-squared fitting error of the neural net plus the
        regularization loss, which is the regularization coefficient times the
        mean L2 norm of the neural net weight arrays (without the square root).
        Note that the fitting error is calculated after normalizing the data, so
        it is in arbitrary units.

        As the neural nets are fit, the loss is recorded every 10 epochs. The
        number of epochs per fit varies, and may be different for different
        nets. The loss will generally increase at the begining of each fit as
        new data points will have been added.

        Also note that a lower loss isn't always better; a loss that is too low
        can be a sign of overfitting.
        '''
        global figure_counter
        figure_counter += 1
        fig = plt.figure(figure_counter)

        all_losses = self.get_losses()

        # Generate set of distinct colors for plotting.
        num_nets = len(all_losses)
        net_colors = _color_list_from_num_of_params(num_nets)

        artists=[]
        legend_labels=[]
        for ind, losses in enumerate(all_losses):
            color = net_colors[ind]
            epoch_numbers = 10 * np.arange(len(losses))
            plt.plot(epoch_numbers, losses, color=color, marker='o', linestyle='')
            artists.append(plt.Line2D((0,1),(0,0), color=color,marker='o',linestyle=''))
            legend_labels.append('Net {net_index}'.format(net_index=ind))

        plt.yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("Fitting Loss")
        plt.title('Loss vs Epoch')
        plt.legend(artists, legend_labels, loc=legend_loc)

    def plot_regularization_history(self):
        '''
        Produces a plot of the regularization coefficient values used.

        The neural nets use L2 regularization to smooth their predicted
        landscapes in an attempt to avoid overfitting the data. The strength of
        the regularization is set by the regularization coefficient, which is a
        hyperparameter that is tuned during the optimization if
        `update_hyperparameters` is set to `True`. Generally larger
        regularization coefficient values force the landscape to be smoother
        while smaller values allow it to vary more quickly. A value too large
        can lead to underfitting while a value too small can lead to
        overfitting. The ideal regularization coefficient value will depend on
        many factors, such as the shape of the actual cost landscape, the SNR of
        the measured costs, and even the number of measured costs.

        This method plots the initial regularization coefficient value and the
        optimal values found for the regularization coefficient when performing
        the hyperparameter tuning. One curve showing the history of values used
        for the regularization coefficient is plotted for each neural net. If
        `update_hyperparameters` was set to `False` during the optimization,
        then only the initial default value will be plotted.
        '''
        global figure_counter
        figure_counter += 1
        fig = plt.figure(figure_counter)

        regularization_histories = self.get_regularization_histories()

        # Generate set of distinct colors for plotting.
        num_nets = len(regularization_histories)
        net_colors = _color_list_from_num_of_params(num_nets)

        artists=[]
        legend_labels=[]
        for ind, regularization_history in enumerate(regularization_histories):
            color = net_colors[ind]
            hyperparameter_fit_numbers = np.arange(len(regularization_history))
            plt.plot(hyperparameter_fit_numbers, regularization_history, color=color, marker='o', linestyle='-')
            artists.append(plt.Line2D((0,1),(0,0), color=color,marker='o',linestyle=''))
            legend_labels.append('Net {net_index}'.format(net_index=ind))

        plt.yscale('log')
        plt.xlabel("Hyperparameter Fit Number")
        plt.ylabel("Regularization Coefficient")
        plt.title("Regularization Tuning History")
        plt.legend(artists, legend_labels, loc=legend_loc)
