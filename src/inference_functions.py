import multiprocessing
# import os
import time
from warnings import warn
import math
import numba
import numpy as np
import pandas as pd
import pymp
# from SALib.sample import saltelli
# from SALib.analyze import sobol
from SALib import ProblemSpec
from pyDOE import lhs
from scipy.stats.distributions import norm
import matplotlib.pyplot as plt

from postprocess_functions import visualise_tornado_sa
from utils import find_first_larger_than, get_nan_value


@numba.njit
def np_all_axis1(x):
    """Numba compatible version of np.all(x, axis=1)."""
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out


# @numba.njit # October 2021 TODO: precompile this function with Numba once I figure out how to update Numba in CSCS - the version of Numba from August 2021 can handle numpy.random.dirichlet https://numba.readthedocs.io/en/stable/release-notes.html
def jiggle_discrete_non_fixed_one(part_binaries, retain_ratio, nb_root_nodes_cdf, nb_root_nodes_range):
    n_parts, n_vars = part_binaries.shape
    alpha = n_parts * (1 - retain_ratio) / (retain_ratio * n_vars - 1)
    on = np.zeros((n_vars), dtype=np.bool_)
    # if np.random.uniform(0, 1) < 0.8:
    if np.random.uniform(0, 1) < retain_ratio:  # 05/12/2021
        N_on = int(np.round(np.sum(part_binaries[int(np.random.randint(0, part_binaries.shape[0])), :])))
    else:
        N_on = find_first_larger_than(item=np.random.uniform(0, 1), vec=nb_root_nodes_cdf) + nb_root_nodes_range[0]  # September 2021

    # Use only the probability of the particles with same number of nodes active
    part_binaries_N = part_binaries[np.sum(part_binaries, axis=1) == N_on, :]
    for j in range(N_on):
        open_sites = np.nonzero(np.logical_not(on))[0]
        alpha_aux = alpha + np.sum(part_binaries_N[
                                   np_all_axis1(part_binaries_N[:, on]), :][:, open_sites],
                                   axis=0)  # TODO: verify that it works well!! This is a Numba compatible implementation of np.all with an axis argument,
        # see: https://stackoverflow.com/questions/61304720/workaround-for-numpy-np-all-axis-argument-compatibility-with-numba
        w = np.random.dirichlet(alpha_aux)  # TODO: Numba new release will support this fucntion, check again in 2022
        r = open_sites[(np.random.multinomial(1, w)).astype(
            dtype=np.bool_)]  # Numpy likes that Generator is used, e.g., np.random.Generator.multinomial(1, w), but to allow for Numba I haven't done it
        on[r] = 1  # True for Numba
    return on


def jiggle_continuous_dependent(c_theta, replacement_theta_continuous):
    nb_particles_to_jiggle = replacement_theta_continuous.shape[0]
    nb_continuous_theta = replacement_theta_continuous.shape[1]
    proposed_theta_continuous = replacement_theta_continuous + np.random.multivariate_normal(np.zeros((nb_continuous_theta)), c_theta, size=nb_particles_to_jiggle)  # Sample speeds
    # TODO This should be done and defined in the adapter!!!!!!!!!!!!!!!
    proposed_theta_continuous = np.round(proposed_theta_continuous, decimals=3)  #  resolution of 1 cm/s   # TODO: this should be decided using sensitivity analysis
    return proposed_theta_continuous


@numba.njit
def comp_prob_sampled_root_nodes(part_binaries, min_nb_root_nodes, nb_root_nodes_pdf, new_binaries, retain_ratio):
    # This code can be compiled as non-python code by Numba, which makes it quite fast
    n_parts, n_vars = part_binaries.shape
    alpha = n_parts * (1 - retain_ratio) / (retain_ratio * n_vars - 1)
    p_trial_i = 0.
    N_on = int(np.sum(new_binaries))  # WARNING! Changed on the 23/06/2020, before it was N_on = np.sum(new_binaries)
    p_nRootNodes = (
                0.8 * np.sum(np.sum(part_binaries, axis=1) == N_on) / part_binaries.shape[0] + 0.2 * nb_root_nodes_pdf[
            int(N_on - 1 - min_nb_root_nodes)])
    # if N_on < 10: # September 2021 changed to always be lower than 10, because otherwise it's computationally intractable - Idea: comparmentise ventricles for computationally tractable more roots
    part_binaries_N = part_binaries[np.sum(part_binaries, axis=1) == N_on, :]
    # Permutations that Numba can understand
    A = np.nonzero(new_binaries)[0]
    k = len(A)
    numba_permutations = [[i for i in range(0)]]
    for i in range(k):
        numba_permutations = [[a] + b for a in A for b in numba_permutations if (a in b) == False]
    for combo in numba_permutations:
        on = np.zeros((n_vars), dtype=np.bool_)
        p_trial_here = p_nRootNodes
        for j in range(len(combo)):
            pb = part_binaries_N[:, on]
            aux_i = np.empty((pb.shape[0]), dtype=np.bool_)
            for part_i in range(pb.shape[0]):
                aux_i[part_i] = np.all(pb[part_i])
            aux_p = part_binaries_N[aux_i, :]
            aux = np.sum(aux_p[:, combo[j]], axis=0)
            aux1 = ((n_vars - j) * alpha + np.sum((aux_p[:, np.logical_not(on)])))
            aux2 = (alpha + aux) / aux1
            p_trial_here *= aux2
            on[combo[j]] = 1
        p_trial_i += p_trial_here
    return p_trial_i


def apply_normal_prior(prior_mean, prior_std, theta):
    return norm(loc=prior_mean, scale=prior_std).ppf(theta)  # set normal prior


def __apply_boundaries_or_priors(boundaries_theta, unscaled_sampled_theta, theta_prior_list):
        """This function works for both continuous or discrete parameters, but returns continuous values
        - The priors must be mean and std pairs of normal distributions."""
        population_size = unscaled_sampled_theta.shape[0]
        nb_theta = unscaled_sampled_theta.shape[1]
        population_theta = np.zeros((population_size, nb_theta))
        # print('unscaled ', unscaled_sampled_theta.shape)
        # print(np.amax(unscaled_sampled_theta, axis=0))
        # print(np.amin(unscaled_sampled_theta, axis=0))
        for theta_i in range(nb_theta):
            print('Applying boundaries or prior to Theta num ', theta_i)
            prior_theta = theta_prior_list[theta_i]
            if prior_theta is not None:
                print('Using prior ', prior_theta)
                # Use priors on theta - this option can yield values outside of the boundaries
                prior_mean = prior_theta[0]
                prior_std = prior_theta[1]
                population_theta[:, theta_i] = apply_normal_prior(prior_mean=prior_mean, prior_std=prior_std,
                                                                  theta=unscaled_sampled_theta[:, theta_i])  # set normal prior
            else:
                # Use boundaries
                boundary_theta = boundaries_theta[theta_i]
                print('Using boundaries ', boundary_theta)
                min_boundary = boundary_theta[0]
                max_boundary = boundary_theta[1]
                population_theta[:, theta_i] = unscaled_sampled_theta[:, theta_i] * (max_boundary - min_boundary) + min_boundary
        # print('scaled')
        # print(np.amax(population_theta, axis=0))
        # print(np.amin(population_theta, axis=0))
        return population_theta


def sample_theta_uniform(boundaries_theta, nb_theta, population_size, theta_prior_list):
    """This function works for both continuous or discrete parameters, but returns continuous values
    - The priors must be mean and std pairs of normal distributions."""
    unscaled_sampled_theta = np.random.uniform(low=0., high=1., size=(population_size, nb_theta))
    return __apply_boundaries_or_priors(boundaries_theta=boundaries_theta,
                                        unscaled_sampled_theta=unscaled_sampled_theta,
                                        theta_prior_list=theta_prior_list)


def sample_theta_lhs(boundaries_theta, nb_theta, population_size, theta_prior_list):
    """This function works for both continuous or discrete parameters, but returns continuous values
    - The priors must be mean and std pairs of normal distributions."""
    if nb_theta > 1:
        unscaled_sampled_theta = lhs(nb_theta, samples=population_size, criterion='maximin')
    else:
        unscaled_sampled_theta = lhs(nb_theta, samples=population_size, criterion='center')
        warn('LHS is using criterion=center because nb_theta < 2')
    # print('sample_theta_lhs')
    # print('nb_theta ', nb_theta)
    return __apply_boundaries_or_priors(boundaries_theta=boundaries_theta,
                                        unscaled_sampled_theta=unscaled_sampled_theta,
                                        theta_prior_list=theta_prior_list)


def sample_theta_saltelli(boundaries_theta, qoi_names, population_size, theta_name_list):
    """This function cannot be used to initialise the inference methods because it needs different inputs."""
    problem = ProblemSpec({
        'names': theta_name_list,
        'num_vars': len(theta_name_list),
        'bounds': boundaries_theta,
        'outputs': qoi_names
    })
    return problem.sample_sobol(sample_size=population_size, calc_second_order=True), problem

# def generateRootNodes(nb_theta, population_size, theta_boundaries_list, theta_prior_list):
#     rootNodes = np.zeros((n, nRootLocs))
#     for i in range(n):
#         N_on = 0  # TODO change to constraint the number of root nodes within the ranges
#         while N_on < nRootNodes_range[0] or N_on > nRootNodes_range[1]:
#             N_on = int(round(np.random.normal(loc=nRootNodes_centre, scale=nRootNodes_std)))
#         rootNodes[i, np.random.permutation(nRootLocs)[:N_on - 1]] = 1
#         rootNodes[i, i % nRootLocs] = 1  # Ensure that all root nodes are represented at least once
#     return rootNodes


def sample_nb_root_nodes_population(population_size, theta_boundaries, nb_root_nodes_cdf):
    population_nb_root_nodes = np.zeros((population_size))
    min_nb_root_nodes = theta_boundaries[0]
    for nb_root_nodes_i in range(population_size):
        population_nb_root_nodes[nb_root_nodes_i] = find_first_larger_than(np.random.uniform(0, 1),
                                                                           nb_root_nodes_cdf) + min_nb_root_nodes
    return population_nb_root_nodes


def sample_location_root_nodes_population(nb_root_node_candidates, population_root_nodes_nb_theta, population_size):
    population_root_nodes_location_theta = np.zeros((population_size, nb_root_node_candidates))
    for location_root_nodes_i in range(population_size):
        nb_root_nodes = int(population_root_nodes_nb_theta[location_root_nodes_i])
        population_root_nodes_location_theta[
            location_root_nodes_i, np.random.permutation(nb_root_node_candidates)[:nb_root_nodes - 1]] = 1
        population_root_nodes_location_theta[
            location_root_nodes_i, location_root_nodes_i % nb_root_node_candidates] = 1  # Ensure that all root nodes are represented at least once
    return population_root_nodes_location_theta


# def sample_speeds_lhs(boundaries_theta, continuous_theta_prior_list, nb_theta, population_size):
#     """This function initialise a population of theta
#     max_discrete_on: this parameter constrains how many discrete parameters can be non-zero at once.
#     This function is thought for speeds and root nodes (number and locations), so, it won't work well for any other example.
#     This function accepts priors for the continuous parameters and for the max_discrete_on parameter.
#     - These priors must be mean and std pairs of normal distributions.
#     """
#     population_speeds_theta = sample_theta_lhs(nb_theta=nb_theta, population_size=population_size,
#                                                theta_boundaries_list=boundaries_theta,
#                                                theta_prior_list=continuous_theta_prior_list
#                                                )
#     return population_speeds_theta


def sample_root_nodes(nb_root_node_boundaries, nb_candiate_root_nodes, nb_root_nodes_cdf, population_size):
    """This function initialise a population of theta
    max_discrete_on: this parameter constrains how many discrete parameters can be non-zero at once.
    This function is thought for speeds and root nodes (number and locations), so, it won't work well for any other example.
    This function accepts priors for the continuous parameters and for the max_discrete_on parameter.
    - These priors must be mean and std pairs of normal distributions.
    """
    population_root_nodes_nb_theta = sample_nb_root_nodes_population(population_size=population_size,
                                                                     theta_boundaries=nb_root_node_boundaries,
                                                                     nb_root_nodes_cdf=nb_root_nodes_cdf)
    population_root_nodes_location_theta = sample_location_root_nodes_population(
        nb_root_node_candidates=nb_candiate_root_nodes,
        population_root_nodes_nb_theta=population_root_nodes_nb_theta,
        population_size=population_size)
    return population_root_nodes_location_theta


def calculate_nb_root_nodes_pdf_cdf(nb_root_node_boundaries, nb_root_node_prior):
    # Refactoring to make the array as long as the values that it can be and normalise it to add up to 1 probability
    nb_root_nodes_pdf = np.empty((nb_root_node_boundaries[1] - nb_root_node_boundaries[0] + 1), dtype='float64')
    for N_on in range(nb_root_node_boundaries[0], nb_root_node_boundaries[1] + 1):
        nb_root_nodes_pdf[N_on - nb_root_node_boundaries[0]] = abs(
            norm.cdf(N_on - 0.5, loc=nb_root_node_prior[0], scale=nb_root_node_prior[1])
            - norm.cdf(N_on + 0.5, loc=nb_root_node_prior[0], scale=nb_root_node_prior[1]))
    nb_root_nodes_pdf = nb_root_nodes_pdf / np.sum(nb_root_nodes_pdf)
    nb_root_nodes_cdf = np.cumsum(nb_root_nodes_pdf)
    nb_root_nodes_cdf[-1] = 1.1  # I set it to be larger than 1 to account for numerical errors from the round functions
    return nb_root_nodes_pdf, nb_root_nodes_cdf  # Variable to sample nb_root_nodes parameter during inference from the prior distribution


# def __unpack_particle_params(self, particle_params):
#     param_list = []
#     for continuous_param_i in range(self.nb_continuous_params):
#         param_list.append(particle_params[continuous_param_i])
#     param_list.append(particle_params[self.nb_continuous_params:])
#     return param_list


# def pack_particle_from_params(self, speed_params, root_node_meta_indexes):
#     """speed_params = [fibre_speed, transmural_speed, normal_speed, endo_dense_speed, endo_sparse_speed]"""
#     if len(speed_params) != self.nb_continuous_params:
#         print('len(speed_params) is ' + str(len(speed_params)) + ', but it should be ' + str(
#             self.nb_continuous_params))
#         raise 'Wrong number of speed parameters!'
#     # Inverse process to __unpack_particle_params
#     particle_params = np.zeros((len(speed_params) + root_node_meta_indexes.shape[0]))
#     for continuous_param_i in range(self.nb_continuous_params):
#         particle_params[continuous_param_i] = speed_params[continuous_param_i]
#     particle_params[self.nb_continuous_params:] = root_node_meta_indexes
#     return particle_params


class SamplingMethod:
    def __init__(self, evaluator, population_size, verbose):
        if verbose:
            print('Initialising Sampling Method')
        self.evaluator = evaluator
        self.population_size = population_size
        self.verbose = verbose

    def sample(self):
        raise NotImplementedError

    def get_parameter_from_theta(self, population_theta):
        return self.evaluator.get_parameter_from_theta(population_theta)

    # def simulate_parameter_particle(self, parameter_particle):
    #     return self.evaluator.simulate_parameter_particle(parameter_particle)
    #
    # def simulate_parameter_population(self, parameter_population):
    #     return self.evaluator.simulate_parameter_population(parameter_population)


# class SensitivityAnalysis(SamplingMethod):
#     def __init__(self, boundaries_theta, evaluator, population_size, sampling_method, theta_prior_list, verbose):
#         super().__init__(evaluator=evaluator, population_size=population_size, verbose=verbose)
#         self.boundaries_theta = boundaries_theta
#         self.evaluator = evaluator
#         self.population_size = population_size
#         self.sampling_method = sampling_method
#         self.theta_prior_list = theta_prior_list
#         self.verbose = verbose
#
#     def sample(self):
#         population_theta = self.sample_theta()
#         print('Has sampled theta')
#         return population_theta
#
#     def sample_theta(self):
#         return self.sampling_method(
#             boundaries_theta=self.boundaries_theta, nb_theta=len(self.boundaries_theta),
#             population_size=self.population_size, theta_prior_list=self.theta_prior_list)
#
#     def compute_quantities_of_interest(self, population_theta):
#         return self.evaluator.evaluate_theta_population(population_theta)


class SaltelliSensitivityAnalysis(SamplingMethod):
    def __init__(self, boundaries_theta, evaluator, qoi_name_list, population_size, verbose):
        # super().__init__(boundaries_theta=boundaries_theta, evaluator=evaluator, population_size=population_size,
        #                  theta_prior_list=[None for i in range(len(boundaries_theta))], verbose=verbose)
        super().__init__(evaluator=evaluator, population_size=population_size, verbose=verbose)
        # self.boundaries_theta = boundaries_theta
        self.evaluator = evaluator
        # self.qoi_names = qoi_names
        self.population_size = population_size
        theta_name_list = self.evaluator.get_theta_names()
        # Initialise SA problem
        self.theta_name_key = 'names'
        self.qoi_key = 'outputs'
        self.bounds_key = 'bounds'
        self.nb_theta_key = 'num_vars'
        self._ini_problem(boundaries_theta=boundaries_theta, qoi_name_list=qoi_name_list, theta_name_list=theta_name_list)
        # self.problem = ProblemSpec({
        #     self.theta_name_key: theta_name_list,
        #     'num_vars': len(theta_name_list),
        #     'bounds': boundaries_theta,
        #     self.qoi_key: qoi_name_list
        # })

        # Sobol index names in SALib Python library in 2023:
        self.total_index = 0
        self.sobol_tag = 'S'
        self.sobol_total_tag = 'T'
        self.sobol_S1_tag = '1'
        self.sobol_S2_tag = '2'
        # Sobol sub values names:
        self.salib_sobol_conf_tag = '_conf'  # The SALib library uses something like 'ST_conf', 'S1_conf', etc. as the
        # conf_key = 'conf'
        # value_key = 'val'

    def _ini_problem(self, boundaries_theta, qoi_name_list, theta_name_list):
        self.problem = ProblemSpec({
            self.theta_name_key: theta_name_list,
            self.nb_theta_key: len(theta_name_list),
            self.bounds_key: boundaries_theta,
            self.qoi_key: qoi_name_list
        })

    def sample(self, max_theta_per_iter):
        '''This function makes heavy use of Pandas Dataframes, to access an index position (these are unique in the
        Dataframe), use dataframe.loc['index_name']; to access a column, use dataframe['column_name']; to modify a value
        at an index and column use dataframe.at[('index_name'), 'column_name']=new_value'''
        if self.verbose:
            print('Start sampling SA ...')
        self.problem.sample_sobol(N=self.population_size, calc_second_order=True)
        population_theta = self.problem.samples
        nb_theta = population_theta.shape[0]
        max_theta_per_iter = min(max_theta_per_iter, nb_theta)
        if self.verbose:
            print('population_theta ', population_theta.shape)
            print('max_theta_per_iter ', max_theta_per_iter)
        population_qoi_part = self.compute_quantities_of_interest(population_theta=population_theta[:max_theta_per_iter, :])
        population_qoi = np.zeros((nb_theta, population_qoi_part.shape[1]))
        population_qoi[:, :] = np.nan
        population_qoi[:max_theta_per_iter, :] = population_qoi_part
        for part_i in range(2, math.ceil(nb_theta/max_theta_per_iter)+1, 1):
            if self.verbose:
                print('part_i ', part_i, ', out of ', math.ceil(nb_theta/max_theta_per_iter)+1)
            population_qoi_part = self.compute_quantities_of_interest(
                population_theta=population_theta[(part_i-1)*max_theta_per_iter:min(nb_theta, part_i*max_theta_per_iter), :])
            population_qoi[(part_i-1)*max_theta_per_iter:min(nb_theta, part_i*max_theta_per_iter), :] = population_qoi_part
        if self.verbose:
            print('Number of nan values in the resulting QOIs: ', np.sum(np.isnan(population_qoi)))
        return population_qoi, population_theta

    def analyse_sa(self, qoi_name_list, population_qoi, population_theta, theta_name_list):
        if self.verbose:
            print('Start analysing SA ...')
        # Assign the QOIs to the SA model for the sobol indices calculation
        warn('You cannot change population_theta after you have called the function set_resuts() or it will throw an error.')
        # This allows adding or removing theta before calculating the SA analysis
        self.problem[self.theta_name_key] = theta_name_list     # This allows adding or removing theta before calculating the SA analysis
        self.problem[self.nb_theta_key] = len(theta_name_list)  # This allows adding or removing theta before calculating the SA analysis
        self.problem[self.qoi_key] = qoi_name_list
        self.problem.set_samples(population_theta)              # This allows adding or removing theta before calculating the SA analysis
        self.problem.set_results(population_qoi)
        # Calclulate total, first and second order sobol indices.
        sobol_indices = self.problem.analyze_sobol(print_to_console=False, calc_second_order=True)
        # Convert the format of the result from the Sobol indices to a list of lists of dataframes
        # Where the outer list is for the QOIs, then Sobol indices, and finally, value and confidence of the indices
        # as a dataframe.
        sobol_list_list_df = sobol_indices.to_df()     # list of lists of dataframes
        return sobol_list_list_df

    def convert_sobol_list_to_df(self, sobol_indecies_df_list):
        # QOI names:
        qoi_name_list = self.problem.get(self.qoi_key)
        print('qoi_name_list ', qoi_name_list)
        # Sobol index names:
        sobol_indecies_name_list = []   # This list must be built in ascending order or the rest of the code won't work!
        prev_sobol_i = self.total_index - 1 # Ensure the ascending order
        for sobol_i in range(self.total_index, len(sobol_indecies_df_list[0]), 1):
            assert sobol_i > prev_sobol_i # Ensure the ascending order # TODO it's possible that this is no longer a requirement
            if sobol_i == self.total_index:
                sobol_i_name = self.sobol_tag + self.sobol_total_tag
            else:   # This was consistent with the SALib library in 2023, which used namings of ST, S1, S2, etc.
                sobol_i_name = self.sobol_tag + str(sobol_i)
            sobol_indecies_name_list.append(sobol_i_name)
            prev_sobol_i = sobol_i
        # Create multiindex for final dataframe to save SA results
        multiindex_tupple_name_list = [] # This list of lists is used as combinatorial (all against all) to create the unique multiindicies in the dataframe
        # This needs to be done in a for loop, because the second order Sobol indices have different number of values
        # than the total or first order ones.
        # first QOI, Total sobol index, Total sobol index name
        theta_name_list = list(sobol_indecies_df_list[0][0][sobol_indecies_name_list[0]].index)

        for qoi_i in range(len(qoi_name_list)):
            qoi_name = qoi_name_list[qoi_i]
            # The following loop assumes that the sobol indices name list is in ascending order.
            for sobol_i in range(len(sobol_indecies_name_list)): # This list needs to be in ascending order!
                sobol_i_name = sobol_indecies_name_list[sobol_i]
                for theta_name_S1_i in range(0, len(theta_name_list), 1):
                    theta_name_S1 = theta_name_list[theta_name_S1_i]
                    if self.sobol_total_tag in sobol_i_name or self.sobol_S1_tag in sobol_i_name:  # TODO Does this invalidate the requirement of the ascending order in the list of sobol names?
                        multiindex_tupple_name_list.append((qoi_name, sobol_i_name, theta_name_S1, theta_name_S1))
                    elif self.sobol_S2_tag in sobol_i_name:   # The second order Sobol indices results have Multiindex in the result from SALib!
                        for theta_name_S2_i in range(theta_name_S1_i+1, len(theta_name_list), 1):
                            theta_name_S2 = theta_name_list[theta_name_S2_i]
                            assert theta_name_S2 != theta_name_S1 # This should not be possible given the for loop design
                            # if theta_name_S2 == theta_name_S1: # This should not be possible given the for loop design
                            #     print('theta_name_list ', theta_name_list)
                            #     print('theta_name_S1_i ', theta_name_S1_i)
                            #     print('theta_name_S2_i ', theta_name_S2_i)
                            #     print('theta_name_S1 ', theta_name_S1)
                            #     print('theta_name_S2 ', theta_name_S2)
                            #     raise Exception("This should not be possible.")
                            multiindex_tupple_name_list.append((qoi_name, sobol_i_name, theta_name_S1, theta_name_S2))
                    else:
                        print('The code for more complex Sobol indices is not implemented!')
                        raise NotImplementedError
        # iterables = [qoi_name_list, sobol_indecies_name_list, sobol_value_name_list] # TODO Delete this line
        # Define the index names inside the dataframe:
        index_qoi_name = 'qoi_name'
        index_sobol_name = 'sobol_i_name'
        index_theta_S1_name = 'theta_S1_name'
        index_theta_S2_name = 'theta_S2_name'
        # Create the dataframes multiindex object:
        sobol_indecies_df_multiindex = pd.MultiIndex.from_tuples(
            multiindex_tupple_name_list, names=[index_qoi_name, index_sobol_name, index_theta_S1_name, index_theta_S2_name])
        # print('sobol_indecies_df_multiindex ', sobol_indecies_df_multiindex)
        # print()
        # Create empty dataframe for SA results with the unique multiindexing system and the theta names as column names
        # Sobol sub values names:
        conf_column_name = 'conf'
        value_column_name = 'val'
        column_name_list_val_conf = [value_column_name, conf_column_name]
        # It's easier to create the dataframe empty but with the right size and columns if known rather than adding them
        # in a for loop, although that would also be possible and easy.
        sobol_indecies_df = pd.DataFrame(np.full((sobol_indecies_df_multiindex.size, len(column_name_list_val_conf)), np.nan),
                                         index=sobol_indecies_df_multiindex, columns=column_name_list_val_conf)     # This creates a dataframe, from now on it's possible to add columns
        # print('sobol_indecies_df3 ', sobol_indecies_df)
        # print()
        # names for the confidence for each Sobol index
        # Iterate over the list of lists of dataframes generated by SALib and populate the new dataframe for saving SA
        for qoi_i in range(len(qoi_name_list)):
            qoi_name = qoi_name_list[qoi_i]
            for sobol_i in range(len(sobol_indecies_name_list)):
                sobol_i_name = sobol_indecies_name_list[sobol_i]
                sobol_conf_name = sobol_i_name + self.salib_sobol_conf_tag
                part_sobol_df = sobol_indecies_df_list[qoi_i][sobol_i][sobol_i_name]
                part_sobol_conf_df = sobol_indecies_df_list[qoi_i][sobol_i][sobol_conf_name]
                # print('part_sobol_df1 ', part_sobol_df)
                # print()
                # print('part_sobol_df5 ', list(part_sobol_df.keys()))
                # print()
                # print('sobol_indecies_df[qoi_name, sobol_i_name, value_key] ', sobol_indecies_df.loc[(qoi_name, sobol_i_name, value_key)])
                # print()
                for theta_name_S1_i in range(0, len(theta_name_list), 1):
                    theta_name_S1 = theta_name_list[theta_name_S1_i]
                    if self.sobol_total_tag in sobol_i_name or self.sobol_S1_tag in sobol_i_name:  # TODO Does this invalidate the requirement of the ascending order in the list of sobol names?
                        sobol_indecies_df.at[(qoi_name, sobol_i_name, theta_name_S1, theta_name_S1), value_column_name] = \
                            part_sobol_df.loc[theta_name_S1]
                        sobol_indecies_df.at[(qoi_name, sobol_i_name, theta_name_S1, theta_name_S1), conf_column_name] = \
                            part_sobol_conf_df.loc[theta_name_S1]
                    elif self.sobol_S2_tag in sobol_i_name:  # The second order Sobol indices results have Multiindex in the result from SALib!
                        for theta_name_S2_i in range(theta_name_S1_i + 1, len(theta_name_list), 1): # This for loop will only explore the lower diagonal of the matrix
                            theta_name_S2 = theta_name_list[theta_name_S2_i]
                            # print('part_sobol_df ', part_sobol_df)
                            # print()
                            # print('(theta_name_S1, theta_name_S2) ', (theta_name_S1, theta_name_S2))
                            # print()
                            # print('kesy ', list(part_sobol_df.keys()))
                            # print()
                            # print('part_sobol_df[theta_name_S1, theta_name_S2] ', part_sobol_df[(theta_name_S1, theta_name_S2)])
                            # print()
                            # print('sobol_indecies_df.at[(qoi_name, sobol_conf_name, theta_name_S1, theta_name_S2), conf_column_name] ', sobol_indecies_df.at[(qoi_name, sobol_conf_name, theta_name_S1, theta_name_S2), conf_column_name])
                            assert theta_name_S2 != theta_name_S1  # This should not be possible given the for loop design
                            sobol_indecies_df.at[(qoi_name, sobol_i_name, theta_name_S1, theta_name_S2), value_column_name] = \
                                part_sobol_df[(theta_name_S1, theta_name_S2)]
                            sobol_indecies_df.at[(qoi_name, sobol_i_name, theta_name_S1, theta_name_S2), conf_column_name] = \
                                part_sobol_conf_df[(theta_name_S1, theta_name_S2)]
        # print('sobol_indecies_df4 ', sobol_indecies_df)
        return sobol_indecies_df, sobol_indecies_name_list, value_column_name, conf_column_name

    def visualise_sa(self, sa_df):
        pass

    def compute_quantities_of_interest(self, population_theta):
        warn('Have al look in here!! The order of the biomarkers may be different than expected')
        # TODO adapt this part of the code to make sure that the biomarkers are in the expected order
        population_biomarker = self.evaluator.evaluate_theta_population(population_theta)
        print('population_biomarker.shape ', population_biomarker.shape)
        return population_biomarker


class BayesianInferenceSMCABC(SamplingMethod):
    def __init__(self, evaluator, keep_fraction, max_mcmc_steps,
                 population_size, retain_ratio, verbose):
        super().__init__(evaluator=evaluator, population_size=population_size, verbose=verbose)
        self.max_mcmc_steps = max_mcmc_steps
        self.retain_ratio = retain_ratio    # TODO is this only for discrete parameters? If so, it should not be here
        self.keep_fraction = keep_fraction
        self.population_theta = None
        self.population_discrepancy = None

    def set_population_theta(self, population_theta):
        self.population_theta = population_theta

    def get_population_theta(self):
        return self.population_theta

    def get_nb_theta(self):
        raise NotImplementedError

    def get_theta_names(self):
        raise NotImplementedError

    def visualise_theta_population(self, ant_discrepancy_population, ant_theta_population, discrepancy_population, theta_population, worst_keep_index):
        theta_population_unique, unique_index, inverse_unique_index = np.unique(theta_population, return_index=True,
                                                                         return_inverse=True, axis=0)
        discrepancy_population_unique = discrepancy_population[unique_index]
        worst_keep_index_unique = inverse_unique_index[worst_keep_index]
        nUnique = theta_population_unique.shape[0]
        npart = theta_population.shape[0]
        cuttoffDiscrepancy = discrepancy_population_unique[worst_keep_index_unique]
        bestInd = np.argmin(discrepancy_population)
        self.evaluator.visualise_theta_population(discrepancy_population=discrepancy_population_unique,
                                                  theta_population=theta_population_unique)
        theta_name_list = self.get_theta_names()
        # Conduction speeds
        theta_name_lits = None
        fig, axs = plt.subplots(2, len(theta_name_list), constrained_layout=True, figsize=(7*len(theta_name_list), 10))
        fig.suptitle('nUnique % ' + str(nUnique / npart * 100), fontsize=24)
        # Prepare previous results
        ant_good_particles = ant_discrepancy_population < cuttoffDiscrepancy
        ant_good_theta = ant_theta_population[ant_good_particles, :]
        ant_bad_theta = ant_theta_population[np.logical_not(ant_good_particles), :]
        # Iterate over speeds
        for theta_i in range(len(theta_name_list)):
            # Plot new results
            axs[1, theta_i].plot(theta_population[:worst_keep_index, theta_i],
                                    discrepancy_population[:worst_keep_index], 'bo', label='kept')
            axs[1, theta_i].plot(theta_population[worst_keep_index:, theta_i],
                                    discrepancy_population[worst_keep_index:], 'go', label='new')
            axs[1, theta_i].plot(theta_population[bestInd, theta_i], discrepancy_population[bestInd],
                                    'yo', label='best')

            axs[1, theta_i].axvline(x=np.median(theta_population[:, theta_i]), c='magenta', label='theta median')

            axs[1, theta_i].set_title('New ' + theta_name_list[theta_i], fontsize=16)
            axs[1, theta_i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)

            # Plot previous results for comparison
            axs[0, theta_i].plot(ant_good_theta[:, theta_i], ant_discrepancy_population[ant_good_particles], 'bo', label='kept')
            axs[0, theta_i].plot(ant_bad_theta[:, theta_i],
                                 ant_discrepancy_population[np.logical_not(ant_good_particles)], 'ro', label='bad')
            axs[0, theta_i].set_title('Ant ' + theta_name_list[theta_i], fontsize=16)

            axs[0, theta_i].axvline(x=np.median(ant_theta_population[:, theta_i]), c='magenta', label='theta median')
            axs[0, theta_i].axhline(y=cuttoffDiscrepancy, color='blue', label='cutoff discrepancy')

        axs[0, 0].set_ylabel('discrepancy', fontsize=16)
        axs[0, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        axs[1, 0].set_ylabel('discrepancy', fontsize=16)
        axs[1, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        plt.show()

    def sample(self, desired_discrepancy, max_sampling_time, unique_stopping_ratio, visualisation_count,
               inference_history_dir=None):
        inference_start_time = time.time()
        if self.verbose:
            print('Start sampling ...')
            # print('cheat for temporal speed up in plotting ecgs')
            # TODO delete the line of code below
            # self.evaluator.visualise_theta_population(discrepancy_population=np.zeros((self.population_theta.shape[0]))+get_nan_value(),
            #                                           theta_population=self.population_theta)
        self.population_discrepancy = self.compute_discrepancy(self.population_theta)
        ant_discrepancy_population = self.population_discrepancy
        ant_theta_population = self.population_theta
        looping = True
        success = False
        worst_keep_index = int(np.round(self.population_size * self.keep_fraction))
        iteration_count = 0
        # TODO Delete or comment these lines:
        # inference_history_path = '/data/Personalisation_projects/meta_data/results/personalisation_data/DTI024/twave_sf_IKs_GKs5_GKr0.5_tjca60_CL_909/Jun_2024/inference_history/'
        # inference_history_path = '/data/Personalisation_projects/meta_data/results/personalisation_data/DTI032/twave_sf_IKs_GKs5_GKr0.5_tjca60_CL_/Jun_2024/inference_history/'
        # inference_history_path = '/data/Personalisation_projects/meta_data/results/personalisation_data/DTI004/twave_sf_IKs_GKs5_GKr0.6_tjca60/smoothing_fibre_256_96_1_count_time/inference_history/'
        # inference_history_path = '/data/Personalisation_projects/meta_data/results/personalisation_data/DTI004/twave_sf_IKs_GKs5_GKr0.6_tjca60/smoothing_fibre_256_96_077_count_time/inference_history/'
        # inference_history_path = '/p/project/icei-prace-2022-0003/wang1/Personalisation_projects/meta_data/results/personalisation_data/DTI004/twave_sf_IKs_GKs5_GKr0.6_tjca60/inference_history/'
        if inference_history_dir is not None:
            np.savetxt(inference_history_dir + 'population_theta_' + str(iteration_count) + '.csv', self.population_theta, delimiter=',')
        while looping:
            # if self.verbose:
            # TODO print the discrepancy cuttoff here!! print('cuttoff_discrepancy ', cuttoff_discrepancy, ' , desired_discrepancy ', desired_discrepancy)
            print('Inference iteration: ' + str(iteration_count) + '  |  Elapsed time '
                  + str(round((time.time() - inference_start_time)/3600.0, 1)) + ' hours, out of '
                  + str(max_sampling_time) + '   |   Time remaining: '
                  + str(max_sampling_time - round((time.time() - inference_start_time)/3600.0, 1)))
                # self.visualise_theta_population(ant_discrepancy_population=ant_discrepancy_population,
                #                                 ant_theta_population=ant_theta_population,
                #                                 discrepancy_population=self.population_discrepancy,
                #                                 theta_population=self.population_theta, worst_keep_index=worst_keep_index)
                # self.evaluator.visualise_theta_population(discrepancy_population=self.population_discrepancy,
                #                                       theta_population=self.population_theta)
            discrepancy_sorting_indexes = np.argsort(self.population_discrepancy)
            self.population_discrepancy = self.population_discrepancy[discrepancy_sorting_indexes]
            self.population_theta = self.population_theta[discrepancy_sorting_indexes, :]
            # Select the new cuttoff discrepancy
            cuttoff_discrepancy = self.population_discrepancy[worst_keep_index]
            # Fold the parameter space: copy good particles on-top of the replaced ones
            nb_particles_to_jiggle = self.population_size - worst_keep_index  # Select which particles are going to be copied into the ones that don't make the cut this round
            replacement_selection = np.random.randint(low=0, high=worst_keep_index,
                                                      size=nb_particles_to_jiggle)  # Chose the particles to be copied on-top of the replaced ones
            # Replace particles that were not good enough with copies from the kept ones
            replacement_theta = self.population_theta[replacement_selection]
            replacement_discrepancies = self.population_discrepancy[replacement_selection]
            kept_theta = self.population_theta[:worst_keep_index]  # kept particles
            kept_discrepancies = self.population_discrepancy[:worst_keep_index]  # kept particles

            # self.population_theta[worst_keep_index:] = self.population_theta[selection]   # Fold the parameter space: copy good particles on-top of the replaced ones
            # self.population_discrepancy[worst_keep_index:] = self.population_discrepancy[selection]   # Update the discrepancy of the replaced particles # After this step, the particles are not in discrepancy order any longer!

            c_theta = 2.38 ** 2 / self.get_nb_theta() * np.cov(self.population_theta[:, :self.get_nb_theta()].T)  # Hyper-parameter to the jiggle step # Optimal factor. TODO: Ask Brodie about why this value.
            # Jiggle just once to estimate how many jiggles are needed
            # aux_time_ini = time.time()
            mcmc_accepted_moves, mcmc_replacement_theta, mcmc_replacement_discrepancies = self.mvn_move(c_theta=c_theta,
                                                                                                        cuttoff_discrepancy=cuttoff_discrepancy,
                                                                                                        kept_theta=kept_theta,
                                                                                                        kept_discrepancies=kept_discrepancies,
                                                                                                        nb_mcmc_moves=1,
                                                                                                        replacement_theta=replacement_theta,
                                                                                                        replacement_discrepancies=replacement_discrepancies)
            # if self.verbose:
            #     aux_time_end = time.time()
            #     print('One move costs: ', round((aux_time_end-aux_time_ini)/60.), ' minutes')
            # Update population using results from MCMC - This was only one MCMC move to evaluate how many steps are required
            self.population_theta[worst_keep_index:] = mcmc_replacement_theta
            self.population_discrepancy[worst_keep_index:] = mcmc_replacement_discrepancies
            est_accepted_move_rate = np.mean(mcmc_accepted_moves)  # This value was being consistently overestimated!!
            # some extra operations have been added to ensure that there are no divisions by zero
            # This computes the number of MCMC jiggles are required in this iteration of the main loop
            nb_mcmc_moves = min(
                math.ceil(math.log(0.05) / math.log(1 - min(max(est_accepted_move_rate, 1e-8), 1 - 1e-8))),
                self.max_mcmc_steps)  # Makes sure that cannot be a division by zero. TODO: Ask Brodie about why this value.
            # Run the remaining MCMC steps to complete the amount just calculated - there is no need to keep the output of accepted nodes
            if self.verbose:
                print('nb_mcmc_moves ', nb_mcmc_moves)
            nb_mcmc_moves = nb_mcmc_moves - 1  # We have already done one move step to determine how many were needed.
            # aux_time_ini = time.time()
            mcmc_accepted_moves, mcmc_replacement_theta, mcmc_replacement_discrepancies = self.mvn_move(
                c_theta=c_theta, cuttoff_discrepancy=cuttoff_discrepancy, kept_theta=kept_theta,
                kept_discrepancies=kept_discrepancies, nb_mcmc_moves=nb_mcmc_moves, replacement_theta=replacement_theta,
                replacement_discrepancies=replacement_discrepancies)
            # if self.verbose:
            #     aux_time_end = time.time()
            #     print(nb_mcmc_moves, ' moves cost: ', round((aux_time_end-aux_time_ini)/60.), ' minutes')
            # Update population using results from MCMC
            self.population_theta[worst_keep_index:] = mcmc_replacement_theta
            self.population_discrepancy[worst_keep_index:] = mcmc_replacement_discrepancies
            # Check stopping criteria
            nb_unique = len(np.unique(self.population_theta, axis=0))
            unique_lim_nb = int(np.round(self.population_size * unique_stopping_ratio))
            if self.verbose:
                print('nb_unique ', nb_unique, ' , unique_lim_nb ', unique_lim_nb)
                print('cuttoff_discrepancy ', cuttoff_discrepancy, ' , desired_discrepancy ', desired_discrepancy)
                print('best discrepancy ', np.amin(self.population_discrepancy))
            if self.verbose and iteration_count % visualisation_count == 0:
                print('Visualise at iteration ', iteration_count)
                self.visualise_theta_population(ant_discrepancy_population=ant_discrepancy_population,
                                                ant_theta_population=ant_theta_population,
                                                discrepancy_population=self.population_discrepancy,
                                                theta_population=self.population_theta,
                                                worst_keep_index=worst_keep_index)
            if (nb_unique < unique_lim_nb) or (cuttoff_discrepancy < desired_discrepancy) or (
                    (time.time() - inference_start_time) / 3600 > max_sampling_time):
                looping = 0
                success = (nb_unique < unique_lim_nb) or (cuttoff_discrepancy < desired_discrepancy)
                print('(nb_unique < unique_lim_nb) ', (nb_unique < unique_lim_nb))
                print('nb_unique ', nb_unique, ' , unique_lim_nb ', unique_lim_nb)
                print('(cuttoff_discrepancy < desired_discrepancy) ', (cuttoff_discrepancy < desired_discrepancy))
                print('cuttoff_discrepancy ', cuttoff_discrepancy, ' , desired_discrepancy ', desired_discrepancy)
                print('((time.time() - inference_start_time) / 3600 > max_sampling_time) ',
                      ((time.time() - inference_start_time) / 3600 > max_sampling_time))
                print('((time.time() - inference_start_time) / 3600: ', (time.time() - inference_start_time) / 3600.0)
                # self.visualise_theta_population(ant_discrepancy_population=ant_discrepancy_population,
                #                                 ant_theta_population=ant_theta_population,
                #                                 discrepancy_population=self.population_discrepancy,
                #                                 theta_population=self.population_theta,
                #                                 worst_keep_index=worst_keep_index)


            iteration_count = iteration_count + 1
            ant_sort = np.argsort(self.population_discrepancy)
            ant_theta_population = self.population_theta[ant_sort, :]
            ant_discrepancy_population = self.population_discrepancy[ant_sort]
            # TODO Delete or comment these lines:
            if inference_history_dir is not None:
                np.savetxt(inference_history_dir + 'population_theta_' + str(iteration_count) + '.csv', self.population_theta, delimiter=',')
        return self.population_theta, success

    def compute_discrepancy(self, population_theta):
        return self.evaluator.evaluate_theta_population(population_theta)

    def mvn_move(self, c_theta, cuttoff_discrepancy, kept_theta, kept_discrepancies, nb_mcmc_moves, replacement_theta,
                 replacement_discrepancies):
        raise NotImplementedError


class ContinuousSMCABC(BayesianInferenceSMCABC):
    '''Calculating the proposal probabilities that are used to adjust the Metropolis-Hastings ratio
    (i.e. the annoying expensive things for the discrete case that are only there to make the sampling 'correct'),
    is NOT needed for the continuous parameters. As you are using random normal jumps in continuous parameter space,
    these are "symmetric" in that p(θ_new|θ) = p(θ|θ_new) and hence the terms cancel. That is why there's no probability
    calculation in the code you're currently looking at.'''
    def __init__(self, boundaries_theta, evaluator, ini_population_theta, keep_fraction, max_mcmc_steps,
                 population_size, retain_ratio, theta_prior_list, verbose):
        super().__init__(evaluator=evaluator, keep_fraction=keep_fraction,
                         max_mcmc_steps=max_mcmc_steps,
                         population_size=population_size, retain_ratio=retain_ratio, verbose=verbose)
        nb_theta = len(boundaries_theta)
        if nb_theta < 2:
            raise Exception("The implementation of SMC-ABC cannot handle less than one theta for the inference.")
        if verbose:
            print('Initialising population')
        initial_population_theta = ini_population_theta(
            boundaries_theta=boundaries_theta, nb_theta=nb_theta,
            population_size=population_size, theta_prior_list=theta_prior_list)
        if verbose:
            print('Finished initialising population')
        self.set_population_theta(initial_population_theta)
        self.boundaries_theta = boundaries_theta
        self.nb_theta = nb_theta

    def get_nb_theta(self):
        return self.nb_theta

    def get_theta_names(self):
        nb_theta = self.get_nb_theta()
        return self.evaluator.get_theta_names()[:nb_theta]

    def mvn_move(self, c_theta, cuttoff_discrepancy, kept_theta, kept_discrepancies, nb_mcmc_moves, replacement_theta,
                 replacement_discrepancies):
        nb_particles_to_jiggle = replacement_theta.shape[0]
        accepted_moves = np.full(shape=(nb_particles_to_jiggle), fill_value=0, dtype=np.bool_)
        for mcmc_move_i in range(nb_mcmc_moves):
            # Jiggle continuous parameters - after jiggling the discrete ones (i.e. conduction speeds and root nodes, respectively)
            proposed_theta = jiggle_continuous_dependent(c_theta=c_theta,
                                                         replacement_theta_continuous=replacement_theta)
            proposed_theta = self.adjust_theta_values(proposed_theta)
            # Apply physiological rules to the proposed theta
            temp_accepted_indexes_theta = self.check_theta_constraints(boundaries_theta=self.boundaries_theta, theta_population=proposed_theta)
            if np.sum(temp_accepted_indexes_theta) > 0:  # Check if at least one sample passed the constraints
                # Compile the pre-accepted proposed theta population and calculate their discrepancy
                pre_accepted_proposed_discrepancy = self.compute_discrepancy(proposed_theta[temp_accepted_indexes_theta, :])
                # Update accepted particles using the rule of having an acceptable discrepancy
                discrepancy_accepted_indexes = pre_accepted_proposed_discrepancy < cuttoff_discrepancy
                temp_accepted_indexes_theta[temp_accepted_indexes_theta] = discrepancy_accepted_indexes
                # Update replacement_theta (these are the moved values)
                replacement_theta[temp_accepted_indexes_theta] = proposed_theta[temp_accepted_indexes_theta]
                replacement_discrepancies[temp_accepted_indexes_theta] = pre_accepted_proposed_discrepancy[
                    discrepancy_accepted_indexes]
            accepted_moves = np.logical_or(temp_accepted_indexes_theta,
                                           accepted_moves)  # Compile how many particles have at least one move
        return accepted_moves, replacement_theta, replacement_discrepancies

    def check_theta_constraints(self, boundaries_theta, theta_population):
        # Regulate max and min speed physiological parameter boundaries - is expecting a list of arrays
        aux_boundaries_theta = np.asarray(boundaries_theta)
        temp_accepted_indexes_theta = np.logical_and(
            theta_population <= aux_boundaries_theta[:, 1],
            theta_population >= aux_boundaries_theta[:, 0])  # Only accept samples within parameter boundaries
        temp_accepted_indexes_theta = np.all(temp_accepted_indexes_theta, axis=1)
        # Combine boundaries validity with other rules defined in the evaluator_ecg
        temp_accepted_indexes_theta = np.logical_and(temp_accepted_indexes_theta, self.evaluator.check_theta_validity(
            theta_population=theta_population))
        return temp_accepted_indexes_theta

    def adjust_theta_values(self, theta_population):
        # Apply operations on theta in a problem specific way to avoid having the same case multiple times with different parameter values
        return self.evaluator.adjust_theta_values(theta_population)


class MixedBayesianInferenceRootNodes(ContinuousSMCABC):
    """This class is coded as semi-generic. Root nodes are treaded in such a special way that it's made explicit that
    this class is inferring root nodes and some other continuous parameters."""

    def __init__(self, boundaries_continuous_theta, continuous_theta_prior_list, evaluator,
                 ini_population_continuous_theta, keep_fraction, max_mcmc_steps,
                 max_root_node_jiggle_rate, nb_candiate_root_nodes, nb_continuous_theta, nb_root_node_boundaries,
                 nb_root_node_prior, population_size, retain_ratio, verbose):
        super().__init__(evaluator=evaluator, boundaries_theta=boundaries_continuous_theta,
                         ini_population_theta=ini_population_continuous_theta, keep_fraction=keep_fraction,
                         max_mcmc_steps=max_mcmc_steps, population_size=population_size, retain_ratio=retain_ratio,
                         theta_prior_list=continuous_theta_prior_list, verbose=verbose)
        initial_population_continuous_theta = self.get_population_theta()
        self.nb_root_nodes_pdf, self.nb_root_nodes_cdf = calculate_nb_root_nodes_pdf_cdf(
            nb_root_node_boundaries=nb_root_node_boundaries,
            nb_root_node_prior=nb_root_node_prior)
        initial_population_discrete_theta = sample_root_nodes(nb_root_node_boundaries=nb_root_node_boundaries,
                                                              nb_candiate_root_nodes=nb_candiate_root_nodes,
                                                              nb_root_nodes_cdf=self.nb_root_nodes_cdf,
                                                              population_size=population_size)
        initial_population_theta = np.concatenate(
            (initial_population_continuous_theta, initial_population_discrete_theta), axis=1)
        self.set_population_theta(initial_population_theta)
        self.nb_candiate_root_nodes = nb_candiate_root_nodes
        self.nb_continuous_theta = nb_continuous_theta
        self.boundaries_continuous_theta = boundaries_continuous_theta
        self.nb_root_node_boundaries = nb_root_node_boundaries
        self.max_root_node_jiggle_rate = max_root_node_jiggle_rate

    def get_nb_theta(self):
        return self.nb_continuous_theta

    def compute_probability_discrete(self, new_binaries, part_binaries):
        return comp_prob_sampled_root_nodes(new_binaries=new_binaries, min_nb_root_nodes=self.nb_root_node_boundaries[0],
                                         nb_root_nodes_pdf=self.nb_root_nodes_pdf, part_binaries=part_binaries,
                                         retain_ratio=self.retain_ratio)

    # def check_continuous_constraints(self, boundaries_theta, theta_population):
    #     # Regulate max and min speed physiological parameter boundaries - is expecting a list of arrays
    #     aux_boundaries_theta = np.asarray(boundaries_theta)
    #     temp_accepted_indexes_theta = np.logical_and(
    #         theta_population <= aux_boundaries_theta[:, 1],
    #         theta_population >= aux_boundaries_theta[:, 0])  # Only accept samples within parameter boundaries
    #     temp_accepted_indexes_theta = np.all(temp_accepted_indexes_theta, axis=1)
    #     # Combine boundaries validity with other rules defined in the evaluator_ecg
    #     temp_accepted_indexes_theta = np.logical_and(temp_accepted_indexes_theta, self.evaluator.check_theta_validity(theta_population=theta_population))
    #     return temp_accepted_indexes_theta

    def mvn_move(self, c_theta, cuttoff_discrepancy, kept_theta, kept_discrepancies, nb_mcmc_moves, replacement_theta, replacement_discrepancies):
        nb_particles_to_jiggle = replacement_theta.shape[0]
        accepted_moves = np.full(shape=(nb_particles_to_jiggle), fill_value=0, dtype=np.bool_)
        for mcmc_move_i in range(nb_mcmc_moves):
            # kept_theta_continuous = kept_theta[:, :self.nb_theta]
            replacement_theta_continuous = replacement_theta[:, :self.nb_continuous_theta]
            kept_theta_discrete = kept_theta[:, self.nb_continuous_theta:]
            replacement_theta_discrete = replacement_theta[:, self.nb_continuous_theta:]

            # Calcualte probablity of sampling the theta discrete from the kept_theta distribution
            replacement_theta_discrete_unique, replacement_theta_discrete_unique_indexes = np.unique(replacement_theta_discrete, return_inverse=True, axis=0)
            replacement_theta_discrete_unique_probabilities = pymp.shared.array((replacement_theta_discrete_unique.shape[0]), dtype=np.float64)
            threads_num = multiprocessing.cpu_count()
            # print('replacement_theta_discrete_unique.shape[0] ', replacement_theta_discrete_unique.shape[0])
            # print('min(threads_num, replacement_theta_discrete_unique.shape[0]) ', min(threads_num, replacement_theta_discrete_unique.shape[0]))
            # Uncomment the following lines to turn off the parallelisation of the Eikonal computation.
            # if True:    # Turns off the parallel functionality
            #     print('Parallel loop turned off in module: ' + 'mvn_move')
            #     for replacement_theta_discrete_unique_i in range(replacement_theta_discrete_unique.shape[0]):    # Turns off the parallel functionality
            with pymp.Parallel(min(threads_num, replacement_theta_discrete_unique.shape[0])) as p1:
                for replacement_theta_discrete_unique_i in p1.range(replacement_theta_discrete_unique.shape[0]):
                    replacement_theta_discrete_unique_probabilities[replacement_theta_discrete_unique_i] = self.compute_probability_discrete(
                        new_binaries=replacement_theta_discrete_unique[replacement_theta_discrete_unique_i, :],
                        part_binaries=kept_theta_discrete)  # Calculate the probability of sampling the replacement discrete theta
            replacement_theta_discrete_probabilities = replacement_theta_discrete_unique_probabilities[replacement_theta_discrete_unique_indexes]

            # Jiggle discrete parameters
            proposed_theta_discrete = np.empty(replacement_theta_discrete.shape, dtype=np.float64)
            for replacement_theta_discrete_i in range(replacement_theta_discrete.shape[0]):
                proposed_theta_discrete[replacement_theta_discrete_i] = jiggle_discrete_non_fixed_one(
                    part_binaries=kept_theta_discrete, retain_ratio=self.retain_ratio,
                    nb_root_nodes_cdf=self.nb_root_nodes_cdf, nb_root_nodes_range=self.nb_root_node_boundaries)

            proposed_theta_discrete_unique, proposed_theta_discrete_unique_indexes = np.unique(proposed_theta_discrete, return_inverse=True,
                                                                                               axis=0)  # Evaluate only unique ones and then copy back to all
            proposed_theta_discrete_unique_probabilities = pymp.shared.array((proposed_theta_discrete_unique.shape[0]), dtype=np.float64)
            # Uncomment the following lines to turn off the parallelisation of the Eikonal computation.
            # if True:  # Turns off the parallel functionality
            #     print('Parallel loop turned off in module: ' + 'mvn_move')
            #     for replacement_theta_discrete_unique_i in range(
            #         replacement_theta_discrete_unique.shape[0]):  # Turns off the parallel functionality
            with pymp.Parallel(min(threads_num, proposed_theta_discrete_unique.shape[0])) as p1:
                for replacement_theta_discrete_unique_i in p1.range(proposed_theta_discrete_unique.shape[0]):
                    proposed_theta_discrete_unique_probabilities[replacement_theta_discrete_unique_i] = \
                        self.compute_probability_discrete(new_binaries=proposed_theta_discrete_unique[replacement_theta_discrete_unique_i, :], part_binaries=kept_theta_discrete)
            proposed_theta_discrete_probabilities = proposed_theta_discrete_unique_probabilities[proposed_theta_discrete_unique_indexes]  # Copy back to all from unique ones
            # Determine which of the proposed discrete particles are rare enough to be taken instead of the original reference
            non_accepted_proposed_theta_discrete_indexes = np.random.rand(nb_particles_to_jiggle) > replacement_theta_discrete_probabilities / proposed_theta_discrete_probabilities
            proposed_theta_discrete[non_accepted_proposed_theta_discrete_indexes, :] = replacement_theta_discrete[non_accepted_proposed_theta_discrete_indexes, :]
            proposed_theta = np.concatenate((replacement_theta_continuous, proposed_theta_discrete), axis=1)  # Compile new root nodes with copied speeds
            proposed_discrepancy = self.compute_discrepancy(proposed_theta)  # Discrepancy of the new root nodes
            accepted_moves = np.logical_or(proposed_discrepancy < cuttoff_discrepancy, accepted_moves)
            # Update replacement particles using the accepted moves for the discrepancy theta
            replacement_theta[proposed_discrepancy < cuttoff_discrepancy] = proposed_theta[proposed_discrepancy < cuttoff_discrepancy]  # Keep the particles that have lower discrepancy than the cuttoff
            replacement_discrepancies[proposed_discrepancy < cuttoff_discrepancy] = proposed_discrepancy[proposed_discrepancy < cuttoff_discrepancy]

            # Jiggle continuous parameters - after jiggling the discrete ones (i.e. conduction speeds and root nodes, respectively)
            proposed_theta_continuous = jiggle_continuous_dependent(c_theta=c_theta, replacement_theta_continuous=replacement_theta_continuous)
            proposed_theta = np.concatenate((proposed_theta_continuous, replacement_theta[:, self.nb_continuous_theta:]), axis=1)
            proposed_theta = self.adjust_theta_values(proposed_theta)
            # Apply physiological rules to the proposed theta
            temp_accepted_indexes_theta = self.check_theta_constraints(
                boundaries_theta=self.boundaries_continuous_theta, theta_population=proposed_theta_continuous)
            if np.sum(temp_accepted_indexes_theta) > 0:     # Check if at least one sample passed the constraints
                # Compile the pre-accepted proposed theta population and calculate their discrepancy
                pre_accepted_proposed_discrepancy = self.compute_discrepancy(proposed_theta[temp_accepted_indexes_theta, :])
                # Update accepted particles using the rule of having an acceptable discrepancy
                discrepancy_accepted_indexes = pre_accepted_proposed_discrepancy < cuttoff_discrepancy
                temp_accepted_indexes_theta[temp_accepted_indexes_theta] = discrepancy_accepted_indexes
                # Update replacement_theta (these are the moved values)
                replacement_theta[temp_accepted_indexes_theta] = proposed_theta[temp_accepted_indexes_theta]
                replacement_discrepancies[temp_accepted_indexes_theta] = pre_accepted_proposed_discrepancy[discrepancy_accepted_indexes]
            accepted_moves = np.logical_or(temp_accepted_indexes_theta,
                                           accepted_moves)  # Compile how many particles have at least one move
        return accepted_moves, replacement_theta, replacement_discrepancies

    
# class BruteForceInference(InferenceMethod):
#     def __init__(self, boundaries_continuous_theta, continuous_theta_prior_list, evaluator,
#                  ini_population_continuous_theta,
#                  nb_continuous_theta, population_size, verbose):
#         super().__init__(evaluator=evaluator, population_size=population_size, verbose=verbose)
#         self.boundaries_continuous_theta = boundaries_continuous_theta
#         self.continuous_theta_prior_list = continuous_theta_prior_list
#         self.ini_population_continuous_theta = ini_population_continuous_theta
#         self.nb_continuous_theta = nb_continuous_theta
#
#     def sample(self):
#         return self.ini_population_continuous_theta(
#             boundaries_theta=self.boundaries_continuous_theta, nb_theta=self.nb_continuous_theta,
#             population_size=self.population_size, theta_prior_list=self.continuous_theta_prior_list)


# def generate_apd_gradient_populations(self, ranges,
#     ranges_names=['apd_min', 'apd_max', 'gab', 'gtm', 'gtv'],
#     method='lhs', verbose=False):
#     if method == 'lhs':
#         sample = lhs(ranges.shape[0], samples=self.sample_size, criterion='corr')
#         for i in range(ranges.shape[0]):
#             sample[:,i] = sample[:,i] * (ranges[i,1] - ranges[i,0]) + ranges[i,0]
#     sample_theta_saltelli(boundaries_theta, nb_theta, population_size, theta_prior_list)
#     return sample

# END