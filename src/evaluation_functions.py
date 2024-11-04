import time

import numpy as np


class ParameterSimulator:
    def __init__(self, adapter, simulator, verbose):
        if verbose:
            print('Initialising Evaluator')
        self.adapter = adapter
        self.simulator = simulator
        self.verbose = verbose

    '''functionality'''
    def simulate_theta_particle(self, theta_particle):
        parameter_particle = self.adapter.translate_theta_to_parameter(theta_particle=theta_particle)
        return self.simulate_parameter_particle(parameter_particle=parameter_particle)

    def simulate_parameter_particle(self, parameter_particle):
        parameter_particle_modules_dict = self.adapter.distribute_parameter_to_modules(parameter_particle=parameter_particle)
        simulated_particle = self.simulator.simulate_particle(parameter_particle_modules_dict=parameter_particle_modules_dict)
        return simulated_particle

    def simulate_theta_population(self, theta_population):
        parameter_population = self.adapter.translate_theta_population_to_parameter(theta_population=theta_population)
        return self.simulate_parameter_population(parameter_population=parameter_population)

    def simulate_parameter_population(self, parameter_population):
        parameter_population_modules_dict = self.adapter.distribute_parameter_population_to_modules(parameter_population=parameter_population)
        simulated_population = self.simulator.simulate_population(parameter_population_modules_dict=parameter_population_modules_dict)
        return simulated_population

    def biomarker_parameter_particle(self, parameter_particle):
        parameter_particle_modules_dict = self.adapter.distribute_parameter_to_modules(parameter_particle=parameter_particle)
        biomarker_particle = self.simulator.biomarker_particle(parameter_particle_modules_dict=parameter_particle_modules_dict)
        return biomarker_particle

    def check_theta_validity(self, theta_population):
        return self.adapter.check_physiological_rules_theta_larger_than(theta_population=theta_population)

    def get_parameter_from_theta(self, theta_population):
        return self.adapter.translate_theta_population_to_parameter(theta_population=theta_population)

    def translate_from_pandas_to_theta(self, pandas_theta):
        return self.adapter.translate_from_pandas_to_theta(pandas_theta)

    def translate_from_pandas_to_parameter(self, pandas_parameter):
        return self.adapter.translate_from_pandas_to_parameter(pandas_parameter)

    def adjust_theta_values(self, theta_population):
        return self.adapter.adjust_theta_values(theta_population=theta_population)

    def get_theta_names(self):
        return self.adapter.get_theta_names()


class ParameterEvaluator(ParameterSimulator):
    def __init__(self, adapter, simulator, verbose):
        super().__init__(adapter=adapter, simulator=simulator, verbose=verbose)

    '''functionality'''
    def evaluate_theta(self, theta_particle):
        parameter_particle = self.adapter.translate_theta_to_parameter(theta_particle=theta_particle)
        return self.evaluate_parameter(parameter_particle=parameter_particle)

    def evaluate_parameter(self, parameter_particle):
        raise NotImplementedError

    def evaluate_theta_population(self, theta_population):
        parameter_population = self.adapter.translate_theta_population_to_parameter(theta_population=theta_population)
        return self.evaluate_parameter_population(parameter_population=parameter_population)

    def evaluate_parameter_population(self, parameter_population):
        raise NotImplementedError


class MetricEvaluator(ParameterEvaluator):
    def __init__(self, adapter, metric, simulator, verbose):
        super().__init__(adapter=adapter, simulator=simulator, verbose=verbose)
        self.metric = metric

    '''functionality'''
    def evaluate_parameter(self, parameter_particle):
        parameter_modules_dict = self.adapter.distribute_parameter_to_modules(parameter_particle=parameter_particle)
        return self.metric.evaluate_metric(
            self.simulator.simulate_particle(parameter_particle_modules_dict=parameter_modules_dict))

    def evaluate_parameter_population(self, parameter_population):
        parameter_population_unique, inverse_unique_indexes = np.unique(parameter_population, return_inverse=True, axis=0)
        parameter_population_modules_dict = self.adapter.distribute_parameter_population_to_modules(
            parameter_population=parameter_population_unique)
        metric_population_unique = self.metric.evaluate_metric_population(predicted_data_population=self.simulator.simulate_population(
            parameter_population_modules_dict=parameter_population_modules_dict))
        return metric_population_unique[inverse_unique_indexes]

    '''visualisation'''
    def visualise_theta_population(self, discrepancy_population, theta_population):
        parameter_population = self.adapter.translate_theta_population_to_parameter(theta_population=theta_population)
        return self.visualise_parameter_population(discrepancy_population=discrepancy_population,
                                                   parameter_population=parameter_population)

    def visualise_parameter_population(self, discrepancy_population, parameter_population):
        parameter_population_unique, unique_indexes = np.unique(parameter_population, return_index=True, axis=0)
        discrepancy_population_unique = discrepancy_population[unique_indexes]
        parameter_population_modules_dict = self.adapter.distribute_parameter_population_to_modules(
            parameter_population=parameter_population_unique)
        return self.simulator.visualise_simulation_population(
            discrepancy_population=discrepancy_population_unique,
            parameter_population_modules_dict=parameter_population_modules_dict)

    # def evaluate_parameter_population(self, parameter_population):
    #     # Sometimes the metric function may return something that is not an array, so it's not possible to use unique
    #     parameter_population_modules_dict = self.adapter.distribute_parameter_population_to_modules(
    #         parameter_population=parameter_population)
    #     return self.metric.evaluate_metric_population(predicted_data_population=self.simulator.simulate_population(
    #         parameter_population_modules_dict=parameter_population_modules_dict))


class DiscrepancyEvaluator(ParameterEvaluator):
    def __init__(self, adapter, discrepancy_metric, simulator, target_data, verbose):
        super().__init__(adapter=adapter, simulator=simulator, verbose=verbose)
        self.discrepancy_metric = discrepancy_metric
        self.target_data = target_data

    '''functionality'''
    def evaluate_parameter(self, parameter_particle):
        parameter_modules_dict = self.adapter.distribute_parameter_to_modules(parameter_particle=parameter_particle)
        return self.discrepancy_metric.evaluate_metric(
            self.simulator.simulate_particle(parameter_particle_modules_dict=parameter_modules_dict), self.target_data)

    def evaluate_parameter_population(self, parameter_population):
        parameter_population_unique, inverse_unique_indexes = np.unique(parameter_population, return_inverse=True, axis=0)
        parameter_population_modules_dict = self.adapter.distribute_parameter_population_to_modules(
            parameter_population=parameter_population_unique)
        # time_s = time.time()
        # predicted_data_population = self.simulator.simulate_population(
        #     parameter_population_modules_dict=parameter_population_modules_dict)
        # time_e = time.time()
        # time_cost = (time_e - time_s) / parameter_population_unique.shape[0]
        # print('parameter_population_unique.shape[0] ', parameter_population_unique.shape[0])
        # print('\nHEY HEY ECG TIME COST ', time_cost)
        # discrepancy_population_unique = self.discrepancy_metric.evaluate_metric_population(predicted_data_population=predicted_data_population, target_data=self.target_data)
        discrepancy_population_unique = self.discrepancy_metric.evaluate_metric_population(
            predicted_data_population=self.simulator.simulate_population(
            parameter_population_modules_dict=parameter_population_modules_dict), target_data=self.target_data)
        return discrepancy_population_unique[inverse_unique_indexes]

    '''visualisation'''
    def visualise_theta_population(self, discrepancy_population, theta_population):
        parameter_population = self.adapter.translate_theta_population_to_parameter(theta_population=theta_population)
        return self.visualise_parameter_population(discrepancy_population=discrepancy_population,
                                                   parameter_population=parameter_population)

    def visualise_parameter_population(self, discrepancy_population, parameter_population):
        parameter_population_unique, unique_indexes = np.unique(parameter_population, return_index=True, axis=0)
        discrepancy_population_unique = discrepancy_population[unique_indexes]
        parameter_population_modules_dict = self.adapter.distribute_parameter_population_to_modules(
            parameter_population=parameter_population_unique)
        return self.simulator.visualise_simulation_population(
            discrepancy_population=discrepancy_population_unique,
            parameter_population_modules_dict=parameter_population_modules_dict)

    # EOF
