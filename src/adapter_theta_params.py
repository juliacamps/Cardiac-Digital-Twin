import time

import numpy as np

from utils import get_values_from_pandas, change_resolution, translate_from_pandas_to_array

"""Glossary
Theta: parameters being inferred. This will NOT include other parameters that are not part of the inference solution.
Parameter: all parameters that need to be given to the simulator in order simulate.
"""

# def generate_parameter_dict(parameter_fixed_value_dict, parameter_name_list, theta_name_list):
#     parameter_dict = {}
#     for parameter_i in range(len(parameter_name_list)):
#         parameter_name = parameter_name_list[parameter_i]
#         has_theta = parameter_name in theta_name_list
#         if has_theta:
#             parameter_dict[parameter_name] = get_nan_value()    # Special value indicating that the parameter does not have a fixed value.
#         else:
#             parameter_dict[parameter_name] = parameter_fixed_value_dict[parameter_name]
#     return parameter_dict


def generate_index_dict(name_list_in_order):
    parameter_indexes_dict = {}
    for parameter_i in range(len(name_list_in_order)):
        name_of_current_parameter = name_list_in_order[parameter_i]
        parameter_indexes_dict[name_of_current_parameter] = parameter_i
    return parameter_indexes_dict


class AdjustTheta:
    def __init__(self):
        pass

    def adjust_resolution(self, theta_population):
        raise NotImplementedError


class RoundTheta(AdjustTheta):
    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution

    def adjust_resolution(self, theta_population):
        # print('adjust - Check this please!')
        # print('Before: ', theta_population)
        # print(len(np.unique(theta_population)))
        # adjusted_theta_population = self.resolution * np.round(theta_population/self.resolution)
        # print('After: ', adjusted_theta_population)
        # print(len(np.unique(adjusted_theta_population)))
        return change_resolution(data=theta_population, resolution=self.resolution)
        # return self.resolution * np.round(theta_population/self.resolution)

class AdapterThetaParams:
    def __init__(self, destination_module_name_list_in_order, parameter_fixed_value_dict, parameter_name_list_in_order,
                 parameter_destination_module_dict, physiological_rules_larger_than_dict,
                 theta_adjust_function_list_in_order, theta_name_list_in_order, verbose):
        """Theta and parameter must have exact same names, but can be in different order."""
        if verbose:
            print('Initialising Adapter')
        self.destination_module_name_list_in_order = destination_module_name_list_in_order
        self.parameter_name_list_in_order = parameter_name_list_in_order
        self.parameter_fixed_value_dict = parameter_fixed_value_dict
        self.parameter_destination_module_dict = parameter_destination_module_dict
        self.parameter_indexes_dict = generate_index_dict(self.parameter_name_list_in_order)
        self.physiological_rules_larger_than_dict = physiological_rules_larger_than_dict
        self.theta_adjust_function_list_in_order = theta_adjust_function_list_in_order
        self.theta_name_list_in_order = theta_name_list_in_order
        self.theta_indexes_dict = generate_index_dict(self.theta_name_list_in_order)

    def get_theta_names(self):
        return self.theta_name_list_in_order

    def distribute_parameter_to_modules(self, parameter_particle):
        destination_module_parameter_dict = {}
        for destination_module_i in range(len(self.destination_module_name_list_in_order)):
            destination_module_name = self.destination_module_name_list_in_order[destination_module_i]
            destination_module_parameter_name_list = self.parameter_destination_module_dict[destination_module_name]
            destination_module_parameter_indexes = []
            for destination_module_parameter_name_i in range(len(destination_module_parameter_name_list)):
                destination_module_parameter_name = destination_module_parameter_name_list[destination_module_parameter_name_i]
                destination_module_parameter_indexes.append(self.parameter_indexes_dict[destination_module_parameter_name])
            destination_module_parameter = parameter_particle[destination_module_parameter_indexes]
            destination_module_parameter_dict[destination_module_name] = destination_module_parameter
        return destination_module_parameter_dict

    def distribute_parameter_population_to_modules(self, parameter_population):
        destination_module_parameter_dict = {}
        for destination_module_i in range(len(self.destination_module_name_list_in_order)):
            destination_module_name = self.destination_module_name_list_in_order[destination_module_i]
            destination_module_parameter_name_list = self.parameter_destination_module_dict[destination_module_name]
            # print('destination_module_name ', destination_module_name)
            # print('self.parameter_destination_module_dict ', self.parameter_destination_module_dict)
            destination_module_parameter_indexes = []
            for destination_module_parameter_name_i in range(len(destination_module_parameter_name_list)):
                destination_module_parameter_name = destination_module_parameter_name_list[destination_module_parameter_name_i]
                destination_module_parameter_indexes.append(self.parameter_indexes_dict[destination_module_parameter_name])
            destination_module_parameter = parameter_population[:, destination_module_parameter_indexes]
            destination_module_parameter_dict[destination_module_name] = destination_module_parameter
        return destination_module_parameter_dict

    def translate_theta_to_parameter(self, theta_particle):
        # print()
        # print('translate_theta_to_parameter')
        # print('theta_particle ', theta_particle)
        # print('self.theta_name_list_in_order ', self.theta_name_list_in_order)
        aux_parameter_dict = self.parameter_fixed_value_dict
        for theta_i in range(len(self.theta_name_list_in_order)):
            name_of_current_theta = self.theta_name_list_in_order[theta_i]
            # print('name_of_current_theta ', name_of_current_theta)
            value_of_current_theta = theta_particle[theta_i]
            # print('value_of_current_theta ', value_of_current_theta)
            aux_parameter_dict[name_of_current_theta] = value_of_current_theta
        parameter_list = []
        for parameter_i in range(len(self.parameter_name_list_in_order)):
            name_of_current_parameter = self.parameter_name_list_in_order[parameter_i]
            # print('name_of_current_parameter ', name_of_current_parameter)
            value_of_current_parameter = aux_parameter_dict[name_of_current_parameter]
            # print('value_of_current_parameter ', value_of_current_parameter)
            parameter_list.append(value_of_current_parameter)
        return np.asarray(parameter_list).flatten()

    def translate_parameter_to_theta(self, parameter_particle):
        aux_theta_dict = {}
        for parameter_i in range(len(parameter_particle)):
            name_of_current_parameter = self.parameter_name_list_in_order[parameter_i]
            value_of_current_parameter = parameter_particle[name_of_current_parameter]
            if name_of_current_parameter in self.theta_name_list_in_order:
                aux_theta_dict[name_of_current_parameter] = value_of_current_parameter
        theta_list = []
        for theta_i in range(len(self.theta_name_list_in_order)):
            name_of_current_theta = self.theta_name_list_in_order[theta_i]
            value_of_current_theta = aux_theta_dict[name_of_current_theta]
            theta_list.append(value_of_current_theta)
        return np.asarray(theta_list).flatten()

    def translate_theta_population_to_parameter(self, theta_population):
        parameter_0 = self.translate_theta_to_parameter(theta_population[0, :])
        parameter_population = np.zeros((theta_population.shape[0], parameter_0.shape[0]))
        parameter_population[0, :] = parameter_0
        for particle_i in range(1, theta_population.shape[0]):
            parameter_population[particle_i, :] = self.translate_theta_to_parameter(theta_population[particle_i, :])
        return parameter_population

    def parameter_population_to_theta(self, parameter_population):
        theta_0 = self.translate_parameter_to_theta(parameter_population[0, :])
        theta_population = np.zeros((parameter_population.shape[0], theta_0.shape[0]))
        theta_population[0, :] = theta_0
        for particle_i in range(1, parameter_population.shape[0]):
            theta_population[particle_i, :] = self.translate_parameter_to_theta(parameter_population[particle_i, :])
        return theta_population

    def check_physiological_rules_theta_larger_than(self, theta_population):
        aux_theta_index_dict = {}
        for theta_i in range(len(self.theta_name_list_in_order)):
            name_of_current_theta = self.theta_name_list_in_order[theta_i]
            aux_theta_index_dict[name_of_current_theta] = theta_i
        valid_theta_population_unfolded = np.ones(theta_population.shape, dtype=bool)
        for larger_theta_i in range(len(self.theta_name_list_in_order)):
            larger_theta_name = self.theta_name_list_in_order[larger_theta_i]
            if larger_theta_name in self.physiological_rules_larger_than_dict:
                smaller_theta_name_list = self.physiological_rules_larger_than_dict[larger_theta_name]
                aux_valid_theta_population_current_unfolded = np.ones((theta_population.shape[0], len(smaller_theta_name_list)), dtype=bool)
                for smaller_theta_name_i in range(len(smaller_theta_name_list)):
                    smaller_theta_name = smaller_theta_name_list[smaller_theta_name_i]
                    aux_valid_theta_population_current_unfolded[:, smaller_theta_name_i] = theta_population[:, aux_theta_index_dict[larger_theta_name]] >= theta_population[:, aux_theta_index_dict[smaller_theta_name]]
                aux_valid_theta_population_current = np.all(aux_valid_theta_population_current_unfolded, axis=1)
                valid_theta_population_unfolded[:, larger_theta_i] = np.logical_and(valid_theta_population_unfolded[:, larger_theta_i], aux_valid_theta_population_current)
        valid_theta_population = np.all(valid_theta_population_unfolded, axis=1)
        return valid_theta_population

    def adjust_theta_values(self, theta_population):
        # return theta_population
        adjusted_theta_population = theta_population
        for theta_i in range(len(self.theta_name_list_in_order)):
            adjust_function = self.theta_adjust_function_list_in_order[theta_i]
            if adjust_function is not None:
                adjusted_theta_population[:, theta_i] = adjust_function.adjust_resolution(theta_population[:, theta_i])
        return adjusted_theta_population
        # # aux_theta_index_dict = {}
        # for theta_i in range(len(self.theta_name_list_in_order)):
        #     name_of_current_theta = self.theta_name_list_in_order[theta_i]
        #     theta_adjust_function_list_in_order
        #
        #     aux_theta_index_dict[name_of_current_theta] = theta_i
        # valid_theta_population_unfolded = np.ones(theta_population.shape, dtype=bool)
        # for larger_theta_i in range(len(self.theta_name_list_in_order)):
        #     larger_theta_name = self.theta_name_list_in_order[larger_theta_i]
        #     if larger_theta_name in self.physiological_rules_larger_than_dict:
        #         smaller_theta_name_list = self.physiological_rules_larger_than_dict[larger_theta_name]
        #         aux_valid_theta_population_current_unfolded = np.ones(
        #             (theta_population.shape[0], len(smaller_theta_name_list)), dtype=bool)
        #         for smaller_theta_name_i in range(len(smaller_theta_name_list)):
        #             smaller_theta_name = smaller_theta_name_list[smaller_theta_name_i]
        #             aux_valid_theta_population_current_unfolded[:, smaller_theta_name_i] = theta_population[:,
        #                                                                                    aux_theta_index_dict[
        #                                                                                        larger_theta_name]] >= theta_population[
        #                                                                                                               :,
        #                                                                                                               aux_theta_index_dict[
        #                                                                                                                   smaller_theta_name]]
        #         aux_valid_theta_population_current = np.all(aux_valid_theta_population_current_unfolded, axis=1)
        #         valid_theta_population_unfolded[:, larger_theta_i] = np.logical_and(
        #             valid_theta_population_unfolded[:, larger_theta_i], aux_valid_theta_population_current)
        # valid_theta_population = np.all(valid_theta_population_unfolded, axis=1)
        # return valid_theta_population

    def translate_from_pandas_to_parameter(self, pandas_parameter):
        return translate_from_pandas_to_array(name_list_in_order=self.parameter_name_list_in_order,
                                              pandas_data=pandas_parameter)
        # parameter_list = []
        # for param_name in self.parameter_name_list_in_order:
        #     parameter_list.append(get_values_from_pandas(dictionary=pandas_parameter, key=param_name))
        # return np.transpose(np.stack(parameter_list))

    def translate_from_pandas_to_theta(self, pandas_theta):
        return translate_from_pandas_to_array(name_list_in_order=self.theta_name_list_in_order, pandas_data=pandas_theta)
        # theta_list = []
        # for theta_name in self.theta_name_list_in_order:
        #     theta_list.append(get_values_from_pandas(dictionary=pandas_theta, key=theta_name))
        # return np.transpose(np.stack(theta_list))


# EOF


