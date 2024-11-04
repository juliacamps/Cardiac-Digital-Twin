from warnings import warn

import math
import pymp
import numpy as np
import multiprocessing
import numba

from utils import get_nan_value, insert_sorted


class EmptyPropagation:
    def __init__(self, module_name, verbose):
        if verbose:
            print('Initialising Propagation')
        self.module_name = module_name
        self.verbose = verbose

    def simulate_propagation(self, parameter_particle_modules_dict):
        raise NotImplementedError

    def simulate_propagation_population(self, parameter_population_modules_dict):
        raise NotImplementedError

    def get_from_module_dict(self, module_dict):
        return module_dict[self.module_name]


class ElectricalPropagation(EmptyPropagation):
    def __init__(self, geometry, module_name, verbose):
        super().__init__(module_name=module_name, verbose=verbose)
        self.geometry = geometry

    # Not very useful since we don't have the "parameter_particle_modules_dict" outside of the inference process
    # def get_root_node_meta_index_from_particle(self, parameter_particle_modules_dict):
    #     parameter_particle = super().get_from_module_dict(parameter_particle_modules_dict)
    #     param_dict, root_node_meta_indexes = self.__repack_particle_params(parameter_particle)
    #     return root_node_meta_indexes

    # def get_nb_candidate_root_node(self):
    #     return self.geometry.get_nb_candidate_root_node()

    def get_candidate_root_node_index(self):
        return self.geometry.get_candidate_root_node_index()

    def get_candidate_root_node_time(self, purkinje_speed):
        return self.geometry.get_candidate_root_node_time(purkinje_speed=purkinje_speed)

    def get_node_vc(self):
        return self.geometry.get_node_vc()

    def get_node_celltype(self):
        return self.geometry.get_node_celltype()

    def spatial_smoothing_of_time_field_using_adjacentcies_orthotropic_fibres(self,
                                                                              # fibre_speed, sheet_speed,
                                                                              # normal_speed, ghost_distance_to_self,
                                                                              original_field_data):
        return self.geometry.spatial_smoothing_of_time_field_using_adjacentcies_orthotropic_fibres(
            # fibre_speed=fibre_speed, sheet_speed=sheet_speed, normal_speed=normal_speed,
            # ghost_distance_to_self=ghost_distance_to_self,
            original_field_data=original_field_data)

    # def spatial_smoothing_of_time_field_using_adjacentcies(self, original_field_data, ghost_distance_to_self):
    #     return self.geometry.spatial_smoothing_of_time_field_using_adjacentcies(
    #         original_field_data=original_field_data, ghost_distance_to_self=ghost_distance_to_self)


class PrescribedLAT(ElectricalPropagation):
    def __init__(self, geometry, lat_prescribed, module_name, verbose):
        super().__init__(geometry=geometry, module_name=module_name, verbose=verbose)
        self.lat = lat_prescribed

    def simulate_propagation(self, parameter_particle_modules_dict):
        return self.lat

    def simulate_propagation_population(self, parameter_population_modules_dict):
        parameter_population = super().get_from_module_dict(parameter_population_modules_dict)
        population_size = parameter_population.shape[0]
        return np.zeros((population_size, self.lat.shape[0])) + self.lat    # TODO check that the axis of this operation work well


# Implementation of the pseudo-ECG method for tetrahedral meshes
class EikonalDjikstraTet(ElectricalPropagation):
    def __init__(self, endo_dense_speed_name, endo_sparse_speed_name, fibre_speed_name, geometry, module_name,
                 nb_speed_parameters, normal_speed_name, parameter_name_list_in_order, purkinje_speed_name,
                 sheet_speed_name, verbose):
        super().__init__(geometry=geometry, module_name=module_name, verbose=verbose)
        self.endo_dense_speed_name = endo_dense_speed_name
        self.endo_sparse_speed_name = endo_sparse_speed_name
        self.fibre_speed_name = fibre_speed_name
        self.nb_speed_parameters = nb_speed_parameters
        self.normal_speed_name = normal_speed_name
        self.parameter_name_list_in_order = parameter_name_list_in_order
        self.purkinje_speed_name = purkinje_speed_name
        self.sheet_speed_name = sheet_speed_name


    def simulate_propagation(self, parameter_particle_modules_dict):
        parameter = super().get_from_module_dict(parameter_particle_modules_dict)
        return self.simulate_lat(parameter)

    def simulate_propagation_population(self, parameter_population_modules_dict):
        parameter_population = super().get_from_module_dict(parameter_population_modules_dict)
        return self.simulate_lat_population(parameter_population)

    def __repack_particle_params(self, parameter_particle):
        # TODO use a dictionary that is built using the inputs for the adapter
        if len(parameter_particle) != len(self.parameter_name_list_in_order):
            warn('Parameter length definition not matching input!')
        speed_parameter = parameter_particle[:self.nb_speed_parameters]
        root_node_parameter = parameter_particle[self.nb_speed_parameters:]
        param_dict = {}
        for param_i in range(len(speed_parameter)):
            param_name = self.parameter_name_list_in_order[param_i]
            param_dict[param_name] = parameter_particle[param_i]
        return param_dict, root_node_parameter

    # def __repack_particle_params(self, parameter_particle):
    #     # TODO use a dictionary that is built using the inputs for the adapter
    #     # TODO enable handling multiple parameters that all refer to the same vc coordinate (e.g., sf GKs and sf Ito)
    #     if len(parameter_particle) != len(self.parameter_name_list_in_order):
    #         warn('Parameter length definition not matching input!')
    #     param_dict = {}
    #     for param_i in range(len(self.parameter_name_list_in_order)):
    #         param_name = self.parameter_name_list_in_order[param_i]
    #         param_dict[param_name] = parameter_particle[param_i]
    #     return param_dict

    # @numba.njit() # TODO Numba used to work in the myfunctions.py version of the code, but not it throws an error
    def __eikonal_part1(self, parameter_particle):
        # TODO: make this part of the code more flexible for future use with further heterogeneity, such as scars
        # TODO: make the part of getting the values of the parameters into a separate function so that it can be called after the inference if needed, especially for the root nodes
        param_dict, root_node_meta_indexes = self.__repack_particle_params(parameter_particle)
        fibre_speed = param_dict[self.fibre_speed_name]
        sheet_speed = param_dict[self.sheet_speed_name]
        normal_speed = param_dict[self.normal_speed_name]
        endo_dense_speed = param_dict[self.endo_dense_speed_name]
        endo_sparse_speed = param_dict[self.endo_sparse_speed_name]
        # scar_fibrosis_speed = ... # TODO
        purkinje_speed = param_dict[self.purkinje_speed_name]

        # TODO REIMPLEMENT THE FOLLOWING SECTION USING FUNCTIONS FROM conduction_system.py
        # Numba compatibility
        y = np.empty_like(root_node_meta_indexes)
        root_node_meta_indexes = np.round_(root_node_meta_indexes, 0, y)
        y = None

        candidate_root_node_indexes = self.get_candidate_root_node_index()
        root_node_indexes = candidate_root_node_indexes[root_node_meta_indexes == 1]
        candidate_root_node_times = self.get_candidate_root_node_time(purkinje_speed=purkinje_speed)
        root_node_times = candidate_root_node_times[root_node_meta_indexes == 1]
        # TODO REIMPLEMENT THE ABOVE SECTION USING FUNCTIONS FROM conduction_system.py

        ## ISOTROPIC REGIONS - without fibre orientation
        # Compute the cost of all endocardial edges
        navigation_costs = np.empty((self.geometry.edge.shape[0]))
        # print('2')
        for index in range(self.geometry.edge.shape[0]):
            # Cost for the propagation in the endocardium
            # if self.geometry.is_endocardial[index]:
            if self.geometry.is_dense_endocardial[index]:  # Distinguish between two PMJ densities of Purkinje network in the endocardium
                navigation_costs[index] = math.sqrt(
                    np.dot(self.geometry.edge_vec[index, :], self.geometry.edge_vec[index, :])) / endo_dense_speed
            elif self.geometry.is_sparse_endocardial[index]:
                navigation_costs[index] = math.sqrt(
                    np.dot(self.geometry.edge_vec[index, :], self.geometry.edge_vec[index, :])) / endo_sparse_speed

        ## ANISOTROPIC REGIONS - with fibre orientation
        # Current Speed Configuration
        g = np.zeros((3, 3), np.float64)  # g matrix
        # 02/12/2021 remove the healthy case because the anisotropy of the wavefront will strongly depend on the fibre orientation planes with respect to the endocardial wall
        np.fill_diagonal(g, [fibre_speed ** 2, sheet_speed ** 2, normal_speed ** 2], wrap=False)  # Needs to square each value
        # Compute EikonalDjikstraTet edge navigation costs
        for index in range(self.geometry.edge.shape[0]):
            if not self.geometry.is_dense_endocardial[index] and not self.geometry.is_sparse_endocardial[index]:
                # Cost equation for the EikonalDjikstraTet model + Fibrosis at the end
                # The fibres need to be in the shape of [node, xyz, fibre-sheet-normal] for the Eikonal to operate on them
                aux1 = np.dot(g, self.geometry.edge_fibre_sheet_normal[index, :, :].T)   # Otherwise, these operations may be doing something different.
                aux2 = np.dot(self.geometry.edge_fibre_sheet_normal[index, :, :], aux1)
                # try:
                aux3 = np.linalg.inv(aux2)
                # The above line can give a Singular matrix error if the fibres are not fully defined.
                # However, the geometry generation process already should correct for this eventuality using the function:
                # correct_and_normlise_ortho_fibre() in geometry_functions.py
                # except np.linalg.LinAlgError:
                #     print('Error of the singularity: ')
                #     print('aux2 ', aux2)
                #     print('index ', index)
                #     print('self.geometry.edge_fibre[index, :, :] ', self.geometry.edge_fibre[index, :, :])
                #     print('g ', g)
                #     quit()
                # except:
                #     print('Error of the something else: ')
                #     print('aux2 ', aux2)
                #     print('index ', index)
                #     print('self.geometry.edge_fibre[index, :, :] ', self.geometry.edge_fibre[index, :, :])
                #     print('g ', g)
                #     quit()
                aux4 = np.dot(self.geometry.edge_vec[index, :], aux3)
                aux5 = np.dot(aux4, self.geometry.edge_vec[index:index + 1, :].T)
                navigation_costs[index] = np.sqrt(aux5)[0]
        # Build adjacentcy costs for current activation parameter values
        adjacent_cost = numba.typed.List()
        for i in range(0, self.geometry.node_xyz.shape[0], 1):
            not_nan_neighbours = self.geometry.neighbours[i][self.geometry.neighbours[i] != get_nan_value()]
            adjacent_cost.append(np.concatenate((self.geometry.unfolded_edge[not_nan_neighbours][:, 1:2],
                                                 np.expand_dims(navigation_costs[
                                                                    not_nan_neighbours % self.geometry.edge.shape[0]],
                                                                -1)), axis=1))
        return adjacent_cost, root_node_indexes, root_node_times

    def simulate_lat(self, parameters):
        # Initialise variables
        predicted_lat = np.zeros((self.geometry.node_xyz.shape[0],), np.float64)
        visited_nodes = np.zeros((self.geometry.node_xyz.shape[0],), dtype=np.bool_)
        # Root nodes will be activated at time ==  self.root_node_times
        temp_times = np.zeros((self.geometry.node_xyz.shape[0],),
                              np.float64) + 1e6  # Initialise times to largely impossible values
        adjacent_cost, eikonal_root_nodes, eikonal_root_lat = self.__eikonal_part1(parameters)
        temp_times[eikonal_root_nodes] = eikonal_root_lat
        time_sorting = np.argsort(eikonal_root_lat)
        eikonal_root_nodes = eikonal_root_nodes[time_sorting]
        eikonal_root_lat = eikonal_root_lat[time_sorting]
        eikonal_root_lat = eikonal_root_lat - eikonal_root_lat[0]
        cumm_cost = eikonal_root_lat[0]
        initial_root_nodes_indexes = eikonal_root_lat <= cumm_cost
        initial_rootNodes = eikonal_root_nodes[initial_root_nodes_indexes]
        initial_rootActivationTimes = eikonal_root_lat[initial_root_nodes_indexes]
        later_rootNodes = eikonal_root_nodes[np.logical_not(initial_root_nodes_indexes)]
        later_rootActivationTimes = eikonal_root_lat[np.logical_not(initial_root_nodes_indexes)]
        ## Run the code for the root nodes
        visited_nodes[initial_rootNodes] = True  # Not simultaneous activation anymore
        predicted_lat[
            initial_rootNodes] = initial_rootActivationTimes  # Not simultaneous activation anymore
        next_nodes = (np.vstack([adjacent_cost[initial_rootNodes[rootNode_i]]
                                 + np.array([0, initial_rootActivationTimes[rootNode_i]]) for rootNode_i in
                                 range(initial_rootNodes.shape[
                                           0])])).tolist()  # Not simultaneous activation anymore
        for rootNode_i in range(later_rootNodes.shape[0]):
            next_nodes.append(np.array([later_rootNodes[rootNode_i], later_rootActivationTimes[rootNode_i]]))

        activeNode_i = eikonal_root_nodes[0]
        sortSecond = lambda x: x[1]
        next_nodes.sort(key=sortSecond, reverse=True)

        while visited_nodes[activeNode_i]:
            nextEdge = next_nodes.pop()
            activeNode_i = int(nextEdge[0])
        cumm_cost = nextEdge[1]
        if next_nodes:  # Check if the list is empty, which can happen while everything being Ok
            temp_times[(np.array(next_nodes)[:, 0]).astype(np.int32)] = np.array(next_nodes)[:,
                                                                        1]  # 04/10/2022 Why is this happening?

        ## Run the whole algorithm
        for i in range(0, self.geometry.node_xyz.shape[0] - np.sum(visited_nodes), 1):
            visited_nodes[activeNode_i] = True
            predicted_lat[
                activeNode_i] = cumm_cost  # Instead of using cumCost, I could use the actual time cost for each node
            adjacents = (adjacent_cost[activeNode_i] + np.array(
                [0, cumm_cost])).tolist()  # Instead of using cumCost, I could use the actual time cost for each node
            # If I use the actual costs, I only have to do it the first time and then it will just propagate, I will have to use decimals though, so no more uint type arrays.
            for adjacent_i in range(0, len(adjacents), 1):
                if (not visited_nodes[int(adjacents[adjacent_i][0])]
                        and (temp_times[int(adjacents[adjacent_i][0])] >
                             adjacents[adjacent_i][1])):
                    insert_sorted(next_nodes, adjacents[adjacent_i])
                    temp_times[int(adjacents[adjacent_i][0])] = adjacents[adjacent_i][1]
            while visited_nodes[activeNode_i] and len(next_nodes) > 0:
                nextEdge = next_nodes.pop()
                activeNode_i = int(nextEdge[0])
            cumm_cost = nextEdge[1]

        # Clean Memory
        adjacent_cost = None  # Clear Mem
        visited_nodes = None  # Clear Mem
        tempTimes = None  # Clear Mem
        next_nodes = None  # Clear Mem
        tempVisited = None  # Clear Mem
        navigationCosts = None  # Clear Mem
        return np.round(predicted_lat).astype(np.int32) + 1

    def simulate_lat_population(self, parameter_population):
        # Simulate local activation times in a population of particles using the EikonalDjikstraTet model.
        parameter_population_unique, unique_indexes = np.unique(parameter_population, return_inverse=True, axis=0)
        lat_list = pymp.shared.array((parameter_population_unique.shape[0], self.geometry.node_xyz.shape[0]), dtype=np.float64)
        lat_list[:, :] = get_nan_value()
        threads_num = multiprocessing.cpu_count()
        # Uncomment the following lines to turn off the parallelisation of the Eikonal computation.
        # if True:    # Turns off the parallel functionality
        #     print('Parallel loop turned off in module: ' + self.module_name)
        #     for conf_i in range(lat_list.shape[0]):    # Turns off the parallel functionality
        with pymp.Parallel(min(threads_num, lat_list.shape[0])) as p1:
            for conf_i in p1.range(lat_list.shape[0]):
                parameters = parameter_population_unique[conf_i, :]
                lat_list[conf_i, :] = np.round(self.simulate_lat(parameters)).astype(np.int32)
        return lat_list[unique_indexes]

# EOF
