import multiprocessing
import time
from warnings import warn
import math
import numpy as np
import pandas as pd
import pymp

from postprocess_functions import generate_activation_map
# from geometry_functions import temporal_smoothing_of_time_field
from utils import get_nan_value, initialise_pandas_dataframe, get_row_id_name


def calculate_node_vm_from_any_ap(ap, lat, simulation_time, upstroke_index):  # TODO: this could be faster when assuming that it's a step function
    # TODO: consider replacing this function with generate_node_vm() since it does not save that much computation time
    # Assign VM values using a reference Action Potential (AP) and using the local activation times (LAT).
    # The AP must be at 1000 Hz
    nb_nodes = lat.shape[0]
    resting_vm = ap[0]  # first value of AP should be baseline
    vm = np.full((nb_nodes, simulation_time), resting_vm, dtype=np.float64)
    # Calculate voltage per timestep
    # TODO implement the pre-upstroke contribution part, this is already implemented in the function generate_node_vm()
    nb_timesteps = min(simulation_time, int(math.ceil(np.amax(lat))))  # 1000 Hz is one evaluation every 1 ms
    ap_duration = ap.shape[0] - upstroke_index
    for timestep_i in range(0, nb_timesteps, 1):  # 1000 Hz is one evaluation every 1 ms
        time_to_fill = min(simulation_time - timestep_i, ap_duration)
        vm[lat == timestep_i, timestep_i:time_to_fill + timestep_i] = ap[upstroke_index:time_to_fill]  # Reference AP case
    return vm


def calculate_node_vm_from_upstroke_step_function(lat, resting_vm, upstroke_vm):  # TODO: this could be faster when assuming that it's a step function
    # Optimised function for step function celltypes with only upstroke
    # lat must start with a value larger than zero, and the upstroke time will be set to zero in the action potentials
    nb_nodes = lat.shape[0]
    nb_timesteps = int(math.ceil(np.amax(lat))) + 1
    simulation_time = nb_timesteps
    vm = np.full((nb_nodes, simulation_time), resting_vm, dtype=np.float64)  # + innactivated_Vm
    for timestep_i in range(1, nb_timesteps, 1):  # 1000 Hz is one evaluation every 1 ms
        vm[lat == timestep_i, timestep_i:] = upstroke_vm
    return vm


# def dummy_approximate_lat_from_vm(vm):
#     return np.argmax(vm, axis=1)


class Electrophysiology:
    def __init__(self, cellular_model, module_name, propagation_model, verbose):
        self.cellular_model = cellular_model
        self.module_name = module_name
        self.propagation_model = propagation_model
        self.verbose = verbose
        if verbose:
            print('Initialising EP')

    def simulate_electrophysiology(self, parameter_particle_modules_dict):
        raise NotImplementedError

    def simulate_electrophysiology_population(self, parameter_population_modules_dict):
        raise NotImplementedError

    # def get_nb_candidate_root_node(self):
    #     return self.propagation_model.get_nb_candidate_root_node()

    def get_from_module_dict(self, module_dict):
        return module_dict[self.module_name]


class PrescribedVM(Electrophysiology):
    def __init__(self, cellular_model, module_name, propagation_model, verbose, vm_prescribed):
        super().__init__(cellular_model=cellular_model, module_name=module_name, propagation_model=propagation_model,
                         verbose=verbose)
        # TODO make the percentage for claculating the LATs into a global varibale to be consistent
        self.lat = generate_activation_map(vm=vm_prescribed, percentage=70)  # This does not get used when simulating the ECGs
        self.vm = vm_prescribed
        # print('lat ', self.lat.shape)
        # print('vm ', self.vm.shape)

    def simulate_electrophysiology(self, parameter_particle_modules_dict):
        return self.lat, self.vm

    def prescribe_vm(self, vm_prescribed):
        self.lat = generate_activation_map(vm=vm_prescribed, percentage=70)
        self.vm = vm_prescribed


class ElectrophysiologyUpstrokeStepFunction(Electrophysiology):
    def __init__(self, cellular_model, module_name, propagation_model, verbose):
        super().__init__(cellular_model=cellular_model, module_name=module_name, propagation_model=propagation_model,
                         verbose=verbose)

    def simulate_electrophysiology(self, parameter_particle_modules_dict):
        lat_simulation = self.propagation_model.simulate_propagation(
            parameter_particle_modules_dict=parameter_particle_modules_dict)
        resting_vm = self.cellular_model.get_resting_vm()
        upstroke_vm = self.cellular_model.get_upstroke_vm()
        vm_simulation = calculate_node_vm_from_upstroke_step_function(lat=lat_simulation, resting_vm=resting_vm,
                                                                      upstroke_vm=upstroke_vm)
        return lat_simulation, vm_simulation

    def simulate_electrophysiology_population(self, parameter_population_modules_dict):
        lat_population = self.propagation_model.simulate_propagation_population(parameter_population_modules_dict=parameter_population_modules_dict)
        lat_population_unique, unique_inverse_indexes = np.unique(lat_population, return_inverse=True, axis=0)
        simulation_time = int(math.ceil(np.amax(lat_population_unique) + 1))
        vm_population_unique = pymp.shared.array((lat_population_unique.shape[0], lat_population_unique.shape[1],
                                                  simulation_time), dtype=np.float64)
        vm_population_unique[:, :, :] = get_nan_value() # TODO be careful that there are no NAN values in the ECG calculation
        resting_vm = self.cellular_model.get_resting_vm()
        upstroke_vm = self.cellular_model.get_upstroke_vm()
        threads_num = multiprocessing.cpu_count()
        # Uncomment the following lines to turn off the parallelisation.
        # if True:
        #     print('Parallel loop turned off in module: ' + self.module_name)
        #     for conf_i in range(vm_population_unique.shape[0]):
        with pymp.Parallel(min(threads_num, vm_population_unique.shape[0])) as p1:
            for conf_i in p1.range(vm_population_unique.shape[0]):
                vm = np.zeros((lat_population_unique.shape[1], simulation_time))
                vm_simulated = calculate_node_vm_from_upstroke_step_function(lat=lat_population_unique[conf_i, :],
                                                                             resting_vm=resting_vm,
                                                                             upstroke_vm=upstroke_vm)
                vm[:, :vm_simulated.shape[1]] = vm_simulated
                vm[:, vm_simulated.shape[1]:] = vm_simulated[:, -1, np.newaxis]
                vm_population_unique[conf_i, :, :] = vm
        return lat_population, vm_population_unique[unique_inverse_indexes, :, :]


class ElectrophysiologySameAP(Electrophysiology):
    def __init__(self, action_potential_duration, celltype_id, cellular_model, module_name, propagation_model, verbose):
        super().__init__(cellular_model=cellular_model, module_name=module_name, propagation_model=propagation_model,
                         verbose=verbose)
        action_potential_simulation, upstroke_index = self.cellular_model.generate_action_potential(
            action_potential_duration=action_potential_duration, celltype_id=celltype_id)
        self.action_potential_simulation = action_potential_simulation[upstroke_index:]
        self.upstroke_index = upstroke_index

    def simulate_electrophysiology(self, parameter_particle_modules_dict):
        lat_simulation = self.propagation_model.simulate_propagation(parameter_particle_modules_dict=parameter_particle_modules_dict)
        max_lat = int(math.ceil(np.amax(lat_simulation) + 1))
        simulation_time = max_lat + self.action_potential_simulation.shape[0]
        vm_simulation = calculate_node_vm_from_any_ap(ap=self.action_potential_simulation, lat=lat_simulation,
                                                      simulation_time=simulation_time, upstroke_index=self.upstroke_index)
        return lat_simulation, vm_simulation

    def simulate_electrophysiology_population(self, parameter_population_modules_dict):
        lat_population = self.propagation_model.simulate_propagation_population(parameter_population_modules_dict=parameter_population_modules_dict)
        lat_population_unique, unique_inverse_indexes = np.unique(lat_population, return_inverse=True, axis=0)
        max_lat = int(math.ceil(np.amax(lat_population_unique) + 1))
        simulation_time = max_lat + self.action_potential_simulation.shape[0]
        vm_population_unique = pymp.shared.array((lat_population_unique.shape[0], lat_population_unique.shape[1],
                                                  simulation_time), dtype=np.float64)
        # vm_population_unique = np.zeros((lat_population_unique.shape[0], lat_population_unique.shape[1], simulation_time), dtype=np.float64)
        vm_population_unique[:, :, :] = get_nan_value() # TODO be careful that there are no NAN values in the ECG calculation
        threads_num = multiprocessing.cpu_count()
        # Uncomment the following lines to turn off the parallelisation.
        # if True:
        #     print('Parallel loop turned off in module: ' + self.module_name)
        #     for conf_i in range(vm_population_unique.shape[0]):
        with pymp.Parallel(min(threads_num, vm_population_unique.shape[0])) as p1:
            for conf_i in p1.range(vm_population_unique.shape[0]):
                vm = np.zeros((lat_population_unique.shape[1], simulation_time))
                vm_simulated = calculate_node_vm_from_any_ap(ap=self.action_potential_simulation,
                                                             lat=lat_population_unique[conf_i, :],
                                                             simulation_time=simulation_time,
                                                             upstroke_index=self.upstroke_index)
                vm[:, :vm_simulated.shape[1]] = vm_simulated
                vm[:, vm_simulated.shape[1]:] = vm_simulated[:, -1, np.newaxis]
                vm_population_unique[conf_i, :, :] = vm
        return lat_population, vm_population_unique[unique_inverse_indexes, :, :]


class ElectrophysiologyAPDmap(Electrophysiology):
    def __init__(self, apd_max_name, apd_min_name, cellular_model, fibre_speed_name,
                 start_smoothing_time_index,
                 end_smoothing_time_index,
                 module_name, normal_speed_name, parameter_name_list_in_order, propagation_model, sheet_speed_name,
                 # smoothing_count,
                 smoothing_dt,
                 smoothing_ghost_distance_to_self, #smoothing_past_present_window,
                 verbose):
        super().__init__(cellular_model=cellular_model, module_name=module_name, propagation_model=propagation_model, verbose=verbose)
        self.apd_max_name = apd_max_name
        self.apd_min_name = apd_min_name
        self.parameter_name_list_in_order = parameter_name_list_in_order
        print('parameter_name_list_in_order ', parameter_name_list_in_order)
        print('parameter_name_list_in_order ', len(parameter_name_list_in_order))
        # self.smoothing_count = smoothing_count
        self.smoothing_dt = smoothing_dt
        self.smoothing_ghost_distance_to_self = smoothing_ghost_distance_to_self
        # self.smoothing_past_present_window = smoothing_past_present_window
        self.start_smoothing_time_index = start_smoothing_time_index
        self.end_smoothing_time_index = end_smoothing_time_index
        # self.fibre_speed_name = fibre_speed_name
        # self.sheet_speed_name = sheet_speed_name
        # self.normal_speed_name = normal_speed_name

    def simulate_electrophysiology(self, parameter_particle_modules_dict):
        print('simulate_electrophysiology in electrophysiology_functions.py')
        print('parameter_particle_modules_dict ', parameter_particle_modules_dict)
        node_celltype = self.propagation_model.get_node_celltype()
        lat_simulation = self.propagation_model.simulate_propagation(parameter_particle_modules_dict=parameter_particle_modules_dict)
        parameter = self.get_from_module_dict(parameter_particle_modules_dict)
        unsmoothed_node_vm = self.generate_node_vm(parameter_particle=parameter, node_lat=lat_simulation,
                                                   node_celltype=node_celltype)
        # TODO Fix this hack!!!
        # print('TODO Fix this hack!!! appears in more than one place!!')
        # fibre_speed = get_nan_value() #6.500000000000000222e-02  # param_dict[self.fibre_speed_name]
        # sheet_speed = get_nan_value() #2.900000000000000147e-02  # param_dict[self.fibre_speed_name]
        # normal_speed = get_nan_value() #4.800000000000000100e-02  # param_dict[self.fibre_speed_name]
        vm_simulation = self.apply_spatiotemporal_smoothing(
            # fibre_speed=fibre_speed,
            nodefield=unsmoothed_node_vm#,
                                                            # normal_speed=normal_speed, sheet_speed=sheet_speed
        )
        return lat_simulation, vm_simulation

    def biomarker_electrophysiology(self, parameter_particle_modules_dict):
        node_celltype = self.propagation_model.get_node_celltype()
        parameter = self.get_from_module_dict(parameter_particle_modules_dict)
        node_biomarker = self.generate_node_biomarker(parameter_particle=parameter, node_celltype=node_celltype)
        return node_biomarker

    def simulate_electrophysiology_population(self, parameter_population_modules_dict):
        node_celltype = self.propagation_model.get_node_celltype()
        lat_population = self.propagation_model.simulate_propagation_population(
            parameter_population_modules_dict=parameter_population_modules_dict)
        parameter_population = self.get_from_module_dict(parameter_population_modules_dict)
        simulation_configuration_population = np.concatenate((lat_population, parameter_population), axis=1)
        simulation_configuration_population_unique, unique_indexes, unique_inverse_indexes = np.unique(
            simulation_configuration_population, return_index=True, return_inverse=True, axis=0)
        simulation_configuration_population = None  # Clear memory
        lat_population_unique = lat_population[unique_indexes, :]
        parameter_population_unique = parameter_population[unique_indexes, :]
        simulation_configuration_population_unique = None  # Clear memory
        max_simulation_time = int(math.ceil(np.amax(lat_population_unique))) + self.cellular_model.get_max_action_potential_len()
        vm_population_unique = pymp.shared.array((parameter_population_unique.shape[0],
                                                  lat_population_unique.shape[1], max_simulation_time), dtype=np.float64)
        # simulation_time_list = pymp.shared.array((simulation_configuration_population_unique.shape[0]), dtype=np.int32)
        # unsmoothed_vm_map_population = np.zeros((simulation_configuration_population_unique.shape[0], lat_population_unique.shape[1], max_simulation_time), dtype=np.float64)
        vm_population_unique[:, :, :] = get_nan_value()  # TODO be careful that there are no NAN values in the ECG calculation
        threads_num = multiprocessing.cpu_count()
        # Uncomment the following lines to turn off the parallelisation.
        # print('# TODO Fix this hack!!!')
        # if True:
        #     print('Parallel loop turned off in module: ' + self.module_name)
        #     for conf_i in range(vm_population_unique.shape[0]):
        with pymp.Parallel(min(threads_num, parameter_population_unique.shape[0])) as p1:
            for conf_i in p1.range(parameter_population_unique.shape[0]):
                parameter_particle = parameter_population_unique[conf_i]
                unsmoothed_node_vm = self.generate_node_vm(parameter_particle=parameter_particle,
                                                           node_lat=lat_population_unique[conf_i],
                                                           node_celltype=node_celltype)
                # param_dict = self.__repack_particle_params(parameter_particle=parameter_particle)
                # simulation_time_list[conf_i] = unsmoothed_node_vm.shape[1]
                # TODO Fix this hack!!!
                # TODO the speeds should be read from the param_dict somehow, perhaps the speeds are also needed for the
                # inference of the T wave characteristics and this should be mapped by the Adapter class
                # fibre_speed = get_nan_value() #6.500000000000000222e-02 #param_dict[self.fibre_speed_name]
                # sheet_speed = get_nan_value() #2.900000000000000147e-02 #param_dict[self.fibre_speed_name]
                # normal_speed = get_nan_value() #4.800000000000000100e-02 #param_dict[self.fibre_speed_name]
                vm_population_unique[conf_i, :, :unsmoothed_node_vm.shape[1]] = self.apply_spatiotemporal_smoothing(
                    # fibre_speed=fibre_speed,
                    nodefield=unsmoothed_node_vm #,
                    # normal_speed=normal_speed,
                    # sheet_speed=sheet_speed
                )
                unsmoothed_node_vm = None  # Clear memory
        # max_simulation_time = int(np.amax(simulation_time_list))
        # vm_population_unique = vm_population_unique[:, :, :max_simulation_time]
        return lat_population, vm_population_unique[unique_inverse_indexes, :, :]

    def __repack_particle_params(self, parameter_particle):
        # TODO use a dictionary that is built using the inputs for the adapter
        # TODO enable handling multiple parameters that all refer to the same vc coordinate (e.g., sf GKs and sf Ito)
        if len(parameter_particle) != len(self.parameter_name_list_in_order):
            warn('Parameter length definition not matching input!')
        param_dict = {}
        for param_i in range(len(self.parameter_name_list_in_order)):
            param_name = self.parameter_name_list_in_order[param_i]
            param_dict[param_name] = parameter_particle[param_i]
        return param_dict

    def generate_node_apd(self, parameter_particle):
        # TODO: Caution!! The gradient parameters are given the same names as the ventricular coordinates to allow for cross-indexing both dictionaries with the vc key names.
        # TODO: If in the future we need some paramters here that can be different every time and do not correspond to a ventricular coordinate, this will need to be rethought.
        # TODO: One option could be to do similiar to the Eikonal code, which has an __unpacking function that translates to the specific parameters expected for the function.
        param_dict = self.__repack_particle_params(parameter_particle)
        apd_max = param_dict[self.apd_max_name]
        apd_min = param_dict[self.apd_min_name]
        node_vc = self.propagation_model.get_node_vc()
        vc_name_list = list(node_vc.keys())
        gradient_map_list = []
        for vc_name in vc_name_list:
            if vc_name in param_dict:
                gradient_map_list.append(node_vc[vc_name] * param_dict[vc_name])
        gradient_map = sum(gradient_map_list)
        normalised_gradient_map = (gradient_map - np.amin(gradient_map))/(np.amax(gradient_map) - np.amin(gradient_map) + 1e-8)
        return normalised_gradient_map * (apd_max - apd_min) + apd_min

    def generate_node_vm(self, parameter_particle, node_lat, node_celltype):
        # node_lat = int(node_lat + 0.5)  # same as ceil and then cast to int
        node_apd = self.generate_node_apd(parameter_particle=parameter_particle)
        node_vm = np.zeros((node_apd.shape[0], int(math.ceil(np.amax(node_lat))) + self.cellular_model.get_max_action_potential_len()))
        # No significant gain in performance by using the call: node_action_potential, node_upstroke_index = self.cellular_model.generate_action_potential_population(
        #             action_potential_duration_population=node_apd, celltype_id_population=node_celltype)
        # instead of calling the function separately for each node_i.
        for node_i in range(node_apd.shape[0]):
            action_potential, upstroke_index = self.cellular_model.generate_action_potential(action_potential_duration=node_apd[node_i],
                                                                                             celltype_id=node_celltype[node_i])
            signal_start_index = int(node_lat[node_i] - upstroke_index)
            signal_end_index = signal_start_index + action_potential.shape[0]
            # Full pre-action potential
            if signal_start_index > 0:
                # pre-action potential
                node_vm[node_i, :signal_start_index] = action_potential[0]
            # No pre-action potential and partial pre-upstroke
            if signal_start_index < 0:
                # action potential with part of the pre-upstroke segment
                node_vm[node_i, :signal_end_index] = action_potential[abs(signal_start_index):]
            # Full pre-upstroke
            else:
                # action potential with pre-upstroke segment
                node_vm[node_i, signal_start_index:signal_end_index] = action_potential
            # These two cases could be combined as: node_vm[node_i, max(0, signal_start_index):signal_end_index] = action_potential[abs(min(0, signal_start_index)):]
            # if signal_end_index < node_vm.shape[1]: # seems to not be needed
            # Full post-action potential
            node_vm[node_i, signal_end_index:] = action_potential[-1]
        return node_vm

    def generate_node_biomarker(self, parameter_particle, node_celltype):
        node_apd = self.generate_node_apd(parameter_particle=parameter_particle)
        # # Initialise dataframe
        # start_node_index = 0
        # node_biomarker = initialise_pandas_dataframe(df=self.cellular_model.generate_biomarker(
        #     action_potential_duration=node_apd[start_node_index],
        #     celltype_id=node_celltype[start_node_index]))
        # print(node_biomarker)
        node_biomarker_list = []
        # Populate dataframe
        for node_i in range(0, node_apd.shape[0]):
            biomarker = initialise_pandas_dataframe(self.cellular_model.generate_biomarker(
                action_potential_duration=node_apd[node_i],
                celltype_id=node_celltype[node_i]))
            biomarker[get_row_id_name()] = node_i
            node_biomarker_list.append(biomarker)
        node_biomarker = pd.concat(node_biomarker_list, ignore_index=True)
        # print('Done')
        # print(node_biomarker)
        return node_biomarker

    # TODO rename function to make it vm specific?
    def apply_spatiotemporal_smoothing(self,
                                       # fibre_speed,
                                       nodefield#,
                                       # normal_speed,
                                       # sheet_speed
                                       ): # TODO node_time_field?
        nodefield_smoothed = np.copy(nodefield)
        # TODO parameterise the effect of the cummulativeness
        # TODO parameterise this:
        # self.full_smoothing_time_index = 400  # (ms) assumming 1000Hz
        # The spatial smoothing is looped over because this way we allow remote effects like in diffusion
        # if self.smoothing_count > 0:
        if self.smoothing_dt > 0:
            # time_jump = int(max(1, round(self.full_smoothing_time_index / self.smoothing_count))) # This is what makes the smoothing dynamic over time (linearly increasing)
            # time_jump = self.smoothing_dt  # This is what makes the smoothing dynamic over time (linearly increasing)
        # time_jump = 0 # TODO Comment this line!! Makes the smoothing static
            current_time_start_smoothing = self.start_smoothing_time_index  # ms Assuming 1000 Hz
            time_end_smoothing = min(self.end_smoothing_time_index, nodefield.shape[1])
            # print('time_end_smoothing ', time_end_smoothing)
            # print('\nHola hola HOLA\n')
            while(current_time_start_smoothing < time_end_smoothing):
            # # for smoothing_i in range(self.smoothing_count):  # Starts at zero
            #     time_start_smoothing = time_jump * smoothing_i
            #     # print('time_start_smoothing ', time_start_smoothing)
            #     # The effect of the spatial smoothing will accumulate over time and remote regions will have an influence
                nodefield_smoothed[:, current_time_start_smoothing:] = \
                    self.propagation_model.spatial_smoothing_of_time_field_using_adjacentcies_orthotropic_fibres(
                        # fibre_speed=fibre_speed,
                        # sheet_speed=sheet_speed,
                        # normal_speed=normal_speed,
                        # ghost_distance_to_self=self.smoothing_ghost_distance_to_self,
                        original_field_data=nodefield_smoothed[:, current_time_start_smoothing:])
                current_time_start_smoothing = current_time_start_smoothing + self.smoothing_dt
        # The temporal smoothing should not be looped over because it only makes sense to influence the present from the past, once
        # warn('Should we have a temporal smoothing as well?')
        # nodefield_smoothed_2 = temporal_smoothing_of_time_field(
        #     original_field_data=nodefield_smoothed, past_present_smoothing_window=self.smoothing_past_present_window)
        # print('diff ', np.sum(np.abs(nodefield_smoothed_2-nodefield_smoothed)))
        # nodefield_smoothed[:, 200:] = nodefield_smoothed[:, 200:]*6
        return nodefield_smoothed
        # for smoothing_i in range(self.smoothing_count):
        #     nodefield_smoothed = self.propagation_model.spatial_smoothing_of_time_field_using_adjacentcies(
        #         original_field_data=nodefield_smoothed, ghost_distance_to_self=self.smoothing_ghost_distance_to_self)
        #     nodefield_smoothed = temporal_smoothing_of_time_field(
        #         original_field_data=nodefield_smoothed, past_present_smoothing_window=self.smoothing_past_present_window)
        # return nodefield_smoothed

    # def apply_spatiotemporal_smoothing(self, nodefield): # TODO node_time_field?
    #     nodefield_smoothed = np.copy(nodefield)
    #     # TODO parameterise the effect of the cummulativeness
    #     # TODO parameterise this:
    #     full_smoothing_time_i = 400  # (ms) assumming 1000Hz
    #     time_jump = int(max(1, round(full_smoothing_time_i/self.smoothing_count)))
    #     for smoothing_i in range(self.smoothing_count):
    #         time_start_smoothing = time_jump * smoothing_i
    #         # print('time_start_smoothing ', time_start_smoothing)
    #         nodefield_smoothed[:, time_start_smoothing:] = self.propagation_model.spatial_smoothing_of_time_field_using_adjacentcies(
    #             original_field_data=nodefield_smoothed[:, time_start_smoothing:], ghost_distance_to_self=self.smoothing_ghost_distance_to_self)
    #         nodefield_smoothed[:, time_start_smoothing:] = temporal_smoothing_of_time_field(
    #             original_field_data=nodefield_smoothed[:, time_start_smoothing:], past_present_smoothing_window=self.smoothing_past_present_window)
    #     return nodefield_smoothed


# Code for checking that the prescribed APD map is being achieved.
#          parameter_particle =[200, 180, 1, 0, 0, 0] # Temporary addition!
#         node_apd = self.generate_node_apd(parameter_particle=parameter_particle)
#         print(parameter_particle)
#         node_xyz = np.loadtxt('/data/Personalisation_projects/meta_data/geometric_data/DTI004/DTI004_coarse/DTI004_coarse_xyz.csv', delimiter=',')
#         fig = plt.figure(figsize=(12, 8))
#         ax = fig.add_subplot(131, projection='3d')
#         p = scatter_visualise(ax=ax, xyz=node_xyz, field=node_apd, title='Node APD')
#         fig.colorbar(p, ax=ax)
#
#
#         node_vm = np.zeros((node_apd.shape[0], int(math.ceil(np.amax(node_lat))) + self.cellular_model.get_max_action_potential_len()))
#         # No significant gain in performance by using the call: node_action_potential, node_upstroke_index = self.cellular_model.generate_action_potential_population(
#         #             action_potential_duration_population=node_apd, celltype_id_population=node_celltype)
#         # instead of calling the function separately for each node_i.
#         for node_i in range(node_apd.shape[0]):
#             action_potential, upstroke_index = self.cellular_model.generate_action_potential(action_potential_duration=node_apd[node_i],
#                                                                                              celltype_id=node_celltype[node_i])
#             signal_start_index = int(node_lat[node_i] - upstroke_index)
#             signal_end_index = signal_start_index + action_potential.shape[0]
#             # Full pre-action potential
#             if signal_start_index > 0:
#                 # pre-action potential
#                 node_vm[node_i, :signal_start_index] = action_potential[0]
#             # No pre-action potential and partial pre-upstroke
#             if signal_start_index < 0:
#                 # action potential with part of the pre-upstroke segment
#                 node_vm[node_i, :signal_end_index] = action_potential[abs(signal_start_index):]
#             # Full pre-upstroke
#             else:
#                 # action potential with pre-upstroke segment
#                 node_vm[node_i, signal_start_index:signal_end_index] = action_potential
#             # These two cases could be combined as: node_vm[node_i, max(0, signal_start_index):signal_end_index] = action_potential[abs(min(0, signal_start_index)):]
#             # if signal_end_index < node_vm.shape[1]: # seems to not be needed
#             # Full post-action potential
#             node_vm[node_i, signal_end_index:] = action_potential[-1]
#         repol_map = generate_repolarisation_map(node_vm)
#         apd_map = repol_map - node_lat
#         ax1 = fig.add_subplot(132, projection='3d')
#         ax2 = fig.add_subplot(133, projection='3d')
#         p1 = scatter_visualise(ax=ax1, xyz=node_xyz, field=repol_map, title='Repol map')
#         fig.colorbar(p1, ax=ax1)
#         p2 = scatter_visualise(ax=ax2, xyz=node_xyz, field=apd_map, title='APD map from Vm')
#         fig.colorbar(p2, ax=ax2)
#         plt.show()
#         quit()
#         return node_vm