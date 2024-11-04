# Class definitions for the cardiac cellular models
from warnings import warn
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

from io_functions import read_pandas
from utils import get_pandas_from_value, get_row_id_name, get_nan_value, change_resolution


class CellularModelEP:
    def __init__(self, verbose):
        if verbose:
            print('Initialising Cellular model')

    '''functionality'''
    def generate_action_potential(self, action_potential_duration, celltype_id):
        raise NotImplementedError

    def generate_action_potential_population(self, action_potential_duration_population, celltype_id_population):
        raise NotImplementedError

    def get_max_action_potential_len(self):
        raise NotImplementedError


class CellularModelCelltypeDictionary(CellularModelEP):
    def __init__(self, list_celltype_name, verbose):
        super().__init__(verbose=verbose)
        # Save the correspondance between celltypes and indexes to sub-dictionaries
        self.celltype_name_from_id = {}  # indexed by id
        self.celltype_id_from_name = {}  # indexed by name
        self.action_potential_from_id = []  # list of arrays, where each celltype has its own array.
        self.time_from_id = []
        self.upstroke_index_from_id = []
        # Celltype IDs are allways positive integers
        self.celltype_invalid_value = -1
        for celltype_i in range(len(list_celltype_name)):
            self.celltype_id_from_name[list_celltype_name[celltype_i]] = celltype_i
            self.celltype_name_from_id[celltype_i] = list_celltype_name[celltype_i]
            self.action_potential_from_id.append(np.array([]))
            self.time_from_id.append(np.array([]))
            self.upstroke_index_from_id.append(np.array([]))
        self.verbose = verbose

    '''celltype'''
    def get_celltype_id(self, celltype_name):
        return self.celltype_id_from_name[celltype_name]

    def get_celltype_name(self, celltype_id):
        return self.celltype_name_from_id[celltype_id]

    def get_celltype_invalid_value(self):
        return self.celltype_invalid_value

    def get_celltype_name_list(self):
        return list(self.celltype_id_from_name.keys())

    def get_celltype_to_id_correspondence(self):
        return self.celltype_name_from_id

    '''AP'''
    def set_action_potential_for_celltype(self, action_potential_population, celltype_id):
        self.action_potential_from_id[celltype_id] = action_potential_population

    def get_action_potential_for_celltype(self, celltype_id):
        return self.action_potential_from_id[celltype_id]

    '''time series'''
    def set_time_for_celltype(self, time_population, celltype_id):
        self.time_from_id[celltype_id] = time_population

    def get_time_for_celltype(self, celltype_id):
        return self.time_from_id[celltype_id]

    '''upstroke'''
    def set_upstroke_index_for_celltype(self, upstroke_index_population, celltype_id):
        self.upstroke_index_from_id[celltype_id] = upstroke_index_population

    def get_upstroke_index_for_celltype(self, celltype_id):
        return self.upstroke_index_from_id[celltype_id]

    '''biomarker'''
    def get_biomarker_range(self, biomarker_name):
        raise NotImplementedError

    '''available action potentials'''
    def get_aligned_action_potential_from_celltype_name(self, celltype_name):
        celltype_id = self.get_celltype_id(celltype_name=celltype_name)
        unaligned_action_potential_population = self.get_action_potential_for_celltype(celltype_id=celltype_id)
        upstroke_index_population = self.get_upstroke_index_for_celltype(celltype_id=celltype_id)
        # Get the min upstroke index to determine how much margin can we give while having the APs aligned
        min_upstroke_index = 1000
        for upstroke_index_population_i in range(len(upstroke_index_population)):
            min_upstroke_index = min(upstroke_index_population[upstroke_index_population_i], min_upstroke_index)
        correction_upstroke_index = min(5, min_upstroke_index)  # TODO why 5? Explain?
        print('correction_upstroke_index ', correction_upstroke_index)
        print('min_upstroke_index ', min_upstroke_index)
        # Align APs using upstroke index corrected using the correction_upstroke_index
        aligned_action_potential_population = []
        for unaligned_action_potential_i in range(len(unaligned_action_potential_population)):
            unaligned_action_potential = unaligned_action_potential_population[unaligned_action_potential_i]
            upstroke_index = upstroke_index_population[unaligned_action_potential_i] - correction_upstroke_index
            aligned_action_potential_population.append(unaligned_action_potential[upstroke_index:])
        return aligned_action_potential_population


# TODO should this actually inherit from the dictionary class or just from the cellular EP class?
class StepFunctionUpstrokeEP(CellularModelCelltypeDictionary):
    def __init__(self, resting_vm_value, upstroke_vm_value, verbose):
        super().__init__(list_celltype_name=[], verbose=verbose)
        self.resting_vm_value = resting_vm_value
        self.upstroke_vm_value = upstroke_vm_value

    def generate_action_potential(self, action_potential_duration, celltype_id):
        warn('Use this function with caution! This class is intended to provide resting and upstroke values instead')
        action_potential = np.full((action_potential_duration), self.upstroke_vm_value, dtype=np.float64)
        action_potential[0] = 0  # first value of AP should be baseline
        upstroke_index = 1
        return action_potential, upstroke_index

    def get_resting_vm(self):
        return self.resting_vm_value

    def get_upstroke_vm(self):
        return self.upstroke_vm_value


# TODO this could be handled using only pandas objects instead of a dictionary of pandas objects
class CellularModelBiomarkerDictionary(CellularModelCelltypeDictionary):
    def __init__(self, biomarker_upstroke_name, biomarker_apd90_name, biomarker_celltype_name, cellular_data_dir,
                 cellular_model_name, list_celltype_name, verbose):
        super().__init__(list_celltype_name=list_celltype_name, verbose=verbose)
        # Using dictionaries slows down the code massively because it loses the ability to index multiple APDs at once.
        # Required properties for the speedup:
        # - Have an array of arrays instead of a hash table, where every location in the array can only have one action potential. This implies that all indexes need to be positive integers (no floats allowed).
        # - It needs to allow for having multiple cell types, and it needs a way to remember where each cell type is stored.
        # - It will use cycle_length as the new length for the action potentials.
        # - Assumes that the action potentials are in the same order as the biomarkers and that there is a biomarker entry for each action potential.
        self.biomarker_upstroke_name = biomarker_upstroke_name
        self.biomarker_apd90_name = biomarker_apd90_name
        self.biomarker_celltype_name = biomarker_celltype_name
        # self.biomarker_len_name = biomarker_len_name
        self.biomarker_from_celltype_id = {}
        max_action_potential_len = 0
        for celltype_name in list_celltype_name:
            celltype_id = self.get_celltype_id(celltype_name=celltype_name)
            action_potential_population = np.loadtxt(
                cellular_data_dir + cellular_model_name + '_vm_' + celltype_name + '.csv', delimiter=',')
            time_population = np.loadtxt(
                cellular_data_dir + cellular_model_name + '_time_' + celltype_name + '.csv', delimiter=',')
            # NOTE: Different celltypes can have different lengths of action potential - this can be useful when the ECG computation is the bottleneck, because it could compute less for the vms with less duration
            self.set_action_potential_for_celltype(action_potential_population=action_potential_population,
                                                   celltype_id=celltype_id)
            max_action_potential_len = max(max_action_potential_len, action_potential_population.shape[1])
            self.set_time_for_celltype(time_population=time_population, celltype_id=celltype_id)
            celltype_biomarker = read_pandas(filename=cellular_data_dir + 'biomarkers_table_' + celltype_name + '.csv')
            celltype_biomarker[self.biomarker_celltype_name] = celltype_name
            self.biomarker_from_celltype_id[celltype_id] = celltype_biomarker
            self.set_upstroke_index_for_celltype(
                upstroke_index_population=self.__generate_upstroke_index_population(celltype_id=celltype_id),
                celltype_id=celltype_id)
        # Make all action potentials from different celltypes the same length to ease their stacking into matrices
        # # TODO: this implies that the time and cai arrays may be shorter than the vm array, however, it's just the last time repeated over and over, rather than additional time, so, it should be fine
        # for celltype_name_i in range(len(list_celltype_name)):
        #     celltype_name = list_celltype_name[celltype_name_i]
        #     celltype_id = self.get_celltype_id(celltype_name=celltype_name)
        #     aux_action_potential_population = action_potential_for_celltype_list[celltype_name_i]
        #     if aux_action_potential_population.shape[1] < max_action_potential_len:
        #         action_potential_population = np.zeros((aux_action_potential_population.shape[0], max_action_potential_len))
        #         action_potential_population[:, :aux_action_potential_population.shape[1]] = aux_action_potential_population
        #         action_potential_population[:, aux_action_potential_population.shape[1]:] = aux_action_potential_population[:, -1]  # TODO test this indexing, it may be missing a np.newaxis somewhere, or a for loop
        #     else:
        #         action_potential_population = aux_action_potential_population
        #     self.set_action_potential_for_celltype(action_potential_population=action_potential_population,
        #                                            celltype_id=celltype_id)
        self.max_action_potential_len = int(max_action_potential_len)

    def get_action_potential_index_from_apd(self, action_potential_duration, celltype_id):
        celltype_apd90, cellytpe_row_index = self.get_biomarker_from_celltype_id(
            biomarker_name=self.biomarker_apd90_name,
            celltype_id=celltype_id)
        action_potential_meta_index = np.argmin(np.abs(celltype_apd90 - action_potential_duration))
        action_potential_index = cellytpe_row_index[action_potential_meta_index]
        return action_potential_index

    '''functionality'''
    def generate_action_potential(self, action_potential_duration, celltype_id):
        action_potential_index = self.get_action_potential_index_from_apd(
            action_potential_duration=action_potential_duration, celltype_id=celltype_id)
        return self.get_action_potential_for_celltype(celltype_id=celltype_id)[action_potential_index, :], \
            self.get_upstroke_index_for_celltype(celltype_id=celltype_id)[action_potential_index]

    def generate_biomarker(self, action_potential_duration, celltype_id):
        action_potential_index = self.get_action_potential_index_from_apd(
            action_potential_duration=action_potential_duration, celltype_id=celltype_id)
        biomarker = self.get_biomarker_from_celltype_id_and_index(celltype_id=celltype_id,
                                                                  row_index=action_potential_index)
        return biomarker

    # TODO following function is well coded but currently NOT in use
    # def generate_action_potential_population(self, action_potential_duration_population, celltype_id_population):
    #     celltype_id_population_unique = np.unique(celltype_id_population)
    #     action_potential_population = np.zeros((action_potential_duration_population.shape[0], self.get_max_action_potential_len()))
    #     upstroke_index_population = np.zeros((action_potential_duration_population.shape[0]), dtype=int)
    #     for celltype_id in celltype_id_population_unique:
    #         action_potential_index = np.argmin(np.abs(
    #             self.get_biomarker_from_celltype_id(biomarker_name=self.biomarker_apd90_name,
    #                                                 celltype_id=celltype_id)[:, np.newaxis] - action_potential_duration_population), axis=0)
    #         celltype_index = celltype_id_population == celltype_id
    #         celltype_action_potential = self.get_action_potential_for_celltype(celltype_id=celltype_id)[action_potential_index, :][celltype_index, :]
    #         action_potential_population[celltype_index, :celltype_action_potential.shape[1]] = celltype_action_potential
    #         upstroke_index_population[celltype_index] = self.get_upstroke_index_for_celltype(celltype_id=celltype_id)[action_potential_index][celltype_index]
    #     return action_potential_population, upstroke_index_population

    '''biomarker'''
    def get_biomarker_from_celltype_id(self, biomarker_name, celltype_id):
        # print('get_biomarker_from_celltype_id biomarker_name ', biomarker_name, ' celltype_id ', celltype_id)   # TODO delete line
        return self.biomarker_from_celltype_id[celltype_id][biomarker_name].values, \
        self.biomarker_from_celltype_id[celltype_id][get_row_id_name()].values

    def get_biomarker_from_celltype_id_and_index(self, celltype_id, row_index):
        return get_pandas_from_value(df=self.biomarker_from_celltype_id[celltype_id], key=get_row_id_name(),
                                     value=row_index)

    def get_biomarker_range(self, biomarker_name):
        aux_max = None
        aux_min = None
        for celltype_name in self.get_celltype_name_list():
            celltype_id = self.get_celltype_id(celltype_name)
            biomarker_value, biomarker_index = self.get_biomarker_from_celltype_id(biomarker_name=biomarker_name,
                                                                                   celltype_id=celltype_id)
            if aux_max is None:
                aux_max = np.amax(biomarker_value)
            else:
                aux_max = min(aux_max, np.amax(biomarker_value))
            if aux_min is None:
                aux_min = np.amin(biomarker_value)
            else:
                aux_min = max(aux_min, np.amin(biomarker_value))
        return int(aux_min), int(aux_max)

    '''max ap length'''
    def get_max_action_potential_len(self):
        return self.max_action_potential_len

    def __generate_upstroke_index_population(self, celltype_id):
        upstroke_time_population, biomarker_index = self.get_biomarker_from_celltype_id(
            biomarker_name=self.biomarker_upstroke_name,
            celltype_id=celltype_id)
        upstroke_index_population = np.argmin(
            np.abs(upstroke_time_population[:, np.newaxis] - self.get_time_for_celltype(
                celltype_id=celltype_id)), axis=1)
        return upstroke_index_population


# TODO restructure classes so that this is just a cell model, maybe we can create an empty or ToRORd cell model class
class MitchellSchaefferEPCelltypeDictionary(CellularModelCelltypeDictionary):
    def __init__(self, cycle_length, list_celltype_name, verbose, vm_max, vm_min):
        super().__init__(list_celltype_name=list_celltype_name, verbose=verbose)
        self.cycle_length = cycle_length
        self.vm_max = vm_max
        self.vm_min = vm_min


    def generate_action_potential_from_scratch(self, action_potential_duration):
        # Inputs: apd - prescribed action potential duration at 90% repolarisation
        # cl - Cycle length.
        # Outputs: apd90 - actual apd90 from MS model
        # Vm - Cellular action potential transient for the duration of the prescribed cycle length.
        t = np.linspace(0, self.cycle_length, self.cycle_length)  # + 1) # TODO revert
        y0 = [0, 1]  # Starting states for Vm and h gating
        dummy_argument = 1
        sol = odeint(self.mitchell_schaeffer, y0, t, args=(action_potential_duration, dummy_argument))
        vm = sol[:, 0]
        vm = vm * (self.vm_max - self.vm_min) + self.vm_min
        # vm = vm * 110 - 90
        # # Evaluate apd90 of solution Vm
        # maxVm = np.amax(sol, axis=0)[0]
        # apd90 = np.where(sol[50:, 0] < 0.1 * maxVm)[0][0] + 50
        return vm

    '''functionality'''
    def generate_action_potential(self, action_potential_duration, celltype_id):
        '''Returns AP and upstroke index in the AP'''
        vm = self.generate_action_potential_from_scratch(action_potential_duration=action_potential_duration)
        upstroke_index = self.get_upstroke_index_for_ap(ap=vm)
        return vm, upstroke_index

    def get_max_action_potential_len(self):
        return self.cycle_length

    def get_upstroke_index_for_ap(self, ap):
        return 0

    # def get_celltype_invalid_value(self):
    #     return get_nan_value()

    # def generate_action_potential_population(self, action_potential_duration_population):
    #     action_potential_duration_population_unique, unique_indexes = np.unique(action_potential_duration_population,
    #                                                                             return_inverse=True, axis=0)
    #     lat_population_unique = self.propagation_model.simulate_propagation_population(
    #         parameter_population=parameter_population_unique)
    #     ap_simulation = self.cellular_model.generate_action_potential()
    #     # Simulate local activation times in a population of particles using the EikonalDjikstraTet model.
    #     vm_population_unique = pymp.shared.array((lat_population_unique.shape[0], lat_population_unique.shape[1],
    #                                               np.amax(lat_population_unique) + 1), dtype=np.float64)
    #     vm_population_unique[:, :,
    #     :] = get_nan_value()  # TODO be careful that there are no NAN values in the ECG calculation
    #     threads_num = multiprocessing.cpu_count()
    #     with pymp.Parallel(min(threads_num, vm_population_unique.shape[0])) as p1:
    #         for conf_i in p1.range(vm_population_unique.shape[0]):
    #             vm_population_unique[conf_i, :] = self.calculate_vm_map(ap=ap_simulation,
    #                                                                     lat=lat_population_unique[conf_i, :])
    #     return lat_population_unique[unique_indexes, :], vm_population_unique[unique_indexes, :, :]

    '''support functions'''
    def get_apd_90(self, action_potential):
        # Evaluate apd90 of action potential vm
        max_vm = np.amax(action_potential)
        return np.where(action_potential[50:] < 0.1 * max_vm)[0][0] + 50

    def mitchell_schaeffer(self, vm_h, t, apd,
                           dummy_arg):  # TODO: make into class so that we can swap cellular models at ease
        # Inputs:
        # vm_h - membrane potential Vm (mV) and gating variable h (dimensionless)
        # t - time (ms)
        # apd - prescribed action potential duration at 90% repolarisation (ms)
        # dummy_arg - dummy argument required for interfacing with odeint ODE solver.
        # Outputs:
        # dydt - vector containing time derivatives of Vm (dVmdt) and h (dhdt)
        #
        # Parameter values from Table 2, Gillette, K., Gsell, M. A., Prassl, A. J., Karabelas,
        # E., Reiter, U., Reiter, G., ... & Plank, G. (2021). A framework for the
        # generation of digital twins of cardiac electrophysiology from clinical
        # 12-leads ECGs. Medical Image Analysis, 71, 102080.
        #
        stim_amp = 0.05
        stim_dur = 5
        vm = vm_h[0]
        h = vm_h[1]
        tau_out = 5.4
        tau_in = 0.3
        tau_open = 80
        tau_close = apd / np.log(
            tau_out / (2.9 * tau_in))  # Gilette paper has 4*tau_in, this gave us larger apd errors than 2.9.
        V_min = -86.2
        V_max = 40.0
        vm_gate = 0.1  # Original Mitchel Schaeffer paper has 0.13 for Vm,gate, we changed this to get better apd match.
        Cm = 1
        if vm < vm_gate:
            dhdt = (1 - h) / tau_open
        else:
            dhdt = -h / tau_close
        J_in = h * vm ** 2 * (1 - vm) / tau_in
        J_out = - vm / tau_out
        I_ion = (J_in + J_out)
        amp = stim_amp
        duration = stim_dur
        if t <= duration:
            Istim = amp
        else:
            Istim = 0.0
        dVmdt = I_ion + Istim
        dydt = [dVmdt, dhdt]
        return dydt

    def check_ms_apd_accuracy(self, apd_min, apd_max, cl):
        result_apds = []
        x = np.linspace(apd_min, apd_max, apd_max - apd_min + 1)
        for i in range(x.shape[0]):
            vm = self.generate_action_potential(action_potential_duration=i + apd_min, cycle_length=cl)
            apd90 = self.get_apd_90(action_potential=vm)
            result_apds.append(apd90)
        plt.plot(x, (np.asarray(result_apds) - x))
        plt.xlabel('apd-prescribed (ms)')
        plt.ylabel('apd-MS - apd-prescribed (ms)')
        plt.title('Error in Mitchell Schaeffer apd simulation')
        plt.show(block=False)
        print('RMSE in MS apd: ' + str(np.sqrt(np.mean((x - np.asarray(result_apds)) ** 2))))
        return np.sqrt(np.mean((x - np.asarray(result_apds)) ** 2))


# Refactor this class to allow for different celltypes like the class version that reads the data from files (CellularModelBiomarkerDictionary)
class MitchellSchaefferAPDdictionary(MitchellSchaefferEPCelltypeDictionary):
    def __init__(self, apd_max, apd_min, apd_resolution, cycle_length, list_celltype_name, verbose, vm_max, vm_min):
        super().__init__(cycle_length=cycle_length, list_celltype_name=list_celltype_name, verbose=verbose,
                         vm_max=vm_max, vm_min=vm_min)
        self.apd_dictionary = self.generate_apd_dictionary(apd_max=apd_max, apd_min=apd_min,
                                                           apd_resolution=apd_resolution)
        self.apd_resolution = apd_resolution

    '''functionality'''
    def generate_action_potential(self, action_potential_duration, celltype_id):
        action_potential_duration = change_resolution(data=action_potential_duration, resolution=self.apd_resolution)
        vm = self.get_action_potential_from_celltype_dict(action_potential_duration=action_potential_duration,
                                                          celltype_id=celltype_id)
        upstroke_index = self.get_upstroke_index_for_ap(ap=vm)
        return vm, upstroke_index

    def generate_apd_dictionary(self, apd_max, apd_min, apd_resolution):
        apd_dictionary = {}
        for apd_i in range(apd_min, apd_max+1, apd_resolution):
            apd_dictionary[apd_i] = self.generate_action_potential_from_scratch(action_potential_duration=apd_i)
            # plt.plot(apd_dictionary[apd_i])
        # plt.hlines(0, 0, 800, 'r')
        # plt.show(block=False)
        return apd_dictionary

    def get_action_potential_from_celltype_dict(self, action_potential_duration, celltype_id):
        return self.apd_dictionary[action_potential_duration]

# EOF
