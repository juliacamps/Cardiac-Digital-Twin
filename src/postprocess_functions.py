import multiprocessing
import os
import numpy as np
import pymp
# import matplotlib
# matplotlib.use('tkagg')  # For use on CSCS daint
# from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import seaborn as sns
# import time
import argparse
import pickle
from scipy.signal import lfilter, resample
from scipy import io
from warnings import warn
import pandas as pd

from utils import get_pandas_from_value, get_nan_value


class V():
    def __init__(self, i, name, visualisation_dir, visualisation_format):
        self.name = name
        self.visualisation_dir = visualisation_dir
        self.visualisation_format = visualisation_format
        # self.I = i

    def _ensight_export_mesh(self, nodes, elems):
        print('ENSIGHT EXPORT: '+self.visualisation_dir+self.name+'.ensi.geo')
        with open(self.visualisation_dir+self.name+'.ensi.geo', 'w') as f:
            f.write('Problem name:  '+self.name+'\nGeometry file\nnode id given\nelement id given\npart\n\t1\nVolume Mesh\ncoordinates\n'+str(len(nodes))+'\n')
            for i in range(0, len(nodes)):
                f.write(str(i+1)+'\n')
            for c in [0,1,2]:
                for i in range(0, len(nodes)):
                    f.write(str(nodes[i,c])+'\n')
            f.write('tetra4\n  '+str(len(elems))+'\n')
            for i in range(0, len(elems)):
                f.write('  '+str(i+1)+'\n')
            for i in range(0, len(elems)):
                f.write(str(elems[i,0])+'\t'+str(elems[i,1])+'\t'+str(elems[i,2])+'\t'+str(elems[i,3])+'\n')
        with open(self.visualisation_dir+self.name+'.ensi.case', 'w') as f:
            f.write('#\n# Alya generated post-process files\n# Ensight Gold Format\n#\n# Problem name:\t'+self.name+'\n#\n')
            f.write('FORMAT\ntype:\tensight gold\nGEOMETRY\nmodel:\t1\t'+self.name+'.ensi.geo\nVARIABLE\n')


    def _ensight_export_scalar_per_node(self, field_name, data):
        print('ENSIGHT EXPORT: '+self.visualisation_dir+self.name+'.ensi.'+field_name)
        with open(self.visualisation_dir+self.name+'.ensi.'+field_name, 'w') as f:
            f.write('Alya Ensight Gold --- Scalar per-node variables file\npart\n\t1\ncoordinates\n')
            for i in range(0, len(data)):
                f.write(str(data[i])+'\n')
        if os.path.exists(self.visualisation_dir+self.name+'.ensi.case'):
            with open(self.visualisation_dir+self.name+'.ensi.case', 'a') as f:
                f.write('scalar per node:	1	'+field_name+'	'+self.name+'.ensi.'+field_name+'\n')
        else:
            with open(self.visualisation_dir+self.name+'.ensi.case', 'w') as f:
                f.write('#\n# Alya generated post-process files\n# Ensight Gold Format\n#\n# Problem name:\t'+self.name+'\n#\n')
                f.write('FORMAT\ntype:\tensight gold\nGEOMETRY\nmodel:\t1\t'+self.name+'.ensi.geo\nVARIABLE\n')
                f.write('scalar per node:	1	'+field_name+'	'+self.name+'.ensi.'+field_name+'\n')

    def _ensight_export_vector_per_node(self, field_name, data):
        print('ENSIGHT EXPORT: '+self.visualisation_dir+self.name+'.ensi.'+field_name)
        with open(self.visualisation_dir+self.name+'.ensi.'+field_name, 'w') as f:
            f.write('Alya Ensight Gold --- Vector per-node variables file\npart\n1\ncoordinates\n')
            for c in [0, 1, 2]:
                for i in range(0, len(data)):
                    f.write(str(data[i,c])+'\n')
        if os.path.exists(self.visualisation_dir+self.name+'.ensi.case'):
            with open(self.visualisation_dir+self.name+'.ensi.case', 'a') as f:
                f.write('vector per node:	1	'+field_name+'	'+self.name+'.ensi.'+field_name+'\n')
        else:
            with open(self.visualisation_dir+self.name+'.ensi.case', 'w') as f:
                f.write('#\n# Alya generated post-process files\n# Ensight Gold Format\n#\n# Problem name:\t'+self.name+'\n#\n')
                f.write('FORMAT\ntype:\tensight gold\nGEOMETRY\nmodel:\t1\t'+self.name+'.ensi.geo\nVARIABLE\n')
                f.write('vector per node:	1	'+field_name+'	'+self.name+'.ensi.'+field_name+'\n')

    def _ensight_export_scalar_per_cell(self, field_name, data):
        print('ENSIGHT EXPORT: '+self.visualisation_dir+self.name+'.ensi.'+field_name)
        with open(self.visualisation_dir+self.name+'.ensi.'+field_name, 'w') as f:
                f.write('Alya Ensight Gold --- Scalar per-cell variables file\npart\n\t1\ntetra4\n')
                for i in range(0, len(data)):
                    f.write(str(data[i])+'\n')
        if os.path.exists(self.visualisation_dir+self.name+'.ensi.case'):
            with open(self.visualisation_dir+self.name+'.ensi.case', 'a') as f:
                f.write('scalar per element:	1	'+field_name+'	'+self.name+'.ensi.'+field_name+'\n')
        else:
            with open(self.visualisation_dir+self.name+'.ensi.case', 'w') as f:
                f.write('#\n# Alya generated post-process files\n# Ensight Gold Format\n#\n# Problem name:\t'+self.name+'\n#\n')
                f.write('FORMAT\ntype:\tensight gold\nGEOMETRY\nmodel:\t1\t'+self.name+'.ensi.geo\nVARIABLE\n')
                f.write('scalar per element:	1	'+field_name+'	'+self.name+'.ensi.'+field_name+'\n')


def scatter_visualise_point_cloud(xyz, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter_visualise_field(ax, xyz, field=np.ones((xyz.shape[0])), title=title)
    plt.show()


def scatter_visualise_field(ax, xyz, field, title):
    assert xyz.shape[0] == field.shape[0]
    p = ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=field, marker='o', s=1)
    ax.set_title(title)
    return p


def visualise_ecg(ecg_list, lead_name_list=None, axes=None, ecg_color='gray', fig=None, label_list=None, linewidth=1.,
                  time_steps=None):
    # if ecg_color is None:
    #     ecg_color = 'gray'
    ecg_0 = ecg_list[0]
    nb_leads = ecg_0.shape[0] #len(lead_name_list)
    ecg_0 = None # Clear argument to avoid recycling
    # TODO THe following complexity may be unnecessary
    # Check if the figure already has been started to generate
    if fig is None or axes is None:
        # Try to make half as many rows as columns
        # rows*cols=nb_leads; rows=cols/2; cols/2*cols=nb_leads; cols=(nb_leads*2)**0.5
        nb_cols = (nb_leads * 2) ** 0.5
        if nb_cols - int(nb_cols) == 0. and nb_cols / 2 - int(nb_cols / 2) == 0.:
            nb_rows = nb_cols / 2
        else:
            # Try to make 2 rows and the necessary columns
            nb_cols = nb_leads / 2
            if nb_cols - int(nb_cols) == 0. and nb_cols / 2 - int(nb_cols / 2) == 0.:
                nb_rows = nb_cols / 2
            else:
                warn('This number of leads cannot be plotted in the default configurations!')
                nb_cols = nb_leads
                nb_rows = 1
        # TODO Automatically Make figure size similar to the ECG figures to obtain the same font sizes everywhere
        fig, axes = plt.subplots(int(nb_rows), int(nb_cols), figsize=(20, 10), constrained_layout=True)
        # fig.tight_layout()
        axes = np.reshape(axes, nb_leads)
    # Iterate per leads as each lead is in a separate sub-figure
    for lead_i in range(nb_leads):
        # Iterate for each ecg allowing them to have different lengths from each other
        for ecg_i in range(len(ecg_list)):
            ecg = ecg_list[ecg_i]
            if time_steps is None:
                time_steps = np.arange(ecg.shape[1])
            if label_list is not None:
                label = label_list[ecg_i]
            else:
                label = None
            axes[lead_i].plot(time_steps, ecg[lead_i, :], color=ecg_color, label=label, linewidth=linewidth)
        # Clinical ECG plotting
        # if reference_ecg is not None:
        #     time_steps = np.arange(reference_ecg.shape[1])
        #     axes[lead_i].plot(time_steps, reference_ecg[lead_i, :], label='Clinical', color='lime', linewidth=linewidth)
        if lead_name_list is not None:
            axes[lead_i].set_title(lead_name_list[lead_i])
            axes[lead_i].title.set_size(24)
        else:
            axes[lead_i].set_title(lead_i)
        axes[lead_i].set_ylim([-1.5, 1.5])
        for tick in axes[lead_i].xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        for tick in axes[lead_i].yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
    # Return the figure objects to allow adding more ecg traces to it
    return axes, fig


def visualise_biomarker(biomarker_list, biomarker_name_list, axes=None, biomerker_color='gray', biomarker_marker='.',
                        biomarker_size=None, fig=None, ground_truth_color='lime', ground_truth_biomarker=None,
                        label_list=None, x_axis_value=0):
    # if biomerker_color is None:
    #     biomerker_color = 'gray'
    # print('visualise_biomarker')
    # print('biomarker_name_list ', biomarker_name_list)
    nb_biomarker = len(biomarker_name_list)
    # Check if the figure already has been started to generate
    if fig is None or axes is None:
        # TODO Automatically Make figure size similar to the ECG figures to obtain the same font sizes everywhere
        fig, axes = plt.subplots(1, int(nb_biomarker), figsize=(20, 5), constrained_layout=True) #figsize=(nb_biomarker*5, 5))
        # fig.tight_layout()
        axes = np.reshape(axes, nb_biomarker)
    # Iterate per leads as each lead is in a separate sub-figure
    for biomarker_name_i in range(nb_biomarker):
        # Iterate for each ecg allowing them to have different lengths from each other
        for biomarker_value_i in range(len(biomarker_list)):
            biomarker_value = biomarker_list[biomarker_value_i, biomarker_name_i]
            if label_list is not None:
                label = label_list[biomarker_value_i]
            else:
                label = None
            axes[biomarker_name_i].plot(x_axis_value, biomarker_value, color=biomerker_color, markersize=biomarker_size, #linewidth=biomarker_size,
                                        marker=biomarker_marker, label=label)
        axes[biomarker_name_i].set_title(biomarker_name_list[biomarker_name_i])
        if ground_truth_biomarker is not None:
            ground_truth_biomarker_value = ground_truth_biomarker[biomarker_name_i]
            axes[biomarker_name_i].axhline(ground_truth_biomarker_value, color=ground_truth_color, linewidth=2.)
        for tick in axes[biomarker_name_i].xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        for tick in axes[biomarker_name_i].yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
    # Return the figure objects to allow adding more ecg traces to it
    return axes, fig


def visualise_action_potential_population(action_potential_list, axes=None, action_potential_color='gray',
                                          fig=None, label=None):
    # Check if the figure already has been started to generate
    if fig is None or axes is None:
        fig, axes = plt.subplots(figsize=(5, 5))
    # Iterate per action potentials in the list
    for action_potential_i in range(len(action_potential_list)):
        action_potential = action_potential_list[action_potential_i]
        axes.plot(action_potential, color=action_potential_color, linewidth=.01)
    axes.plot(action_potential, color=action_potential_color, linewidth=.01, label=label)
    for tick in axes.xaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
    for tick in axes.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
    # Return the figure objects to allow adding more action potential types
    return axes, fig


# def plot_ecg(ecg_list, ecg_label_list, lead_names):
#     nb_leads = len(lead_names)
#     nb_cols = (nb_leads * 2) ** 0.5
#     if nb_cols - int(nb_cols) == 0. and nb_cols / 2 - int(nb_cols / 2) == 0.:
#         nb_rows = nb_cols / 2
#     else:
#         # Try to make 2 rows and the necessary columns
#         nb_cols = nb_leads / 2
#         if nb_cols - int(nb_cols) == 0. and nb_cols / 2 - int(nb_cols / 2) == 0.:
#             nb_rows = nb_cols / 2
#         else:
#             warn('This number of leads cannot be plotted in the default configurations!')
#             nb_cols = nb_leads
#             nb_rows = 1
#     fig, axes = plt.subplots(int(nb_rows), int(nb_cols), figsize=(20, 10))
#     axes = np.reshape(axes, nb_leads)
#     for lead_i in range(nb_leads):
#         for ecg_i in range(len(ecg_list)):
#             time_steps = np.arange(len(ecg_list[ecg_i][lead_i, :]))
#             axes[lead_i].plot(time_steps, ecg_list[ecg_i][lead_i, :],  label=ecg_label_list[ecg_i], linewidth=1.0)
#         axes[lead_i].set_title(lead_names[lead_i], fontsize=20)
#         axes[lead_i].set_ylim([-1.5, 1.5])
#         for tick in axes[lead_i].xaxis.get_major_ticks():
#             tick.label1.set_fontsize(14)
#         for tick in axes[lead_i].yaxis.get_major_ticks():
#             tick.label1.set_fontsize(14)
#     axes[lead_i].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
#     plt.show(block=False)
#     return fig


def visualise_heatmap_sa(x_axis_df, y_axis_df):
    # These dataframes have number of rows as number of samples in the SA and columns for each data they store.
    x_axis_column_names = list(x_axis_df.keys())
    y_axis_column_names = list(y_axis_df.keys())
    if x_axis_df.equals(y_axis_df):
        data = x_axis_df
    else:
        data = pd.concat([x_axis_df, y_axis_df], axis=1)
    fig = plt.figure(figsize=(8, 8))
    data_corr = data.corr()
    data_corr = data_corr.loc[x_axis_column_names]  # Select only relevant indexes
    data_corr = data_corr[y_axis_column_names]  # Select only relevant columns
    corrplot(data_corr, size_scale=300)
    plt.show()
    return fig


def visualise_scatter_sa(x_axis_df, y_axis_df):
    # These dataframes have number of rows as number of samples in the SA and columns for each data they store.
    x_axis_column_names = list(x_axis_df.keys())
    y_axis_column_names = list(y_axis_df.keys())
    # Scatter plots with correlation coefficients
    if len(y_axis_df.shape) == 1:
        nb_y_axis_columns = 1
    else:
        nb_y_axis_columns = y_axis_df.shape[1]
    nb_x_axis_columns = x_axis_df.shape[1]
    fig = plt.figure(tight_layout=True, figsize=(18, 10))
    # fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    fig.suptitle('N=' + str(y_axis_df.shape[0]))
    gs = GridSpec(nb_y_axis_columns, x_axis_df.shape[1])
    for y_axis_column_i in range(nb_y_axis_columns):
        # If the plot is against itself, only plot lower diagonal
        if x_axis_df.equals(y_axis_df):
            x_axis_column_end = y_axis_column_i
        else:
            x_axis_column_end = nb_x_axis_columns
        for x_axis_column_i in range(0, x_axis_column_end, 1):
            ax = fig.add_subplot(gs[y_axis_column_i, x_axis_column_i])
            x_axis_column_data = x_axis_df.values[:, x_axis_column_i]
            if nb_y_axis_columns == 1:
                y_axis_column_data = y_axis_df.values
            else:
                y_axis_column_data = y_axis_df.values[:, y_axis_column_i]
            sns.regplot(x=x_axis_column_data, y=y_axis_column_data, ax=ax, scatter_kws={'s': 1})
            ax.text(x=np.amin(x_axis_column_data), y=np.amax(y_axis_column_data), va='top', ha='left',
                    s='p=%.2f' % (np.corrcoef(x_axis_column_data, y_axis_column_data)[0, 1]), fontsize=18,
                    bbox=dict(facecolor='white', alpha=0.75))
            if y_axis_column_i == nb_y_axis_columns - 1:
                ax.set_xlabel(x_axis_column_names[x_axis_column_i], fontsize=14)
            if x_axis_column_i == 0:
                ax.set_ylabel(y_axis_column_names[y_axis_column_i], fontsize=14)
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(14)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(14)
    return fig


def visualise_tornado_sa(qoi_name_list_list, sobol_indices_df, sobol_indices_name_list_in_order, theta_name_list,
                         value_column_name):
    num_figure_rows = len(qoi_name_list_list)
    num_sobol_indecies = len(sobol_indices_name_list_in_order)
    sobol_inicies_colour_list = ['pastel', 'muted', 'dark']     # TODO the following constraint means that this will only be able to cope with up to 2 Sobol indices at once
    assert len(sobol_inicies_colour_list) >= num_sobol_indecies # TODO this should be automatic, to have as many colours as needed
    fig = plt.figure(tight_layout=True, figsize=(10, 6*num_figure_rows))    # Arbitrary size that looks good
    # Count max number of QOIs
    num_qois = 0
    for figure_row_i in range(num_figure_rows):
        qoi_name_list = qoi_name_list_list[figure_row_i]
        num_qois = max(len(qoi_name_list), num_qois)
    # Define figure shape
    gs = GridSpec(num_figure_rows, num_qois)
    sns.set_theme(style='whitegrid')
    # Populate figure (split into subfigures if the input is a list of lists)
    for figure_row_i in range(num_figure_rows):
        qoi_name_list = qoi_name_list_list[figure_row_i]
        num_qois = len(qoi_name_list)
        # Generate each subfigure here:
        for qoi_i in range(num_qois):
            qoi_name = qoi_name_list[qoi_i]
            ax = fig.add_subplot(gs[figure_row_i, qoi_i])
            qoi_df = sobol_indices_df.loc[qoi_name].loc[sobol_indices_name_list_in_order]
            qoi_df = qoi_df.droplevel(2)    # This will remove the level which is only needed for the second order effects.
            # ST
            qoi_st_df = qoi_df.loc[sobol_indices_name_list_in_order[0]]
            qoi_st_df = qoi_st_df.rename(columns={value_column_name:sobol_indices_name_list_in_order[0]})
            # S1
            qoi_s1_df = qoi_df.loc[sobol_indices_name_list_in_order[1]]
            qoi_s1_df = qoi_s1_df.rename(columns={value_column_name: sobol_indices_name_list_in_order[1]})
            # ST_S1
            st_s1_data = pd.concat([qoi_st_df[sobol_indices_name_list_in_order[0]], qoi_s1_df[sobol_indices_name_list_in_order[1]]], axis=1)
            st_s1_data = st_s1_data.loc[theta_name_list]
            # Sort from large to small using ST values only
            sorted_data = st_s1_data.reindex(st_s1_data.abs().sort_values(sobol_indices_name_list_in_order[0], ascending=False).index)
            # This list was already as input, but here it will be sorted differently
            sorted_theta_name_list = []
            for row in sorted_data.index:
                sorted_theta_name_list.append(row)
            # Iterate over sobol indices to plot them into the tornado
            for sobol_indices_i in range(len(sobol_indices_name_list_in_order)):
                sns.set_color_codes(sobol_inicies_colour_list[sobol_indices_i])
                # Sobol total index should be in the first position of the name list (ST) and then it should include the other indices in ascending order: S1, S2, etc.
                sns.barplot(data=sorted_data, x=sobol_indices_name_list_in_order[sobol_indices_i], y=sorted_theta_name_list, label=sobol_indices_name_list_in_order[sobol_indices_i], color='b')
            ax.set(ylabel="", xlabel=qoi_name)
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(14)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(14)
            # ax.set_xlim([0., 1.])
            if qoi_i == num_qois - 1:
                ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=True)
    plt.show()
    return fig


# TODO prepare for APD40 and APD50
def generate_repolarisation_map(vm):
    # assert vm.shape[1] == t.shape[0], "Evaluate RELMAP failed: vm.shape[1] == t.shape[0]"
    t = np.arange(0, vm.shape[1], 1)  # Assumes 1000 Hz

    # Uncomment to activate parallelisaion, but it cannot be nested into another parallel loop
    # relmap = pymp.shared.array((vm.shape[0]), dtype=np.float64) + get_nan_value()
    # threadsNum = multiprocessing.cpu_count()
    # with pymp.Parallel(min(threadsNum, vm.shape[0])) as p1:
    #     for i in p1.range(vm.shape[0]):
    relmap = np.zeros((vm.shape[0])) + get_nan_value()
    if True:
        for i in range(vm.shape[0]): # Loop through every node in mesh
            vm0 = vm[i][0] # Assume resting membrane potential at first time point.
            # assert vm0 < -80, "Resting membrane potential not found for node number "+str(i)
            maxamp = 0
            maxfound = 0
            for j in range(1, t.shape[0]): # Loop through time
                prev_v = vm[i][j-1]
                prev_time = t[j-1]
                # Find maximum AP amplitude for this node
                if (maxamp < vm[i][j]) & (maxfound == 0):
                    maxamp = vm[i][j]
                if (prev_v > vm[i][j]) & (vm[i][j] > 0):
                    maxfound = 1
                    maxamp = min(maxamp, 40)  #  This should do the trick so that the monodomain repol maps are aligned with the RE
                # Linearly interpolate time when AP amplitude reaches 90% of peak.
                if (maxfound == 1) & (relmap[i] == get_nan_value()):
                    ap90 = 0.1 * (maxamp - vm0) + vm0
                    if ((prev_v - ap90)*(vm[i][j]-ap90) < 0.0):
                        relmap[i] = (t[j]- prev_time)*abs((prev_v-ap90)/(prev_v-vm[i][j])) + prev_time # Fixed 2022/10/11
    if np.any(relmap == get_nan_value()):
        print('Warning when calling generate_repolarisation_map() Make sure the code is correct!')
        warn('Some Repol values ' + str(np.sum(relmap == get_nan_value())) + ' were not assigned and stayed as ' + str(get_nan_value()))
    return relmap


def generate_activation_map(vm, percentage):
    """vm has shape (nodes, time)
    percentage is the percent of APD of interest (e.g., percent=90 will give APD90)
    This function assumes 1000 Hz"""
    activation_map = np.ones((vm.shape[0]))
    vm_range = np.amax(vm, axis=1)-np.amin(vm, axis=1)
    vm_threshold = vm_range * (1.0 - percentage/100.0) + np.amin(vm, axis=1)
    for node_i in range(vm.shape[0]):  # Loop through every node in mesh
        local_vm = vm[node_i, :]
        index = np.nonzero(local_vm>vm_threshold[node_i])[0][0] #np.searchsorted(local_vm, vm_threshold[node_i])
        activation_map[node_i] = index
    return activation_map


def heatmap(x, y, **kwargs):
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors)

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs:
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs:
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order', 'xlabel', 'ylabel'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size],
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')

    ax.set_xlabel(kwargs.get('xlabel', ''))
    ax.set_ylabel(kwargs.get('ylabel', ''))

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right


def corrplot(data, size_scale=500, marker='s'):
    corr = pd.melt(data.reset_index(), id_vars='index').replace(np.nan, 0)
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        # x_order=data.columns,
        x_order=data.index,
        y_order=data.columns[::-1],
        # y_order=data.index[::-1],
        size_scale=size_scale
    )


def plot_histogram(data):
    fig = plt.figure()
    plt.hist(data)
    plt.show()
    return fig









class ECGPV_visualisation:
    def __init__(self, CL):
        self.CL = CL
        self.beat_fig_size = [5, 5]
        self.pv_fig_size = [6, 7]

    def read_ecg_pv(self, name, dir):
        meshname = dir + name
        print(meshname)
        if os.path.exists(dir + 'ecgs.pl'):
            ecgs = pickle.load(open(dir + 'ecgs.pl', 'rb'))
        else:
            ecgs = self._read_ECG(meshname)
            pickle.dump(ecgs, open(dir + 'ecgs.pl', 'wb'))
        if os.path.exists(dir + 'pvs.pl'):
            pvs = pickle.load(open(dir + 'pvs.pl', 'rb'))
        else:
            pvs = self._read_PV(meshname)
            pickle.dump(pvs, open(dir + 'pvs.pl', 'wb'))
        # Save as .mat file for delineation using Karlsruhe code:
        raw_leads = np.vstack(
            [ecgs['I'] / ecgs['max_all_leads'], ecgs['II'] / ecgs['max_all_leads'], ecgs['III'] / ecgs['max_all_leads'],
             ecgs['aVL'] / ecgs['max_all_leads'], ecgs['aVR'] / ecgs['max_all_leads'],
             ecgs['aVF'] / ecgs['max_all_leads'], ecgs['V1'] / ecgs['max_all_leads'],
             ecgs['V2'] / ecgs['max_all_leads'], ecgs['V3'] / ecgs['max_all_leads'], ecgs['V4'] / ecgs['max_all_leads'],
             ecgs['V5'] / ecgs['max_all_leads'], ecgs['V6'] / ecgs['max_all_leads']]).T

        raw_leads_resampled = np.zeros((int(ecgs['t'][-1] / 0.002), 12))
        # Down sample to 500 Hz
        for i in range(0, 12):
            nsample = int(ecgs['t'][-1] / 0.002)
            raw_leads_resampled[:, i] = resample(raw_leads[:, i], nsample)
        io.savemat('raw_ecg_leads.mat', {'signal': raw_leads_resampled})
        io.savemat('ecgs.mat', {'ecgs': ecgs})
        io.savemat('pvs.mat', {'pvs': pvs})
        return ecgs, pvs

    def _read_ECG(self, meshname):
        filename = meshname + '.exm.vin'
        with open(filename, 'r') as f:
            data = f.readlines()
        LA = RA = LL = RL = V1 = V2 = V3 = V4 = V5 = V6 = t = []

        # First 7 lines are header lines.
        if len(data) > 7:
            LA = np.zeros(len(data) - 7)
            RA = np.zeros(len(data) - 7)
            LL = np.zeros(len(data) - 7)
            RL = np.zeros(len(data) - 7)
            V1 = np.zeros(len(data) - 7)
            V2 = np.zeros(len(data) - 7)
            V3 = np.zeros(len(data) - 7)
            V4 = np.zeros(len(data) - 7)
            V5 = np.zeros(len(data) - 7)
            V6 = np.zeros(len(data) - 7)
            t = np.zeros(len(data) - 7)

        for i in range(7, len(data)):
            t[i - 7] = float(data[i].split()[-12])
            LA[i - 7] = float(data[i].split()[-10])
            RA[i - 7] = float(data[i].split()[-9])
            LL[i - 7] = float(data[i].split()[-8])
            RL[i - 7] = float(data[i].split()[-7])
            V1[i - 7] = float(data[i].split()[-6])
            V2[i - 7] = float(data[i].split()[-5])
            V3[i - 7] = float(data[i].split()[-4])
            V4[i - 7] = float(data[i].split()[-3])
            V5[i - 7] = float(data[i].split()[-2])
            V6[i - 7] = float(data[i].split()[-1])

        sort_i = np.argsort(t)
        t = t[sort_i]
        LA = LA[sort_i]
        RA = RA[sort_i]
        LL = LL[sort_i]
        RL = RL[sort_i]
        V1 = V1[sort_i]
        V2 = V2[sort_i]
        V3 = V3[sort_i]
        V4 = V4[sort_i]
        V5 = V5[sort_i]
        V6 = V6[sort_i]

        # Ealuate Wilson's central terminal
        VW = 1.0 / 3.0 * (RA + LA + LL)

        # Evaluate simulated ECG lead traces
        V1 = V1 - VW
        V2 = V2 - VW
        V3 = V3 - VW
        V4 = V4 - VW
        V5 = V5 - VW
        V6 = V6 - VW
        I = LA - RA
        II = LL - RA
        III = LL - LA
        aVL = LA - (RA + LL) / 2.0
        aVF = LL - (LA + RA) / 2.0
        aVR = RA - (LA + LL) / 2.0

        all_leads = np.concatenate((V1, V2, V3, V4, V5, V6, I, II, III, aVR, aVL, aVF))
        precord_leads = np.concatenate((V1, V2, V3, V4, V5, V6))
        limb_leads = np.concatenate((I, II, III, aVR, aVL, aVF))
        max_all_leads = max(abs(all_leads))
        max_precord_leads = max(abs(precord_leads))
        max_limb_leads = max(abs(limb_leads))

        # Divide into beats
        ts, V1s = self._divide_signal(t, V1, self.CL)
        ts, V2s = self._divide_signal(t, V2, self.CL)
        ts, V3s = self._divide_signal(t, V3, self.CL)
        ts, V4s = self._divide_signal(t, V4, self.CL)
        ts, V5s = self._divide_signal(t, V5, self.CL)
        ts, V6s = self._divide_signal(t, V6, self.CL)
        ts, aVRs = self._divide_signal(t, aVR, self.CL)
        ts, aVLs = self._divide_signal(t, aVL, self.CL)
        ts, aVFs = self._divide_signal(t, aVF, self.CL)
        ts, Is = self._divide_signal(t, I, self.CL)
        ts, IIs = self._divide_signal(t, II, self.CL)
        ts, IIIs = self._divide_signal(t, III, self.CL)

        output_dict = {'t': t, 'ts': ts, 'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5, 'V6': V6,
                       'aVR': aVR, 'aVL': aVL, 'aVF': aVF, 'I': I, 'II': II, 'III': III,
                       'V1s': V1s, 'V2s': V2s, 'V3s': V3s, 'V4s': V4s, 'V5s': V5s, 'V6s': V6s,
                       'aVRs': aVRs, 'aVLs': aVLs, 'aVFs': aVFs, 'Is': Is, 'IIs': IIs, 'IIIs': IIIs,
                       'max_all_leads': max_all_leads, 'max_precord_leads': max_precord_leads,
                       'max_limb_leads': max_limb_leads}
        return output_dict

    def _read_PV(self, meshname):
        filename = meshname + '-cardiac-cycle.sld.res'
        with open(filename, 'r') as f:
            data = f.readlines()
        if (len(data) > 18):
            pl = np.zeros(len(data) - 17)
            pr = np.zeros(len(data) - 17)
            vl = np.zeros(len(data) - 17)
            vr = np.zeros(len(data) - 17)
            curtime = np.zeros(len(data) - 17)
            phasel = np.zeros(len(data) - 17)
            phaser = np.zeros(len(data) - 17)
            vl[0] = float(data[18].split()[5])
            vr[0] = float(data[18].split()[-3])

            for i in range(18, len(data)):
                pl[i - 17] = float(data[i].split()[6]) / 10000
                pr[i - 17] = float(data[i].split()[-2]) / 10000
                vl[i - 17] = float(data[i].split()[5])
                vr[i - 17] = float(data[i].split()[-3])
                curtime[i - 17] = float(data[i].split()[1])
                phasel[i - 17] = float(data[i].split()[4])
                phaser[i - 17] = float(data[i].split()[10])

        sort_i = np.argsort(curtime)
        curtime = curtime[sort_i]
        pl = pl[sort_i]
        pr = pr[sort_i]
        vl = vl[sort_i]
        vr = vr[sort_i]
        phasel = phasel[sort_i]
        phaser = phaser[sort_i]

        # Divide into beats
        ts, pls = self._divide_signal(curtime, pl, self.CL)
        ts, vls = self._divide_signal(curtime, vl, self.CL)
        ts, prs = self._divide_signal(curtime, pr, self.CL)
        ts, vrs = self._divide_signal(curtime, vr, self.CL)
        ts, phasels = self._divide_signal(curtime, phasel, self.CL)
        ts, phasers = self._divide_signal(curtime, phaser, self.CL)
        output_dict = {'t': curtime, 'ts': ts, 'pl': pl, 'vl': vl, 'pr': pr, 'vr': vr, 'phasel': phasel,
                       'phaser': phaser,
                       'pls': pls, 'vls': vls, 'prs': prs, 'vrs': vrs, 'phasels': phasels, 'phasers': phasers,
                       'plabel': 'Pressure (kPa)', 'vlabel': 'Volume (mL)'}
        return output_dict

    def _divide_signal(self, curtime, signal, CL):
        if (curtime.max() > CL):
            t_offsets = []
            signals = []
            idx_start = []
            idx_end = []
            for i in range(0, int(curtime.max() / CL) + 1):
                idx_start.append(np.where(curtime >= i * CL)[0][0])
                if curtime.max() > (i + 1) * CL:
                    idx_end.append(np.where(curtime >= (i + 1) * CL)[0][0])
                else:
                    idx_end.append(len(curtime) - 1)
                t_offsets.append(curtime[idx_start[i]:idx_end[i]] - curtime[idx_start[i]])
                signals.append(signal[idx_start[i]:idx_end[i]])
        else:
            t_offsets = [curtime]
            signals = [signal]
        return t_offsets, signals

    def plot_ecgpv_live(self, ecgs, pvs, title, show, ecgs2=[], pvs2=[]):
        print('Plotting ECG PV live')
        matplotlib.rcParams.update({'font.size': '11'})
        matplotlib.rcParams.update({'text.color': 'black'})
        matplotlib.rcParams.update({'lines.linewidth': '1'})
        fig = plt.figure(tight_layout=True, figsize=[15, 7])
        fig.suptitle(title)
        gs = GridSpec(3, 6)
        axs = []
        axs.append(fig.add_subplot(gs[:, 0]))
        axs.append(fig.add_subplot(gs[0, 1]))
        axs.append(fig.add_subplot(gs[1, 1]))
        axs.append(fig.add_subplot(gs[2, 1]))
        axs.append(fig.add_subplot(gs[0, 2]))
        axs.append(fig.add_subplot(gs[1, 2]))
        axs.append(fig.add_subplot(gs[2, 2]))
        axs.append(fig.add_subplot(gs[0, 3]))
        axs.append(fig.add_subplot(gs[1, 3]))
        axs.append(fig.add_subplot(gs[2, 3]))
        axs.append(fig.add_subplot(gs[0, 4]))
        axs.append(fig.add_subplot(gs[1, 4]))
        axs.append(fig.add_subplot(gs[2, 4]))
        axs.append(fig.add_subplot(gs[0, 5]))
        axs.append(fig.add_subplot(gs[1, 5]))
        axs.append(fig.add_subplot(gs[2, 5]))

        def animate_single(i):
            # Plot PV
            axs[0].clear()
            axs[0].plot(pvs['vl'], pvs['pl'], pvs['vr'], pvs['pr'])
            axs[1].clear()
            axs[1].plot(pvs['t'], pvs['pl'], pvs['t'], pvs['pr'])
            axs[2].clear()
            axs[2].plot(pvs['t'], pvs['vl'], pvs['t'], pvs['vr'])
            axs[4].clear()
            axs[3].plot(pvs['t'], pvs['phasel'], pvs['t'], pvs['phaser'])

            # Plot ECGs:
            axs[4].clear()
            axs[4].plot(ecgs['t'], ecgs['I'] / ecgs['max_limb_leads'])
            axs[5].clear()
            axs[5].plot(ecgs['t'], ecgs['II'] / ecgs['max_limb_leads'])
            axs[6].clear()
            axs[6].plot(ecgs['t'], ecgs['III'] / ecgs['max_limb_leads'])
            axs[7].clear()
            axs[7].plot(ecgs['t'], ecgs['aVR'] / ecgs['max_limb_leads'])
            axs[8].clear()
            axs[8].plot(ecgs['t'], ecgs['aVL'] / ecgs['max_limb_leads'])
            axs[9].clear()
            axs[9].plot(ecgs['t'], ecgs['aVF'] / ecgs['max_limb_leads'])
            axs[10].clear()
            axs[10].plot(ecgs['t'], ecgs['V1'] / ecgs['max_precord_leads'])
            axs[11].clear()
            axs[11].plot(ecgs['t'], ecgs['V2'] / ecgs['max_precord_leads'])
            axs[12].clear()
            axs[12].plot(ecgs['t'], ecgs['V3'] / ecgs['max_precord_leads'])
            axs[13].clear()
            axs[13].plot(ecgs['t'], ecgs['V4'] / ecgs['max_precord_leads'])
            axs[14].clear()
            axs[14].plot(ecgs['t'], ecgs['V5'] / ecgs['max_precord_leads'])
            axs[15].clear()
            axs[15].plot(ecgs['t'], ecgs['V6'] / ecgs['max_precord_leads'])

            axs[0].set_xlabel('Volume (mL)')
            axs[1].set_xlabel('Time (ms)')
            axs[2].set_xlabel('Time (ms)')
            axs[3].set_xlabel('Time (ms)')
            axs[0].set_ylabel('Pressure (kPa)')
            axs[1].set_ylabel('Pressure (kPa)')
            axs[2].set_ylabel('Volume (mL)')
            axs[3].set_ylabel('Phase')
            axs[3].set_ylim([0, 5])
            axs[1].set_title('Pressure transient')
            axs[2].set_title('Volume transient')
            axs[3].set_title('Cardiac cycle')
            axs[4].set_xlabel('Time (s)')
            axs[5].set_xlabel('Time (s)')
            axs[6].set_xlabel('Time (s)')
            axs[7].set_xlabel('Time (s)')
            axs[8].set_xlabel('Time (s)')
            axs[9].set_xlabel('Time (s)')
            axs[10].set_xlabel('Time (s)')
            axs[11].set_xlabel('Time (s)')
            axs[12].set_xlabel('Time (s)')
            axs[13].set_xlabel('Time (s)')
            axs[14].set_xlabel('Time (s)')
            axs[15].set_xlabel('Time (s)')
            axs[4].set_ylabel('Normalised ECG')
            axs[5].set_ylabel('Normalised ECG')
            axs[6].set_ylabel('Normalised ECG')
            axs[7].set_ylabel('Normalised ECG')
            axs[8].set_ylabel('Normalised ECG')
            axs[9].set_ylabel('Normalised ECG')
            axs[10].set_ylabel('Normalised ECG')
            axs[11].set_ylabel('Normalised ECG')
            axs[12].set_ylabel('Normalised ECG')
            axs[13].set_ylabel('Normalised ECG')
            axs[14].set_ylabel('Normalised ECG')
            axs[15].set_ylabel('Normalised ECG')
            axs[4].set_ylim(-1, 1)
            axs[5].set_ylim(-1, 1)
            axs[6].set_ylim(-1, 1)
            axs[7].set_ylim(-1, 1)
            axs[8].set_ylim(-1, 1)
            axs[9].set_ylim(-1, 1)
            axs[10].set_ylim(-1, 1)
            axs[11].set_ylim(-1, 1)
            axs[12].set_ylim(-1, 1)
            axs[13].set_ylim(-1, 1)
            axs[14].set_ylim(-1, 1)
            axs[15].set_ylim(-1, 1)
            axs[4].set_title('I')
            axs[5].set_title('II')
            axs[6].set_title('III')
            axs[7].set_title('aVR')
            axs[8].set_title('aVL')
            axs[9].set_title('aVF')
            axs[10].set_title('V1')
            axs[11].set_title('V2')
            axs[12].set_title('V3')
            axs[13].set_title('V4')
            axs[14].set_title('V5')
            axs[15].set_title('V6')

        def animate_double(i):
            # Plot PV
            axs[0].clear()
            axs[0].plot(pvs['vl'], pvs['pl'], pvs['vr'], pvs['pr'], pvs2['vl'], pvs2['pl'], '--', pvs2['vr'],
                        pvs2['pr'], '--')
            axs[1].clear()
            axs[1].plot(pvs['t'], pvs['pl'], pvs['t'], pvs['pr'], pvs2['t'], pvs2['pl'], '--', pvs2['t'], pvs2['pr'],
                        '--')
            axs[2].clear()
            axs[2].plot(pvs['t'], pvs['vl'], pvs['t'], pvs['vr'], pvs2['t'], pvs2['vl'], '--', pvs2['t'], pvs2['vr'],
                        '--')
            axs[4].clear()
            axs[3].plot(pvs['t'], pvs['phasel'], pvs['t'], pvs['phaser'], pvs2['t'], pvs2['phasel'], '--', pvs2['t'],
                        pvs2['phaser'], '--')
            axs[0].set_xlabel('Volume (mL)')
            axs[1].set_xlabel('Time (ms)')
            axs[2].set_xlabel('Time (ms)')
            axs[3].set_xlabel('Time (ms)')
            axs[0].set_ylabel('Pressure (kPa)')
            axs[1].set_ylabel('Pressure (kPa)')
            axs[2].set_ylabel('Volume (mL)')
            axs[3].set_ylabel('Phase')
            axs[3].set_ylim([0, 5])

            axs[1].set_title('Pressure transient')
            axs[2].set_title('Volume transient')
            axs[3].set_title('Cardiac cycle')

            # Plot ECGs:
            max_all_leads = max([ecgs['max_all_leads'], ecgs2['max_all_leads']])
            max_precord_leads = max([ecgs['max_precord_leads'], ecgs2['max_precord_leads']])
            max_limb_leads = max([ecgs['max_limb_leads'], ecgs2['max_limb_leads']])
            axs[4].clear()
            axs[4].plot(ecgs['t'], ecgs['I'] / max_limb_leads, ecgs2['t'], ecgs2['I'] / max_limb_leads, '--')
            axs[5].clear()
            axs[5].plot(ecgs['t'], ecgs['II'] / max_limb_leads, ecgs2['t'], ecgs2['II'] / max_limb_leads, '--')
            axs[6].clear()
            axs[6].plot(ecgs['t'], ecgs['III'] / max_limb_leads, ecgs2['t'], ecgs2['III'] / max_limb_leads, '--')
            axs[7].clear()
            axs[7].plot(ecgs['t'], ecgs['aVR'] / max_limb_leads, ecgs2['t'], ecgs2['aVR'] / max_limb_leads, '--')
            axs[8].clear()
            axs[8].plot(ecgs['t'], ecgs['aVL'] / max_limb_leads, ecgs2['t'], ecgs2['aVL'] / max_limb_leads, '--')
            axs[9].clear()
            axs[9].plot(ecgs['t'], ecgs['aVF'] / max_limb_leads, ecgs2['t'], ecgs2['aVF'] / max_limb_leads, '--')
            axs[10].clear()
            axs[10].plot(ecgs['t'], ecgs['V1'] / max_precord_leads, ecgs2['t'], ecgs2['V1'] / max_precord_leads, '--')
            axs[11].clear()
            axs[11].plot(ecgs['t'], ecgs['V2'] / max_precord_leads, ecgs2['t'], ecgs2['V2'] / max_precord_leads, '--')
            axs[12].clear()
            axs[12].plot(ecgs['t'], ecgs['V3'] / max_precord_leads, ecgs2['t'], ecgs2['V3'] / max_precord_leads, '--')
            axs[13].clear()
            axs[13].plot(ecgs['t'], ecgs['V4'] / max_precord_leads, ecgs2['t'], ecgs2['V4'] / max_precord_leads, '--')
            axs[14].clear()
            axs[14].plot(ecgs['t'], ecgs['V5'] / max_precord_leads, ecgs2['t'], ecgs2['V5'] / max_precord_leads, '--')
            axs[15].clear()
            axs[15].plot(ecgs['t'], ecgs['V6'] / max_precord_leads, ecgs2['t'], ecgs2['V6'] / max_precord_leads, '--')
            axs[4].set_xlabel('Time (s)')
            axs[5].set_xlabel('Time (s)')
            axs[6].set_xlabel('Time (s)')
            axs[7].set_xlabel('Time (s)')
            axs[8].set_xlabel('Time (s)')
            axs[9].set_xlabel('Time (s)')
            axs[10].set_xlabel('Time (s)')
            axs[11].set_xlabel('Time (s)')
            axs[12].set_xlabel('Time (s)')
            axs[13].set_xlabel('Time (s)')
            axs[14].set_xlabel('Time (s)')
            axs[15].set_xlabel('Time (s)')
            axs[4].set_ylabel('Normalised ECG')
            axs[5].set_ylabel('Normalised ECG')
            axs[6].set_ylabel('Normalised ECG')
            axs[7].set_ylabel('Normalised ECG')
            axs[8].set_ylabel('Normalised ECG')
            axs[9].set_ylabel('Normalised ECG')
            axs[10].set_ylabel('Normalised ECG')
            axs[11].set_ylabel('Normalised ECG')
            axs[12].set_ylabel('Normalised ECG')
            axs[13].set_ylabel('Normalised ECG')
            axs[14].set_ylabel('Normalised ECG')
            axs[15].set_ylabel('Normalised ECG')
            axs[4].set_ylim(-1, 1)
            axs[5].set_ylim(-1, 1)
            axs[6].set_ylim(-1, 1)
            axs[7].set_ylim(-1, 1)
            axs[8].set_ylim(-1, 1)
            axs[9].set_ylim(-1, 1)
            axs[10].set_ylim(-1, 1)
            axs[11].set_ylim(-1, 1)
            axs[12].set_ylim(-1, 1)
            axs[13].set_ylim(-1, 1)
            axs[14].set_ylim(-1, 1)
            axs[15].set_ylim(-1, 1)
            axs[4].set_title('I')
            axs[5].set_title('II')
            axs[6].set_title('III')
            axs[7].set_title('aVR')
            axs[8].set_title('aVL')
            axs[9].set_title('aVF')
            axs[10].set_title('V1')
            axs[11].set_title('V2')
            axs[12].set_title('V3')
            axs[13].set_title('V4')
            axs[14].set_title('V5')
            axs[15].set_title('V6')

        # if ecgs2:
        #     ani = animation.FuncAnimation(fig, animate_double, fargs=( axs, ecgs, pvs, ecgs2, pvs2), interval=1000)
        # else:
        ani = animation.FuncAnimation(fig, animate_single, interval=1000)
        if show:
            plt.show(block=True)

    def _set_ecg_ticks(self, ax, t_end, CL):
        t_start = 0
        t_end = np.ceil(t_end / CL) * CL
        t_end = int(np.ceil(t_end * 10.0 / 2.0) * 2) / 10.0
        # t_end = np.round(t_end*10.0)/10.0
        # t_end = np.round(t_end)
        # t_end = self.CL
        minor_ticks = np.arange(0, t_end + 0.04, 0.04)
        major_ticks = np.arange(0, t_end + 0.2, 0.2)
        # minor_ticks = np.linspace(0, t_end, int((t_end-t_start)/0.04) + 1)
        # major_ticks = np.linspace(0, t_end, int((t_end-t_start)/0.2) + 1)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xticks(major_ticks)
        minor_ticks = np.linspace(-1, 1, 21)
        major_ticks = np.linspace(-1, 1, 5)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.grid(which="minor", color='r', linestyle='-', linewidth=1, alpha=0.5)
        ax.grid(which="major", color='r', linestyle='-', linewidth=2, alpha=0.5)

    def _set_full_ecg_ticks(self, ax, t_end, v_max, v_min, CL):
        t_start = 0
        t_end = np.ceil(t_end / CL) * CL
        t_end = int(np.ceil(t_end * 10.0 / 2.0) * 2) / 10.0
        minor_ticks = np.arange(0, t_end + 0.04, 0.04)
        major_ticks = np.arange(0, t_end + 0.2, 0.2)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xticks(major_ticks)
        v_max = np.ceil(v_max)
        minor_ticks = np.arange(v_min, v_max + 0.1, 0.1)
        major_ticks = np.arange(v_min, v_max + 0.5, 0.5)
        ax.set_yticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.grid(which="minor", color='r', linestyle='-', linewidth=1, alpha=0.3)
        ax.grid(which="major", color='r', linestyle='-', linewidth=2, alpha=0.3)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        for tic in ax.xaxis.get_major_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)
            tic.label1.set_visible(False)
            tic.label2.set_visible(False)
        for tic in ax.xaxis.get_minor_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)
            tic.label1.set_visible(False)
            tic.label2.set_visible(False)
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)
            tic.label1.set_visible(False)
            tic.label2.set_visible(False)
        for tic in ax.yaxis.get_minor_ticks():
            tic.tick1line.set_visible(False)
            tic.tick2line.set_visible(False)
            tic.label1.set_visible(False)
            tic.label2.set_visible(False)

    def _set_pv_ticks(self, ax):
        x_ticks = np.arange(50, 250, 50)
        y_ticks = np.arange(0, 140, 20)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        # ax.grid(which="major", color='k', linestyle='--', linewidth=1, alpha=0.5)

    def plot_ecg_lead(self, ecgs, lead_name, filename, show, beat=0, ecg2=[]):
        print('Plotting ECG lead ' + str(lead_name))
        matplotlib.rcParams.update({'font.size': '24'})
        matplotlib.rcParams.update({'text.color': 'black'})
        matplotlib.rcParams.update({'lines.linewidth': '3'})
        if ecg2:
            if beat > 0:
                max_all_leads = max([ecgs['max_all_leads'], ecgs2['max_all_leads']])
                self._plot_double(self.beat_fig_size,
                                  ecgs['ts'][beat - 1], ecgs[lead_name + 's'][beat - 1] / max_all_leads, '-',
                                  ecgs2['ts'][beat - 1], ecgs[lead_name + 's'][beat - 1] / max_all_leads, '--',
                                  'Time (s)', lead_name, filename + '_beat' + str(beat) + '_comparison.png',
                                  show, ecg_grid=True)
            else:
                figsize = [self.beat_fig_size[0] * (int(ecgs['t'][-1] / 1.0) + 1), self.beat_fig_size[1]]
                self._plot_double(figsize, ecgs['t'], ecgs[lead_name] / max_all_leads, '-',
                                  ecgs2['t'], ecgs2[lead_name] / max_all_leads, '--'
                                                                                'Time (s)', lead_name,
                                  filename + '_full_comparison.png', show, ecg_grid=True)
        else:
            if beat > 0:
                self._plot_single(self.beat_fig_size, ecgs['ts'][beat - 1],
                                  ecgs[lead_name + 's'][beat - 1] / ecgs['max_all_leads'],
                                  'Time (s)', lead_name, filename + '_beat' + str(beat) + '.png',
                                  show, ecg_grid=True)
            else:
                figsize = [self.beat_fig_size[0] * (int(ecgs['t'][-1] / 1.0) + 1), self.beat_fig_size[1]]
                self._plot_single(figsize, ecgs['t'], ecgs[lead_name] / ecgs['max_all_leads'],
                                  'Time (s)', lead_name, filename + '_full.png', show, ecg_grid=True)

    def plot_ecg_all_leads(self, ecgs, show, beat=0, ecgs2=[]):
        print('Plotting all ECG leads')
        matplotlib.rcParams.update({'font.size': '24'})
        matplotlib.rcParams.update({'text.color': 'black'})
        matplotlib.rcParams.update({'lines.linewidth': '1'})

        if ecgs2:
            print('Full ECG comparison not yet implemented...')
        else:
            if beat > 0:
                # t_padding
                t_padding = 0.2
                v_padding = 0.5

                # Generate calibrating step function
                t_res = ecgs['t'][1] - ecgs['t'][0]
                t_calib = np.arange(0, 0.2 + t_res, t_res)
                v_calib = np.zeros(np.shape(t_calib))
                for i in range(0, len(t_calib)):
                    if t_calib[i] < 0.04:
                        v_calib[i] = 0.0
                    elif t_calib[i] > t_calib[-1] - 0.04:
                        v_calib[i] = 0.0
                    else:
                        v_calib[i] = 1.0

                # Concatenate the leads together to plot in a single figure
                t_end = ecgs['ts'][beat - 1][-1]
                full_t = np.concatenate(
                    [t_calib, ecgs['ts'][beat - 1] + t_calib[-1], ecgs['ts'][beat - 1] + t_end + t_calib[-1],
                     ecgs['ts'][beat - 1] + 2 * t_end + t_calib[-1],
                     ecgs['ts'][beat - 1] + 3 * t_end + t_calib[-1]]) + t_padding
                scale = 1
                fig_size = [np.ceil(full_t[-1] + t_padding) * 0.5 / 0.2 * scale, 6 * scale]
                fig = plt.figure(tight_layout=True, figsize=fig_size)
                gs = GridSpec(1, 1)
                ax = fig.add_subplot(gs[0, 0])

                # Bottom row: III, aVF, V3, V6
                bottom_V = np.concatenate([v_calib, ecgs['IIIs'][beat - 1] / ecgs['max_limb_leads'],
                                              ecgs['aVFs'][beat - 1] / ecgs['max_limb_leads'],
                                              ecgs['V3s'][beat - 1] / ecgs['max_precord_leads'],
                                              ecgs['V6s'][beat - 1] / ecgs['max_precord_leads']])
                ax.plot(full_t, bottom_V, 'k')

                # Middle row: II, aVL, V2, V5
                offset = 2
                midrow_V = np.concatenate([v_calib + offset, ecgs['IIs'][beat - 1] / ecgs['max_limb_leads'] + offset,
                                              ecgs['aVLs'][beat - 1] / ecgs['max_limb_leads'] + offset,
                                              ecgs['V2s'][beat - 1] / ecgs['max_precord_leads'] + offset,
                                              ecgs['V5s'][beat - 1] / ecgs['max_precord_leads'] + offset])
                ax.plot(full_t, midrow_V, 'k')

                # Top row: I, aVR, V1, V4
                toprow_V = np.concatenate(
                    [v_calib + offset * 2, ecgs['Is'][beat - 1] / ecgs['max_limb_leads'] + offset * 2,
                     ecgs['aVRs'][beat - 1] / ecgs['max_limb_leads'] + offset * 2,
                     ecgs['V1s'][beat - 1] / ecgs['max_precord_leads'] + offset * 2,
                     ecgs['V4s'][beat - 1] / ecgs['max_precord_leads'] + offset * 2])
                ax.plot(full_t, toprow_V, 'k')

                # ax.set_xlim([0, t_end*4])
                # ax.set_ylim([-1, offset*2+1])
                #
                # self._set_full_ecg_ticks(ax, full_t[-1]+t_padding, offset*2+1, self.CL)

                # Add ECG red grid
                self._set_full_ecg_ticks(ax, full_t[-1] + t_padding * 2, offset * 2 + 1 + v_padding, -1 - v_padding,
                                         self.CL)
                ax.set_xlim([0, full_t[-1] + t_padding * 2])
                ax.set_ylim([-1 - v_padding, offset * 2 + 1 + v_padding])

                # Label the leads
                label_fontsize = 'large'
                label_t_offset = t_calib[-1] + t_padding
                label_v_offset = 0.1
                plt.text(0 + label_t_offset, -1 + label_v_offset, 'III', fontsize=label_fontsize)
                plt.text(t_end + label_t_offset, -1 + label_v_offset, 'aVF', fontsize=label_fontsize)
                plt.text(2 * t_end + label_t_offset, -1 + label_v_offset, 'V3', fontsize=label_fontsize)
                plt.text(3 * t_end + label_t_offset, -1 + label_v_offset, 'V6', fontsize=label_fontsize)
                plt.text(0 + label_t_offset, -1 + offset + label_v_offset, 'II', fontsize=label_fontsize)
                plt.text(t_end + label_t_offset, -1 + offset + label_v_offset, 'aVL', fontsize=label_fontsize)
                plt.text(2 * t_end + label_t_offset, -1 + offset + label_v_offset, 'V2', fontsize=label_fontsize)
                plt.text(3 * t_end + label_t_offset, -1 + offset + label_v_offset, 'V5', fontsize=label_fontsize)
                plt.text(0 + label_t_offset, -1 + offset * 2 + label_v_offset, 'I', fontsize=label_fontsize)
                plt.text(t_end + label_t_offset, -1 + offset * 2 + label_v_offset, 'aVR', fontsize=label_fontsize)
                plt.text(2 * t_end + label_t_offset, -1 + offset * 2 + label_v_offset, 'V1', fontsize=label_fontsize)
                plt.text(3 * t_end + label_t_offset, -1 + offset * 2 + label_v_offset, 'V4', fontsize=label_fontsize)
                plt.savefig('full_ecg_beat_' + str(beat) + '.png')
                if show:
                    plt.show()
            else:
                # Padding
                t_padding = 0.2
                v_padding = 0.5

                # Generate calibrating step function
                t_res = ecgs['t'][1] - ecgs['t'][0]
                t_calib = np.arange(0, 0.2 + t_res, t_res)
                v_calib = np.zeros(np.shape(t_calib))
                for i in range(0, len(t_calib)):
                    if t_calib[i] < 0.04:
                        v_calib[i] = 0.0
                    elif t_calib[i] > t_calib[-1] - 0.04:
                        v_calib[i] = 0.0
                    else:
                        v_calib[i] = 1.0

                # Concatenate the leads together to plot in a single figure
                t_end = ecgs['t'][-1]
                full_t = np.concatenate([t_calib, ecgs['t'] + t_calib[-1], ecgs['t'] + t_end + t_calib[-1],
                                            ecgs['t'] + 2 * t_end + t_calib[-1],
                                            ecgs['t'] + 3 * t_end + t_calib[-1]]) + t_padding
                scale = 0.8
                fig_size = [np.ceil(full_t[-1] + t_padding) * 0.5 / 0.2 * scale, 6 * scale]
                fig = plt.figure(tight_layout=True, figsize=fig_size)
                gs = GridSpec(1, 1)
                ax = fig.add_subplot(gs[0, 0])
                # Bottom row: III, aVF, V3, V6
                bottom_V = np.concatenate(
                    [v_calib, ecgs['III'] / ecgs['max_limb_leads'], ecgs['aVF'] / ecgs['max_limb_leads'],
                     ecgs['V3'] / ecgs['max_precord_leads'], ecgs['V6'] / ecgs['max_precord_leads']])
                ax.plot(full_t, bottom_V, 'k')

                # Middle row: II, aVL, V2, V5
                offset = 2
                midrow_V = np.concatenate([v_calib + offset, ecgs['II'] / ecgs['max_limb_leads'] + offset,
                                              ecgs['aVL'] / ecgs['max_limb_leads'] + offset,
                                              ecgs['V2'] / ecgs['max_precord_leads'] + offset,
                                              ecgs['V5'] / ecgs['max_precord_leads'] + offset])
                ax.plot(full_t, midrow_V, 'k')

                # Top row: I, aVR, V1, V4
                toprow_V = np.concatenate([v_calib + offset * 2, ecgs['I'] / ecgs['max_limb_leads'] + offset * 2,
                                              ecgs['aVR'] / ecgs['max_limb_leads'] + offset * 2,
                                              ecgs['V1'] / ecgs['max_precord_leads'] + offset * 2,
                                              ecgs['V4'] / ecgs['max_precord_leads'] + offset * 2])
                ax.plot(full_t, toprow_V, 'k')

                # Add ECG red grid
                self._set_full_ecg_ticks(ax, full_t[-1] + t_padding * 2, offset * 2 + 1 + v_padding, -1 - v_padding,
                                         self.CL)
                ax.set_xlim([0, full_t[-1] + t_padding * 2])
                ax.set_ylim([-1 - v_padding, offset * 2 + 1 + v_padding])

                # Label the leads
                label_fontsize = 'xx-small'
                label_t_offset = t_calib[-1] + t_padding
                label_v_offset = 0.1
                plt.text(0 + label_t_offset, -1 + label_v_offset, 'III', fontsize=label_fontsize)
                plt.text(t_end + label_t_offset, -1 + label_v_offset, 'aVF', fontsize=label_fontsize)
                plt.text(2 * t_end + label_t_offset, -1 + label_v_offset, 'V3', fontsize=label_fontsize)
                plt.text(3 * t_end + label_t_offset, -1 + label_v_offset, 'V6', fontsize=label_fontsize)
                plt.text(0 + label_t_offset, -1 + offset + label_v_offset, 'II', fontsize=label_fontsize)
                plt.text(t_end + label_t_offset, -1 + offset + label_v_offset, 'aVL', fontsize=label_fontsize)
                plt.text(2 * t_end + label_t_offset, -1 + offset + label_v_offset, 'V2', fontsize=label_fontsize)
                plt.text(3 * t_end + label_t_offset, -1 + offset + label_v_offset, 'V5', fontsize=label_fontsize)
                plt.text(0 + label_t_offset, -1 + offset * 2 + label_v_offset, 'I', fontsize=label_fontsize)
                plt.text(t_end + label_t_offset, -1 + offset * 2 + label_v_offset, 'aVR', fontsize=label_fontsize)
                plt.text(2 * t_end + label_t_offset, -1 + offset * 2 + label_v_offset, 'V1', fontsize=label_fontsize)
                plt.text(3 * t_end + label_t_offset, -1 + offset * 2 + label_v_offset, 'V4', fontsize=label_fontsize)
                plt.savefig('full_ecg_all_beats.png')
                if show:
                    plt.show()

    def plot_pv_signal(self, pvs, signal_name, filename, show, beat=0, pvs2=[]):
        matplotlib.rcParams.update({'font.size': '28'})
        matplotlib.rcParams.update({'text.color': 'black'})
        matplotlib.rcParams.update({'lines.linewidth': '3'})
        if pvs2:
            if beat > 0:
                if (signal_name == 'p') | (signal_name == 'v'):
                    self._plot_quadruple(self.pv_fig_size,
                                         pvs['ts'][beat - 1], pvs[signal_name + 'ls'][beat - 1], 'b--',
                                         pvs['ts'][beat - 1], pvs[signal_name + 'rs'][beat - 1], 'g--',
                                         pvs2['ts'][beat - 1], pvs2[signal_name + 'ls'][beat - 1], 'b-',
                                         pvs2['ts'][beat - 1], pvs2[signal_name + 'rs'][beat - 1], 'g-',
                                         'Time (s)', pvs[signal_name + 'label'],
                                         filename + '_beat' + str(beat) + '_compare.png', show)
                elif (signal_name == 'pv'):
                    self._plot_quadruple(self.pv_fig_size,
                                         pvs['vls'][beat - 1], pvs['pls'][beat - 1] * 7.5, 'b--',
                                         pvs['vrs'][beat - 1], pvs['prs'][beat - 1] * 7.5, 'g--',
                                         pvs2['vls'][beat - 1], pvs2['pls'][beat - 1] * 7.5, 'b-',
                                         pvs2['vrs'][beat - 1], pvs2['prs'][beat - 1] * 7.5, 'g-',
                                         'Volume (mL)', 'Pressure (mmHg)',
                                         filename + '_beat' + str(beat) + '_compare.png', show)
            else:
                if (signal_name == 'p') | (signal_name == 'v'):
                    figsize = [self.pv_fig_size[0] * (int(pvs['t'][-1] / 1.0) + 1), self.pv_fig_size[1]]
                    self._plot_quadruple(figsize,
                                         pvs['t'], pvs[signal_name + 'l'], 'b--',
                                         pvs['t'], pvs[signal_name + 'r'], 'g--',
                                         pvs2['t'], pvs2[signal_name + 'l'], 'b-',
                                         pvs2['t'], pvs2[signal_name + 'r'], 'g-', 'Time (s)',
                                         pvs[signal_name + 'label'], filename + '_full_compare.png', show)
                elif (signal_name == 'pv'):
                    self._plot_quadruple(self.pv_fig_size,
                                         pvs['vl'], pvs['pl'] * 7.5, 'b--',
                                         pvs['vr'], pvs['pr'] * 7.5, 'g--',
                                         pvs2['vl'], pvs2['pl'] * 7.5, 'b-',
                                         pvs2['vr'], pvs2['pr'] * 7.5, 'g-',
                                         'Volume (mL)', 'Pressure (mmHg)', filename + '_full_compare.png', show)
        else:
            if beat > 0:
                if (signal_name == 'p') | (signal_name == 'v'):
                    self._plot_double(self.pv_fig_size,
                                      pvs['ts'][beat - 1], pvs[signal_name + 'ls'][beat - 1], 'b-',
                                      pvs['ts'][beat - 1], pvs[signal_name + 'rs'][beat - 1], 'g-',
                                      'Time (s)', pvs[signal_name + 'label'], filename + '_beat' + str(beat) + '.png',
                                      show)
                elif (signal_name == 'pv'):
                    self._plot_double(self.pv_fig_size,
                                      pvs['vls'][beat - 1], pvs['pls'][beat - 1], 'b-',
                                      pvs['vrs'][beat - 1], pvs['prs'][beat - 1], 'g-',
                                      'Volume (mL)', 'Pressure (kPa)',
                                      filename + '_beat' + str(beat) + '.png', show)
            else:
                if (signal_name == 'p') | (signal_name == 'v'):
                    figsize = [self.pv_fig_size[0] * (int(pvs['t'][-1] / 1.0) + 1), self.pv_fig_size[1]]
                    self._plot_double(figsize, pvs['t'], pvs[signal_name + 'l'], 'b-',
                                      pvs['t'], pvs[signal_name + 'r'], 'g-',
                                      'Time (s)', pvs[signal_name + 'label'], filename + '_full.png', show)
                elif (signal_name == 'pv'):
                    self._plot_double(self.pv_fig_size, pvs['vl'], pvs['pl'], 'b-',
                                      pvs['vr'], pvs['pr'], 'r-',
                                      'Volume (mL)', 'Pressure (kPa)', filename + '_full.png', show)

    def _plot_single(self, fig_size, x, y, xlabel, ylabel, filename, show, ecg_grid=False):
        fig = plt.figure(tight_layout=True, figsize=fig_size)
        gs = GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ecg_grid:
            print('setting ecg grids')
            self._set_ecg_ticks(ax, x[-1], self.CL)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False, right=False, top=False, bottom=False, labelleft=False, labelbottom=False)
            ax.set_xlim([0, np.ceil(x[-1] / self.CL) * CL])
        if show:
            plt.show(block=True)
        plt.savefig(filename)

    def _plot_double(self, fig_size, x, y, linestyle, x1, y1, linestyle1, xlabel, ylabel, filename, show,
                     ecg_grid=False):
        fig = plt.figure(tight_layout=True, figsize=fig_size)
        gs = GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(x, y, linestyle, x1, y1, linestyle1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ecg_grid:
            self._set_ecg_ticks(ax, x[-1], self.CL)
        if show:
            plt.show(block=True)
        plt.savefig(filename)

    def _plot_quadruple(self, fig_size, x, y, linestyle, x1, y1, linestyle1, x2, y2, linestyle2, x3, y3, linestyle3,
                        xlabel, ylabel, filename, show, ecg_grid=False):
        fig = plt.figure(tight_layout=True, figsize=fig_size)
        gs = GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(x, y, linestyle, x1, y1, linestyle1, x2, y2, linestyle2, x3, y3, linestyle3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim([0, 120])
        self._set_pv_ticks(ax)
        if ecg_grid:
            self._set_ecg_ticks(ax, x[-1], self.CL)
        if show:
            plt.show(block=True)
        plt.savefig(filename)

    def read_single_cell(self, dir, material):
        # Read membrane potentials
        filename = dir + 'torord_ode.m' + str(material) + 'c3.csv'
        with open(filename, 'r') as f:
            data = f.readlines()
        data = data[1:]
        epi = np.zeros((len(data), 5))
        for i in range(0, len(epi)): epi[i, 1] = float(data[i].split()[2]) * 1000000.0  # Calcium transient
        for i in range(0, len(epi)): epi[i, 2] = float(data[i].split()[3])  # Membrane potential
        for i in range(0, len(epi)): epi[i, 0] = float(data[i].split()[0]) / 1000.0  # real time
        for i in range(0, len(epi)): epi[i, 3] = float(data[i].split()[1]) / 1000.0  # each beat time
        for i in range(0, len(epi)): epi[i, 4] = float(data[i].split()[4])  # Ta

        filename = dir + 'torord_ode.m' + str(material) + 'c2.csv'
        with open(filename, 'r') as f:
            data = f.readlines()
        data = data[1:]
        mid = np.zeros((len(data), 5))
        for i in range(0, len(mid)): mid[i, 1] = float(data[i].split()[2]) * 1000000.0  # Calcium transient
        for i in range(0, len(mid)): mid[i, 2] = float(data[i].split()[3])  # Membrane potential
        for i in range(0, len(mid)): mid[i, 0] = float(data[i].split()[0]) / 1000.0  # real time
        for i in range(0, len(mid)): mid[i, 3] = float(data[i].split()[1]) / 1000.0  # each beat time
        for i in range(0, len(mid)): mid[i, 4] = float(data[i].split()[4])  # Ta

        filename = dir + 'torord_ode.m' + str(material) + 'c1.csv'
        with open(filename, 'r') as f:
            data = f.readlines()
        data = data[1:]
        endo = np.zeros((len(data), 5))
        for i in range(0, len(endo)): endo[i, 1] = float(data[i].split()[2]) * 1000000.0  # Calcium transient
        for i in range(0, len(endo)): endo[i, 2] = float(data[i].split()[3])  # Membrane potential
        for i in range(0, len(endo)): endo[i, 0] = float(data[i].split()[0]) / 1000.0  # real time
        for i in range(0, len(endo)): endo[i, 3] = float(data[i].split()[1]) / 1000.0  # each beat time
        for i in range(0, len(endo)): endo[i, 4] = float(data[i].split()[4])  # Ta
        return epi, mid, endo

    def plot_single_cell(self, epi, mid, endo, material, analysis, show, epi2=[], mid2=[], endo2=[]):
        # Set up subplots
        fig = plt.figure(tight_layout=True, figsize=[15, 5])
        gs = GridSpec(3, 7)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[:, 1:3])
        ax5 = fig.add_subplot(gs[:, 3:5])
        ax6 = fig.add_subplot(gs[:, 5:7])

        if len(epi2) > 0:
            ax1.plot(epi[:, 0], epi[:, 2], mid[:, 0], mid[:, 2], endo[:, 0], endo[:, 2],
                     epi2[:, 0], epi2[:, 2], '--', mid2[:, 0], mid2[:, 2], '--', endo2[:, 0], endo2[:, 2], '--')
            ax2.plot(epi[:, 0], epi[:, 1], mid[:, 0], mid[:, 1], endo[:, 0], endo[:, 1],
                     epi2[:, 0], epi2[:, 1], '--', mid2[:, 0], mid2[:, 1], '--', endo2[:, 0], endo2[:, 1], '--')
            ax3.plot(epi[:, 0], epi[:, 4], mid[:, 0], mid[:, 4], endo[:, 0], endo[:, 4],
                     epi2[:, 0], epi2[:, 4], '--', mid2[:, 0], mid2[:, 4], '--', endo2[:, 0], endo2[:, 4], '--')
            dt = epi[1, 0] - epi[0, 0]
            idx = int(self.CL / dt)
            t_s_epi = epi[-idx, 0]
            t_s_mid = mid[-idx, 0]
            t_s_endo = endo[-idx, 0]
            t_s_epi2 = epi2[-idx, 0]
            t_s_mid2 = mid2[-idx, 0]
            t_s_endo2 = endo2[-idx, 0]
            ax4.plot(epi[-idx:, 0] - t_s_epi, epi[-idx:, 2], mid[-idx:, 0] - t_s_mid, mid[-idx:, 2],
                     endo[-idx:, 0] - t_s_endo, endo[-idx:, 2],
                     epi2[-idx:, 0] - t_s_epi2, epi2[-idx:, 2], '--', mid2[-idx:, 0] - t_s_mid2, mid2[-idx:, 2], '--',
                     endo2[-idx:, 0] - t_s_endo2, endo2[-idx:, 2], '--')
            ax5.plot(epi[-idx:, 0] - t_s_epi, epi[-idx:, 1], mid[-idx:, 0] - t_s_mid, mid[-idx:, 1],
                     endo[-idx:, 0] - t_s_endo, endo[-idx:, 1],
                     epi2[-idx:, 0] - t_s_epi2, epi2[-idx:, 1], '--', mid2[-idx:, 0] - t_s_mid2, mid2[-idx:, 1], '--',
                     endo2[-idx:, 0] - t_s_endo2, endo2[-idx:, 1], '--')
            ax6.plot(epi[-idx:, 0] - t_s_epi, epi[-idx:, 4], mid[-idx:, 0] - t_s_mid, mid[-idx:, 4],
                     endo[-idx:, 0] - t_s_endo, endo[-idx:, 4],
                     epi2[-idx:, 0] - t_s_epi2, epi2[-idx:, 4], '--', mid2[-idx:, 0] - t_s_mid2, mid2[-idx:, 4], '--',
                     endo2[-idx:, 0] - t_s_endo2, endo2[-idx:, 4], '--')
            if analysis:
                print('1: Diastolic calcium (nM) epi: ' + str(epi[-1, 1]) + ', mid: ' + str(
                    mid[-1, 1]) + ', endo: ' + str(endo[-1, 1]))
                print('1: Calcium amplitude (nM) epi: ' + str(max(epi[-idx:, 1])) + ', mid: ' + str(
                    max(mid[-idx:, 1])) + ', endo: ' + str(max(endo[-idx:, 1])))
                print('1: Diastolic active tension (kPa) epi: ' + str(epi[-1, 4]) + ', mid: ' + str(
                    mid[-1, 4]) + ', endo: ' + str(endo[-1, 4]))
                print('1: Active tension amplitude (kPa) epi: ' + str(max(epi[-idx:, 4])) + ', mid: ' + str(
                    max(mid[-idx:, 4])) + ', endo: ' + str(max(endo[-idx:, 4])))
                print('2: Diastolic calcium (nM) epi: ' + str(epi2[-1, 1]) + ', mid: ' + str(
                    mid2[-1, 1]) + ', endo: ' + str(endo2[-1, 1]))
                print('2: Calcium amplitude (nM) epi: ' + str(max(epi2[-idx:, 1])) + ', mid: ' + str(
                    max(mid2[-idx:, 1])) + ', endo: ' + str(max(endo2[-idx:, 1])))
                print('2: Diastolic active tension (kPa) epi: ' + str(epi2[-1, 4]) + ', mid: ' + str(
                    mid2[-1, 4]) + ', endo: ' + str(endo2[-1, 4]))
                print('2: Active tension amplitude (kPa) epi: ' + str(max(epi2[-idx:, 4])) + ', mid: ' + str(
                    max(mid2[-idx:, 4])) + ', endo: ' + str(max(endo2[-idx:, 4])))
        else:
            ax1.plot(epi[:, 0], epi[:, 2], mid[:, 0], mid[:, 2], endo[:, 0], endo[:, 2])
            ax2.plot(epi[:, 0], epi[:, 1], mid[:, 0], mid[:, 1], endo[:, 0], endo[:, 1])
            ax3.plot(epi[:, 0], epi[:, 4], mid[:, 0], mid[:, 4], endo[:, 0], endo[:, 4])
            dt = epi[1, 0] - epi[0, 0]
            idx = int(self.CL / dt)
            t_s_epi = epi[-idx, 0]
            t_s_mid = mid[-idx, 0]
            t_s_endo = endo[-idx, 0]
            ax4.plot(epi[-idx:, 0] - t_s_epi, epi[-idx:, 2], mid[-idx:, 0] - t_s_mid, mid[-idx:, 2],
                     endo[-idx:, 0] - t_s_endo, endo[-idx:, 2])
            ax5.plot(epi[-idx:, 0] - t_s_epi, epi[-idx:, 1], mid[-idx:, 0] - t_s_mid, mid[-idx:, 1],
                     endo[-idx:, 0] - t_s_endo, endo[-idx:, 1])
            ax6.plot(epi[-idx:, 0] - t_s_epi, epi[-idx:, 4], mid[-idx:, 0] - t_s_mid, mid[-idx:, 4],
                     endo[-idx:, 0] - t_s_endo, endo[-idx:, 4])
            if analysis:
                print('Diastolic calcium (nM) epi: {:.2f}, mid: {:.2f}, endo: {:.2f}'.format(epi[-1, 1], mid[-1, 1],
                                                                                             endo[-1, 1]))
                print('Calcium amplitude (nM) epi: {:.2f}, mid: {:.2f}, endo: {:.2f}'.format(max(epi[-idx:, 1]),
                                                                                             max(mid[-idx:, 1]),
                                                                                             max(endo[-idx:, 1])))
                print('Diastolic active tension (kPa) epi: {:.2f}, mid: {:.2f}, endo: {:.2f}'.format(epi[-1, 4],
                                                                                                     mid[-1, 4],
                                                                                                     endo[-1, 4]))
                print('Active tension amplitude (kPa) epi: {:.2f}, mid: {:.2f}, endo: {:.2f}'.format(max(epi[-idx:, 4]),
                                                                                                     max(mid[-idx:, 4]),
                                                                                                     max(endo[-idx:,
                                                                                                         4])))

        ax1.set_ylabel('(mV)')
        ax1.set_xlabel('Time (s)')
        # ax1.set_title('Action potentials')
        ax1.legend(['Epi', 'Mid', 'Endo'])

        # ax2.set_ylabel('Intracellular Calcium (nM)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('CaT (nM)')

        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Active tension (kPa)')

        dt = epi[1, 0] - epi[0, 0]
        idx = int(0.8 / dt)

        ax4.set_ylabel('Vm (mV)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylim([-100, 40])
        ax4.grid()
        ax4.set_xticks(np.arange(0, self.CL + 0.2, 0.2))

        ax5.set_ylabel('CaT (nM)')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylim([0, 1000])
        ax5.grid()
        ax5.set_xticks(np.arange(0, self.CL + 0.2, 0.2))

        ax6.set_ylabel('Active tension (kPa)')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylim([0, 70])
        ax6.grid()
        ax6.set_xticks(np.arange(0, self.CL + 0.2, 0.2))

        if show:
            plt.show(block=True)
        plt.savefig('single_cell_' + str(material) + '.png')

    def analysis_PV(self, pvs, beat=0):
        if beat > 0:
            EDVL = max(pvs['vls'][beat - 1])
            EDVR = max(pvs['vrs'][beat - 1])
            ESVL = min(pvs['vls'][beat - 1])
            ESVR = min(pvs['vrs'][beat - 1])
            PmaxL = int(max(pvs['pls'][beat - 1]))
            PmaxR = int(max(pvs['prs'][beat - 1]))
            LVEF = int((EDVL - ESVL) / EDVL * 100)
            RVEF = int((EDVR - ESVR) / EDVR * 100)
            SVL = int(EDVL - ESVL)
            SVR = int(EDVR - ESVR)

        else:
            LVEF = []
            RVEF = []
            EDVL = []
            EDVR = []
            ESVL = []
            ESVR = []
            PmaxL = []
            PmaxR = []
            SVL = []
            SVR = []
            for i in range(0, len(pvs['vls'])):
                EDVL.append(max(pvs['vls'][i]))
                ESVL.append(min(pvs['vls'][i]))
                LVEF.append(int((EDVL[i] - ESVL[i]) / EDVL[i] * 100))
                PmaxL.append(int(max(pvs['pls'][i])))
                SVL.append(int(EDVL[i] - ESVL[i]))
                EDVR.append(max(pvs['vrs'][i]))
                ESVR.append(min(pvs['vrs'][i]))
                RVEF.append(int((EDVR[i] - ESVR[i]) / EDVR[i] * 100))
                PmaxR.append(int(max(pvs['prs'][i])))
                SVR.append(int(EDVR[i] - ESVR[i]))
        print('EDVL:' + str(int(EDVL)) + ' mL,\tEDVR: ' + str(int(EDVR)) + ' mL')
        print('ESVL: ' + str(int(ESVL)) + ' mL,\tESVR: ' + str(int(ESVR)) + ' mL')
        print('LVEF: ' + str(LVEF) + ' %,\tRVEF: ' + str(RVEF) + ' %')
        print('PmaxL: ' + str(PmaxL) + ' kPa,\tPmaxR: ' + str(PmaxR) + ' kPa')
        print('SVL: ' + str(SVL) + ' mL,\tSVR: ' + str(SVR) + ' mL')

    def analysis_ECG(self, ecgs, beat, show):
        fig = plt.figure(tight_layout=True, figsize=[11, 9])
        gs = GridSpec(3, 4)
        ax5 = fig.add_subplot(gs[0, 0])
        ax6 = fig.add_subplot(gs[1, 0])
        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[0, 1])
        ax9 = fig.add_subplot(gs[1, 1])
        ax10 = fig.add_subplot(gs[2, 1])
        ax11 = fig.add_subplot(gs[0, 2])
        ax12 = fig.add_subplot(gs[1, 2])
        ax13 = fig.add_subplot(gs[2, 2])
        ax14 = fig.add_subplot(gs[0, 3])
        ax15 = fig.add_subplot(gs[1, 3])
        ax16 = fig.add_subplot(gs[2, 3])

        ecg_biomarkers = np.zeros((12, 6))
        ecg_biomarkers[0, :] = self._plot_with_landmarks(ax5, ecgs['ts'][beat - 1], ecgs['max_all_leads'],
                                                         ecgs['Is'][beat - 1], 'I', CL)
        ecg_biomarkers[1, :] = self._plot_with_landmarks(ax6, ecgs['ts'][beat - 1], ecgs['max_all_leads'],
                                                         ecgs['IIs'][beat - 1], 'II', CL)
        ecg_biomarkers[2, :] = self._plot_with_landmarks(ax7, ecgs['ts'][beat - 1], ecgs['max_all_leads'],
                                                         ecgs['IIIs'][beat - 1], 'III', CL)
        ecg_biomarkers[3, :] = self._plot_with_landmarks(ax8, ecgs['ts'][beat - 1], ecgs['max_all_leads'],
                                                         ecgs['aVRs'][beat - 1], 'aVR', CL)
        ecg_biomarkers[4, :] = self._plot_with_landmarks(ax9, ecgs['ts'][beat - 1], ecgs['max_all_leads'],
                                                         ecgs['aVLs'][beat - 1], 'aVL', CL)
        ecg_biomarkers[5, :] = self._plot_with_landmarks(ax10, ecgs['ts'][beat - 1], ecgs['max_all_leads'],
                                                         ecgs['aVFs'][beat - 1], 'aVF', CL)
        ecg_biomarkers[6, :] = self._plot_with_landmarks(ax11, ecgs['ts'][beat - 1], ecgs['max_all_leads'],
                                                         ecgs['V1s'][beat - 1], 'V1', CL)
        ecg_biomarkers[7, :] = self._plot_with_landmarks(ax12, ecgs['ts'][beat - 1], ecgs['max_all_leads'],
                                                         ecgs['V2s'][beat - 1], 'V2', CL)
        ecg_biomarkers[8, :] = self._plot_with_landmarks(ax13, ecgs['ts'][beat - 1], ecgs['max_all_leads'],
                                                         ecgs['V3s'][beat - 1], 'V3', CL)
        ecg_biomarkers[9, :] = self._plot_with_landmarks(ax14, ecgs['ts'][beat - 1], ecgs['max_all_leads'],
                                                         ecgs['V4s'][beat - 1], 'V4', CL)
        ecg_biomarkers[10, :] = self._plot_with_landmarks(ax15, ecgs['ts'][beat - 1], ecgs['max_all_leads'],
                                                          ecgs['V5s'][beat - 1], 'V5', CL)
        ecg_biomarkers[11, :] = self._plot_with_landmarks(ax16, ecgs['ts'][beat - 1], ecgs['max_all_leads'],
                                                          ecgs['V6s'][beat - 1], 'V6', CL)

        if show:
            plt.show()
        plt.savefig('ecg_analysis.png')
        print('QRS dur, T dur, T pe, T op, T amp, QT dur')
        print(ecg_biomarkers)

        QRS_mean = np.average(ecg_biomarkers[:, 0])
        QRS_std = np.std(ecg_biomarkers[:, 0])
        T_dur_mean = np.average(ecg_biomarkers[:, 1])
        T_dur_std = np.std(ecg_biomarkers[:, 1])
        T_pe_mean = np.average(ecg_biomarkers[:, 2])
        T_pe_std = np.std(ecg_biomarkers[:, 2])
        T_op_mean = np.average(ecg_biomarkers[:, 3])
        T_op_std = np.std(ecg_biomarkers[:, 3])
        QT_dur_mean = np.average(ecg_biomarkers[:, 5])
        QT_dur_std = np.std(ecg_biomarkers[:, 5])
        QT_dispersion_6 = ecg_biomarkers[6:12, 5].max() - ecg_biomarkers[6:12, 5].min()
        QT_dispersion_12 = ecg_biomarkers[:, 5].max() - ecg_biomarkers[:, 5].min()
        print('QRS: {:.3f} +- {:.3f}'.format(QRS_mean, QRS_std))
        print('T duration: {:.3f} +- {:.3f}'.format(T_dur_mean, T_dur_std))
        print('T pe: {:.3f} +- {:.3f}'.format(T_pe_mean, T_pe_std))
        print('T op: {:.3f} +- {:.3f}'.format(T_op_mean, T_op_std))
        print('QT duration: {:.3f} +- {:.3f}'.format(QT_dur_mean, QT_dur_std))
        print('QT dispersion (precordial): {:.3f}'.format(QT_dispersion_6))
        print('QT dispersion (12 leads): {:.3f}'.format(QT_dispersion_12))

    def _plot_with_landmarks(self, ax, t, max_all_leads, V, name, CL):
        ax.clear()
        qrs_dur, t_dur, t_amp, t_ep, t_op, qt_dur, landmarks = self._measurements(V, 0.0, CL, t, 2e-5, 0.02)
        ax.plot(t, V / max_all_leads, landmarks[:, 0], landmarks[:, 1] / max_all_leads, '*')
        ax.set_title(name + ' ' + str(int(qrs_dur * 1000)) + ' ' + str(int(qt_dur * 1000)) + '\n' + str(
            int(t_dur * 1000)) + ' ' + str(int(t_amp / max_all_leads)))
        ax.set_xlim(0.0, CL)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalised ECG')
        return qrs_dur, t_dur, t_amp, t_ep, t_op, qt_dur

    def _measurements(self, V, start_t, end_t, t, t_tol, v_tol):
        idx_start = np.where(abs(np.array(t - start_t)) < t_tol)[0][0]
        try:
            idx_end = np.where(abs(np.array(t - end_t)) < t_tol)[0][0]
        except:
            idx_end = len(V) - 1
        # Offset voltage using t_end
        n = 100
        b = [1.0 / n] * n
        V = V - V[idx_end]
        V = lfilter(b, 1, V)
        dV = np.gradient(V)
        dV[0:2] = 0.0  # Remove gradient artefacts

        n = 500
        b = [1.0 / n] * n
        dV = lfilter(b, 1, dV)
        dV = abs(dV) / abs(dV).max()
        ddV = np.gradient(dV)
        ddV[0:2] = 0.0  # Remove gradient artefacts
        ddV = lfilter(b, 1, ddV)
        ddV = abs(ddV) / abs(ddV).max()

        TOL = v_tol

        # Find Q start
        for i in range(idx_start, len(V)):
            if (dV[i] > TOL) & (i > 10):
                break
        q_start_idx = i
        q_start_t = t[i]

        # Find T end
        for i in range(idx_end - 1, idx_start, -1):
            if (dV[i] > TOL):
                break
        t_end_idx = i
        t_end_t = t[i]

        # Find QRS end
        window = 5000
        dV = abs(dV)
        max_window = np.zeros(t_end_idx - q_start_idx + 1)
        for i in range(q_start_idx, t_end_idx):
            max_window[i - q_start_idx] = dV[i:(i + window)].max()
        print('evaluated max window')
        min = max_window[0]
        for i in range(0, len(max_window)):
            if (min >= max_window[i]):
                min = max_window[i]
            elif (min < (max_window.min() + 0.05)):
                break
        qrs_end_idx = i + q_start_idx
        qrs_end_t = t[qrs_end_idx]

        qrs_dur = qrs_end_t - q_start_t

        # Find T peak timing and amplitude
        segment = V[qrs_end_idx:(t_end_idx + 1)]
        t_amplitude = abs(segment).max()
        t_peak_idx = np.where(abs(segment) == t_amplitude)[0][0] + qrs_end_idx
        t_sign = np.sign(segment[t_peak_idx - qrs_end_idx])
        t_amplitude = t_sign * t_amplitude
        t_peak_t = t[t_peak_idx]

        # Find T start
        segment = ddV[qrs_end_idx:t_peak_idx]
        min_dd_idx = np.where(segment == segment.min())[0][0] + qrs_end_idx
        for i in range(min_dd_idx, t_peak_idx):
            if (abs(ddV[i]) > 0.0005):
                break
        t_start_idx = i
        t_start_t = t[t_start_idx]

        t_dur = t_end_t - t_start_t

        qt_dur = t_end_t - q_start_t
        t_pe = t_end_t - t_peak_t
        t_op = t_peak_t - t_start_t

        landmarks = np.array([[q_start_t, V[q_start_idx]], [qrs_end_t, V[qrs_end_idx]], [t_start_t, V[t_start_idx]],
                                 [t_peak_t, V[t_peak_idx]], [t_end_t, V[t_end_idx]]])
        return qrs_dur, t_dur, t_pe, t_op, t_amplitude, qt_dur, landmarks


if __name__ == '__main__':
    false = ['False', 'false', 'F', 'f']
    true = ['True', 'true', 'T', 't']
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", help='Either ecg or pv, or single cell')
    parser.add_argument("--name", help='Name of simulation', default='heart_remeshed_3D')
    parser.add_argument("--refresh", help='Delete previous reading of ECG and PV', default=False)
    parser.add_argument("--lead", help='Name of the signal lead to plot')
    parser.add_argument("--CL", help='Cycle length of simulation, (s)', default=0.8)
    parser.add_argument("--beat", help='Beat number for single beat plots', default=0)
    parser.add_argument("--figure_title", help='Title of the figure', default='')
    parser.add_argument("--analysis", help='Title of the figure', default=False)
    parser.add_argument("--compare", help='Directory to compare with', default='')
    parser.add_argument("--material", help='Directory to compare with', default=1)
    parser.add_argument("--show", help='Toggle whether to show plot', default=True)
    args = parser.parse_args()
    plot_type = args.type
    name = args.name
    refresh = bool(args.refresh)
    if refresh in true:
        os.system('rm ecgs.pl pvs.pl')
    lead_name = args.lead
    CL = float(args.CL)
    beat = int(args.beat)
    figure_title = args.figure_title
    analysis = args.analysis
    compare = args.compare
    material = int(args.material)
    show = args.show

    if show in false:
        show = False
    if analysis in true:
        analysis = True

    # Set up visualisation
    a = ECGPV_visualisation(CL)
    if not (plot_type == 'cell'):
        ecgs, pvs = a.read_ecg_pv(name, './')
    epi, mid, endo = a.read_single_cell('./', material)
    if compare:
        ecgs2, pvs2 = a.read_ecg_pv(name, compare)
        epi2, mid2, endo2 = a.read_single_cell(compare, material)
        if analysis:
            if (plot_type == 'live') | (plot_type == 'p') | (plot_type == 'v') | (plot_type == 'pv'):
                a.analysis_PV(pvs, beat)
                print('second one...')
                a.analysis_PV(pvs2, beat)
            elif (plot_type == 'ecg'):
                a.analysis_ECG(ecgs, beat, show)
                a.analysis_ECG(ecgs2, beat, show)
        if plot_type == 'live':
            a.plot_ecgpv_live(ecgs, pvs, figure_title, show, ecgs2, pvs2)
        elif plot_type == 'ecg':
            if lead_name:
                a.plot_ecg_lead(ecgs, lead_name, lead_name, show, beat, ecgs2)
            else:
                a.plot_ecg_all_leads(ecgs, show, beat, ecgs2)
        elif plot_type == 'p':
            a.plot_pv_signal(pvs, 'p', 'Pt', show, beat, pvs2)
        elif plot_type == 'v':
            a.plot_pv_signal(pvs, 'v', 'Vt', show, beat, pvs2)
        elif plot_type == 'pv':
            a.plot_pv_signal(pvs, 'pv', 'PV_loop', show, beat, pvs2)
        elif plot_type == 'cell':
            a.plot_single_cell(epi, mid, endo, material, analysis, show, epi2, mid2, endo2)
    else:
        if analysis:
            if (plot_type == 'live') | (plot_type == 'p') | (plot_type == 'v') | (plot_type == 'pv'):
                a.analysis_PV(pvs, beat)
            elif (plot_type == 'ecg'):
                a.analysis_ECG(ecgs, beat, show)
        if plot_type == 'live':
            a.plot_ecgpv_live(ecgs, pvs, figure_title, show)
        elif plot_type == 'ecg':
            if lead_name:
                assert lead_name, 'Please enter lead name using option: lead='
                a.plot_ecg_lead(ecgs, lead_name, lead_name, show, beat)
            else:
                a.plot_ecg_all_leads(ecgs, show, beat)
        elif plot_type == 'p':
            a.plot_pv_signal(pvs, 'p', 'Pt', show, beat)
        elif plot_type == 'v':
            a.plot_pv_signal(pvs, 'v', 'Vt', show, beat)
        elif plot_type == 'pv':
            a.plot_pv_signal(pvs, 'pv', 'PV_loop', show, beat)
        elif plot_type == 'cell':
            a.plot_single_cell(epi, mid, endo, material, analysis, show)

# EOF
