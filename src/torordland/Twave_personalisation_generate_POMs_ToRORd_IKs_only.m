clear
clc
close all
nb_models = 10000;
stimAmp_step = [-53]; 
% stimAmp_diffusion = (2:0.1:2.5);%-53*(0.5:0.3:1.5); 

t = 0:100;
Istim = get_diffusion_current(t, 1);
max_Istim = max(Istim);
% stimAmp_diffusion = unique(round(stimAmp_diffusion*max_Istim));

stimAmp_diffusion = [11]; % Calibrated to achieve stimulation of all POMs for epi and endo and mid. 
stimulus_function_str_list = {'diffusion'}; %, 'step'};
simulation_protocol_list = {'not_converged'}; %, 'converged'};
celltype_list = {'endo'}; %, 'epi', 'mid'};
rng(1);
for simulation_protocol_i =1:length(simulation_protocol_list)  % simulation protocol
    simulation_protocol = simulation_protocol_list{simulation_protocol_i};
    for stimulus_function_str_i = 1:length(stimulus_function_str_list)  % stimulus_function
        stimulus_function_str = stimulus_function_str_list{stimulus_function_str_i};
        if strcmp(stimulus_function_str, 'step')
            stimAmp_list = stimAmp_step;
        elseif strcmp(stimulus_function_str, 'diffusion')
            stimAmp_list = stimAmp_diffusion;
        end
        for stimAmp_i = 1:length(stimAmp_list)  % stimulus amplitude
            stimAmp = stimAmp_list(stimAmp_i);
            stimAmp = round(stimAmp);
            result_dir = ['/data/Personalisation_projects/meta_data/cellular_data/' ...
                simulation_protocol '_' stimulus_function_str '_' num2str(abs(stimAmp)) '_GKs_only_GKs5_GKr0.5_tjca60_visualisation'];
            if ~isfolder(result_dir)
                mkdir(result_dir);
            end
            disp(['Processing ' result_dir]);
                for celltype_i = 1:length(celltype_list)
                    celltype = celltype_list{celltype_i};
                    if strcmp(simulation_protocol, 'not_converged')
                        Twave_personalisation_POMs_ToRORd_IKs_only_not_converged(nb_models, celltype, stimAmp, stimulus_function_str, result_dir);
                    elseif strcmp(simulation_protocol, 'converged')
                        Twave_personalisation_POMs_ToRORd_converged(nb_models, celltype, stimAmp, stimulus_function_str, result_dir);
                    end
                end
        end
    end
end
