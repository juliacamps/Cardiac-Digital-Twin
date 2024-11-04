clear
clc
close all
nb_models = 1000;
stimAmps = -53*(0.5:0.3:1.5); 
for i = 1:length(stimAmps)
    Twave_personalisation_POMs_ToRORd_converged(nb_models, 'endo', stimAmps(i),stimulus_function, result_dir);
%     Twave_personalisation_POMs_ToRORd(nb_models, 'mid', stimAmps(i));
    Twave_personalisation_POMs_ToRORd_converged(nb_models,'epi', stimAmps(i), stimulus_function, result_dir);
end