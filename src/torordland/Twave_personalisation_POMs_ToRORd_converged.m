function [] = Twave_personalisation_POMs_ToRORd_converged(nb_models,celltype,stimAmp, stimulus_function_str, result_dir)
% Population of models generation for T-wave personalisation
% Adapted from scriptDemonstration_2_ParameterComparison.m:
% TODO: rename all endo_ variables to not say endo_
%% Setting parameters
param.bcl = 800; % basic cycle length in ms
if strcmp(stimulus_function_str, 'step')
    param.model = @model_ToRORd_step_stimulus; % which model is to be used
elseif strcmp(stimulus_function_str, 'diffusion')  
    param.model = @model_ToRORd_diffusion_current; % which model is to be used
end
param.verbose = false; % printing numbers of beats simulated.
param.stimAmp = stimAmp;
if strcmp(stimulus_function_str, 'diffusion')
    t = 0:100;
    Istim = get_diffusion_current(t, 1);
    max_Istim = max(Istim);
else
    max_Istim = 1;
end
param.Istim_sf = stimAmp / max_Istim;

lhs_multipliers = lhsdesign(nb_models, 15) * 1.5 + 0.5;
params(1:size(lhs_multipliers,1)) = param;
for ipopulation = 1:size(lhs_multipliers,1)
    params(ipopulation).INa_Multiplier = lhs_multipliers(ipopulation,1);
    params(ipopulation).INaL_Multiplier = lhs_multipliers(ipopulation,2);
    params(ipopulation).Ito_Multiplier = lhs_multipliers(ipopulation,3);
    params(ipopulation).IKr_Multiplier = lhs_multipliers(ipopulation,4);
    params(ipopulation).IKs_Multiplier = lhs_multipliers(ipopulation,5);
    params(ipopulation).IK1_Multiplier = lhs_multipliers(ipopulation,6);
    params(ipopulation).INCX_Multiplier = lhs_multipliers(ipopulation,7);
    params(ipopulation).INaK_Multiplier = lhs_multipliers(ipopulation,8);
    params(ipopulation).ICaL_Multiplier = lhs_multipliers(ipopulation,9);
    params(ipopulation).Jrel_Multiplier = lhs_multipliers(ipopulation,10);
    params(ipopulation).Jup_Multiplier = lhs_multipliers(ipopulation,11);
    %params(ipopulation).ca50_Multiplier = lhs_multipliers(ipopulation,12);
    %params(ipopulation).kuw_Multiplier = lhs_multipliers(ipopulation,13);
    %params(ipopulation).kws_Multiplier = lhs_multipliers(ipopulation,14);
    %params(ipopulation).ksu_Multiplier = lhs_multipliers(ipopulation,15);
end
options = [];
beats = 100;
ignoreFirst = beats - 1;

%% Simulation and output extraction
% Now, the structure of parameters is used to run multiple models in a
% parallel-for loop.
% figure;
% hold on;
% oliTraces = load('data/oliTraces.mat');
% for i = 1:size(oliTraces.tracesRealigned,2)
%     plot(oliTraces.referenceTime, oliTraces.tracesRealigned(:,i), 'k');
% end
endo_apd40 = zeros(size(params));
endo_apd50 = zeros(size(params));
endo_apd60 = zeros(size(params));
endo_apd70 = zeros(size(params));
endo_apd80 = zeros(size(params));
endo_apd90 = zeros(size(params));
endo_dvdt_max = zeros(size(params));
endo_vpeak = zeros(size(params));
endo_CTD50 = zeros(size(params));
endo_CTD90 = zeros(size(params));
endo_RMP = zeros(size(params));
endo_CaTmax = zeros(size(params));
endo_CaTmin = zeros(size(params));
tic
for i = 1:length(params)
%     celltype = 'endo';
    X0 = getStartingState_ToRORd(['m_',celltype]); % starting state - can be also m_mid or m_epi for midmyocardial or epicardial cells respectively.
    
    % Simulation and extraction of outputs
    params(i).cellType = celltype; %0 endo, 1 epi, 2 mid
    [time, X] = modelRunner_ToRORd(X0, options, params(i), beats, ignoreFirst);
    
    currents = getCurrentsStructure_ToRORd(time, X, beats, params(i), 0);
    if strcmp(stimulus_function_str, 'diffusion')
        [currents.V, currents.Cai, currents.time] = trim_vm_on_activation(currents.time, currents.V, currents.Cai,70, 2);     
    end 
    endo_V(i).value = currents.V;
    endo_time(i).value = currents.time;
    endo_cai(i).value = currents.Cai;
    
    endo_apd40(i) = DataReporter.getAPD_ignore_first_10ms(currents.time, currents.V, 0.4);
    endo_apd50(i) = DataReporter.getAPD_ignore_first_10ms(currents.time, currents.V, 0.5);
    endo_apd60(i) = DataReporter.getAPD_ignore_first_10ms(currents.time, currents.V, 0.6);
    endo_apd70(i) = DataReporter.getAPD_ignore_first_10ms(currents.time, currents.V, 0.7);
    endo_apd80(i) = DataReporter.getAPD_ignore_first_10ms(currents.time, currents.V, 0.8);
    endo_apd90(i) = DataReporter.getAPD_ignore_first_10ms(currents.time, currents.V, 0.9);

    endo_dvdt_max(i) = DataReporter.getPeakDVDT(currents.time, currents.V, -1.0);
    endo_CTD50(i) = DataReporter.getAPD_ignore_first_10ms(currents.time, currents.Cai, 0.5);
    endo_CTD90(i) = DataReporter.getAPD_ignore_first_10ms(currents.time, currents.Cai, 0.9);
    endo_CaTmax(i) = max(currents.Cai);
    endo_CaTmin(i) = min(currents.Cai);
    endo_vpeak(i) = max(currents.V);
    endo_RMP(i) = min(currents.V);
%     figure
%     plot(currents.time, currents.V);
%     figure
%     plot(currents.time, currents.Cai);
%     endo_apd40(i)
%     endo_apd90(i)
%     endo_CTD50(i)
%     endo_CTD90(i)
    
end  
disp('Time per model: ')
disp(toc/nb_models)
% for i = 1:length(params)
%      plot(endo_time(i).value, endo_V(i).value, 'b');
% end
% hold off; 
endo_tri_90_40 = endo_apd90 - endo_apd40;


%% Calibration based on Table 2 of Passini et al. 2019 https://doi.org/10.1111/bph.14786
apd40_min = 85;
apd40_max = 320;
apd50_min = 110;
apd50_max = 350;
apd90_min = 180;
apd90_max = 440;
tri_90_40_min = 50;
tri_90_40_max = 150;
dvdt_max_min = 100;
dvdt_max_max = 1000;
vpeak_min = 10;
vpeak_max = 55;
rmp_min = -95;
rmp_max = -80;
ctd50_min = 120;
ctd50_max = 420;
ctd90_min = 220;
ctd90_max = 785;

calibration_criteria_min = [apd40_min, apd50_min, apd90_min, tri_90_40_min,...
    dvdt_max_min, vpeak_min, rmp_min, ctd50_min, ctd90_min];
calibration_criteria_max = [apd40_max, apd50_max, apd90_max, tri_90_40_max,...
    dvdt_max_max, vpeak_max, rmp_max, ctd50_max, ctd90_max];
endo_population_biomarkers = [endo_apd40; endo_apd50; endo_apd90; endo_tri_90_40; ...
    endo_dvdt_max; endo_vpeak; endo_RMP; endo_CTD50; endo_CTD90]';
calibrated_endo_population_biomarkers_index = (endo_population_biomarkers > calibration_criteria_min) & ( endo_population_biomarkers < calibration_criteria_max);
calibrated_endo_population_biomarkers_index = all(calibrated_endo_population_biomarkers_index, 2);
calibrated_endo_population_biomarkers = endo_population_biomarkers(calibrated_endo_population_biomarkers_index,:);
calibrated_endo_population_V = endo_V(calibrated_endo_population_biomarkers_index);
calibrated_endo_population_t = endo_time(calibrated_endo_population_biomarkers_index);

% Resample action potentials at 1000 Hz
t = 0:param.bcl; % time given by model is in ms.
v = zeros(size(calibrated_endo_population_V,1), length(t));
for i = 1:size(calibrated_endo_population_V,2)
    simulation_t = calibrated_endo_population_t(i).value;
    idx = knnsearch(simulation_t, t'); % Knnsearch needs second input to be column vector. 
    v(i,:) = calibrated_endo_population_V(i).value(idx);
end


% Save as table based on integer values of APD90 and APD50 (based on
% variability) in ms. 
APD40 = endo_apd40(calibrated_endo_population_biomarkers_index)';
APD50 = endo_apd50(calibrated_endo_population_biomarkers_index)';
APD60 = endo_apd60(calibrated_endo_population_biomarkers_index)';
APD70 = endo_apd70(calibrated_endo_population_biomarkers_index)';
APD80 = endo_apd80(calibrated_endo_population_biomarkers_index)';
APD90 = endo_apd90(calibrated_endo_population_biomarkers_index)';
tri_90_40 = endo_tri_90_40(calibrated_endo_population_biomarkers_index)';
dvdt_max = endo_dvdt_max(calibrated_endo_population_biomarkers_index)';
CTD50 = endo_CTD50(calibrated_endo_population_biomarkers_index)';
CTD90 = endo_CTD90(calibrated_endo_population_biomarkers_index)';
vpeak = endo_vpeak(calibrated_endo_population_biomarkers_index)';
RMP = endo_RMP(calibrated_endo_population_biomarkers_index)';
CaT_max = endo_CaTmax(calibrated_endo_population_biomarkers_index)';
CaT_min = endo_CaTmin(calibrated_endo_population_biomarkers_index)';
sf_INa = lhs_multipliers(calibrated_endo_population_biomarkers_index,1);
sf_INaL = lhs_multipliers(calibrated_endo_population_biomarkers_index,2);
sf_Ito = lhs_multipliers(calibrated_endo_population_biomarkers_index,3);
sf_IKr = lhs_multipliers(calibrated_endo_population_biomarkers_index,4);
sf_IKs = lhs_multipliers(calibrated_endo_population_biomarkers_index,5);
sf_IK1 = lhs_multipliers(calibrated_endo_population_biomarkers_index,6);
sf_INCX = lhs_multipliers(calibrated_endo_population_biomarkers_index,7);
sf_INaK = lhs_multipliers(calibrated_endo_population_biomarkers_index,8);
sf_ICaL = lhs_multipliers(calibrated_endo_population_biomarkers_index,9);
sf_Jrel = lhs_multipliers(calibrated_endo_population_biomarkers_index,10);
sf_Jup = lhs_multipliers(calibrated_endo_population_biomarkers_index,11);
%sf_ca50 = lhs_multipliers(calibrated_endo_population_biomarkers_index,12);
%sf_kuw = lhs_multipliers(calibrated_endo_population_biomarkers_index,13);
%sf_kws = lhs_multipliers(calibrated_endo_population_biomarkers_index,14);
%sf_ksu = lhs_multipliers(calibrated_endo_population_biomarkers_index,15);
sf_ca50 = ones(size(sf_Jup));
sf_kuw = ones(size(sf_Jup));
sf_kws = ones(size(sf_Jup));
sf_ksu = ones(size(sf_Jup));
T = table(APD40, APD50, APD60, APD70, APD80, APD90, tri_90_40, dvdt_max, ...
    CTD50, CTD90, vpeak, RMP, CaT_max, CaT_min, sf_INa, sf_INaL, sf_Ito, ...
    sf_IKr, sf_IKs, sf_IK1, sf_INCX, sf_INaK, sf_ICaL, sf_Jrel, sf_Jup, sf_ca50, ...
    sf_kuw, sf_kws, sf_ksu);
writetable(T,[result_dir '/biomarkers_table_' celltype '.csv'],'WriteRowNames',false)
writematrix(v, [result_dir '/torord_calibrated_pom_aps_1000Hz_' celltype '.csv']);

% apd40_importance = std(APD40./APD90)
% apd50_importance = std(APD50./APD90)
% apd60_importance = std(APD60./APD90)
% apd70_importance = std(APD70./APD90)
% apd80_importance = std(APD80./APD90)

%% Plot calibrated population with experimental traces
figure;
title(result_dir);
hold on;
oliTraces = load('data/oliTraces.mat');
for i = 1:size(oliTraces.tracesRealigned,2)
    plot(oliTraces.referenceTime, oliTraces.tracesRealigned(:,i), 'k');
end
for i = 1:size(v,1)
    plot(v(i,:), 'g');
end
hold off;
%% 
% figure; hold on;
% target_apds = 180:440;
% idx = knnsearch(APD90, target_apds');
% plot(target_apds, APD90(idx));
% xlabel('Target APD90 (ms)');
% ylabel('Provided APD90 (ms)');
% title('Error in APD90 generation in dictionary');
end

function [vm_trimmed, cai_trimmed, time_trimmed] = trim_vm_on_activation(time, vm, cai, percentage, front_padding_time)
    vm_trimmed = vm;
    cai_trimmed = cai;
    time_trimmed = time;
    vm_range = max(vm) - min(vm);
    vm_threshold = vm_range * (1 - percentage/100.0) + min(vm);
    cut_index = find(vm > vm_threshold,1);
    time_of_cut = time(cut_index);
    front_shift_idx = find((time_of_cut-time)<front_padding_time, 1);
    front_shift_time =time(front_shift_idx);
    % padded_index = max(cut_index - front_padding_idx, 1);
    vm_trimmed(1:end-front_shift_idx+1) = vm(front_shift_idx:end);
    vm_trimmed(end-front_shift_idx:end) = vm(end);
    cai_trimmed(1:end-front_shift_idx+1) = cai(front_shift_idx:end);
    cai_trimmed(end-front_shift_idx:end) = cai(end);
    time_trimmed(1:end-front_shift_idx+1) = time(front_shift_idx:end) - front_shift_time;
    last_trimmed_time = time_trimmed(end-front_shift_idx);
    time_trimmed(end-front_shift_idx:end) = linspace(last_trimmed_time, time(end), front_shift_idx+1)';
end

