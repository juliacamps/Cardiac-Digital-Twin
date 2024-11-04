        function [] = Twave_personalisation_POMs_ToRORd_IKs_not_converged(nb_models, celltype, stimAmp, stimulus_function_str, result_dir)
    % Population of models generation for T-wave personalisation
    % Adapted from scriptDemonstration_2_ParameterComparison.m:
    %% Setting parameters
    param_baseline.bcl = 800; % basic cycle length in ms
    param_baseline.model = @model_ToRORd_step_stimulus_rescaled_IKs_60_percent_GKr; %@model_ToRORd_rescaled_IKs; % which model is to be used
    
    %%
    param_baseline.verbose = false; % printing numbers of beats simulated.
    param_baseline.stimAmp = -53;
    options = [];
    beats = 100;
    ignoreFirst = beats - 1;


    %% Run for ### beats at baseline multipliers
    X0 = getStartingState_ToRORd(['m_',celltype]); % starting state - can be also m_mid or m_epi for midmyocardial or epicardial cells respectively.

    % Simulation and extraction of outputs
    if strcmp(celltype, 'endo')
        param_baseline.cellType = 0; %0 endo, 1 epi, 2 mid
    elseif strcmp(celltype, 'epi')
        param_baseline.cellType = 1; %0 endo, 1 epi, 2 mid
    elseif strcmp(celltype, 'mid')
        param_baseline.cellType = 2; %0 endo, 1 epi, 2 mid
    else
        error('Cell type defined is not one of endo, epi, mid');
    end
    [time, X_converged_baseline] = modelRunner_ToRORd(X0, options, param_baseline, beats, ignoreFirst);
%     %% For debugging comparison with Alya:
%     currents = getCurrentsStructure_ToRORd(time, X_converged_baseline, beats, param_baseline, 0);
%     subplot(1,2,1); plot(currents.time, currents.V);
%     subplot(1,2,2); plot(currents.time, currents.Cai);
%     
    %%
    X_converged_baseline = X_converged_baseline{end}(size(X_converged_baseline{end},1),:);

    %% Simulation and output extraction
    % Now, the structure of parameters is used to run multiple models in a
    % parallel-for loop.
    param_dictionary.bcl = param_baseline.bcl; % basic cycle length in ms
    if strcmp(stimulus_function_str, 'step')
        param_dictionary.model = @model_ToRORd_step_stimulus_rescaled_IKs_60_percent_GKrr; % which model is to be used
    elseif strcmp(stimulus_function_str, 'diffusion')
        param_dictionary.model = @model_ToRORd_diffusion_current_rescaled_IKs_60_percent_GKr; % which model is to be used
    end
    param_dictionary.verbose = false; % printing numbers of beats simulated.
    param_dictionary.celltype = celltype;

%     lhs_multipliers = lhsdesign(nb_models, 1);
    lhs_multipliers = linspace(0., 1., nb_models)';
    %lhs_multipliers(:,1) = lhs_multipliers(:,1) * 2.7 + 0.3;
    %lhs_multipliers(:,1) = lhs_multipliers(:,1) * 9.9 + 0.1;
    lhs_multipliers(:,1) = lhs_multipliers(:,1) * 49.98 + 0.02;
    params_dictionary(1:size(lhs_multipliers,1)) = param_dictionary;
    for ipopulation = 1:size(lhs_multipliers,1)
%         params_dictionary(ipopulation).Ito_Multiplier = 1.0 ; %lhs_multipliers(ipopulation,1);
        params_dictionary(ipopulation).IKs_Multiplier = lhs_multipliers(ipopulation,1);
    end


    biomarker_apd40 = zeros(size(params_dictionary));
    biomarker_apd50 = zeros(size(params_dictionary));
    biomarker_apd60 = zeros(size(params_dictionary));
    biomarker_apd70 = zeros(size(params_dictionary));
    biomarker_apd80 = zeros(size(params_dictionary));
    biomarker_apd90 = zeros(size(params_dictionary));
    biomarker_dvdt_max = zeros(size(params_dictionary));
    biomarker_vpeak = zeros(size(params_dictionary));
    biomarker_CTD50 = zeros(size(params_dictionary));
    biomarker_CTD90 = zeros(size(params_dictionary));
    biomarker_RMP = zeros(size(params_dictionary));
    biomarker_CaTmax = zeros(size(params_dictionary));
    biomarker_CaTmin = zeros(size(params_dictionary));

    biomarker_activation_time = zeros(size(params_dictionary));
    biomarker_signal_time = zeros(size(params_dictionary));


    beats = 1; % To mimick Alya simulation conditions
    ignoreFirst = beats -1;
    tic
    if strcmp(stimulus_function_str, 'diffusion')
        t = 0:100;
        Istim = get_diffusion_current(t, 1);
        max_Istim = max(Istim);
    else
        max_Istim = 1;
    end

    for j = 1:length(params_dictionary)
        params_dictionary(j).stimAmp = stimAmp;
        params_dictionary(j).Istim_sf = stimAmp / max_Istim;
    end

    parfor i = 1:length(params_dictionary)
        % Simulation and extraction of outputs
        [time, X] = modelRunner_ToRORd(X_converged_baseline, options, params_dictionary(i), beats, ignoreFirst);
        currents = getCurrentsStructure_ToRORd(time, X, beats, params_dictionary(i), 0);
        shared_vm(i).value = currents.V;
        shared_time(i).value = currents.time;
        shared_cai(i).value = currents.Cai;
        activation_index = 1;
        if strcmp(stimulus_function_str, 'diffusion')
            [activation_index] = find_ap_activation(currents.time, currents.V,70, 2);
        end
        trimmed_vm = currents.V(activation_index:end);
        trimmed_time = currents.time(activation_index:end) - currents.time(activation_index);
        trimmed_cai = currents.Cai(activation_index:end);
%         shared_vm(i).value = trimmed_vm;
%         shared_time(i).value = trimmed_time;
%         shared_cai(i).value = currents.Cai;
        biomarker_activation_time(i) = currents.time(activation_index);
%         biomarker_activation_time(i) = 1; % There is no need for this improvement, because there is no activity before the upstroke

        biomarker_apd40(i) = DataReporter.getAPD_ignore_first_10ms(trimmed_time, trimmed_vm, 0.4);
        biomarker_apd50(i) = DataReporter.getAPD_ignore_first_10ms(trimmed_time, trimmed_vm, 0.5);
        biomarker_apd60(i) = DataReporter.getAPD_ignore_first_10ms(trimmed_time, trimmed_vm, 0.6);
        biomarker_apd70(i) = DataReporter.getAPD_ignore_first_10ms(trimmed_time, trimmed_vm, 0.7);
        biomarker_apd80(i) = DataReporter.getAPD_ignore_first_10ms(trimmed_time, trimmed_vm, 0.8);
        biomarker_apd90(i) = DataReporter.getAPD_ignore_first_10ms(trimmed_time, trimmed_vm, 0.9);

        biomarker_dvdt_max(i) = DataReporter.getPeakDVDT(trimmed_time, trimmed_vm, -1.0);
        biomarker_CTD50(i) = DataReporter.getAPD_ignore_first_10ms(trimmed_time, trimmed_cai, 0.5);
        biomarker_CTD90(i) = DataReporter.getAPD_ignore_first_10ms(trimmed_time, trimmed_cai, 0.9);
        biomarker_CaTmax(i) = max(trimmed_cai);
        biomarker_CaTmin(i) = min(trimmed_cai);
        biomarker_vpeak(i) = max(trimmed_vm);
        biomarker_RMP(i) = min(trimmed_vm);

    end


    biomarker_tri_90_40 = biomarker_apd90 - biomarker_apd40;


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
    vpeak_max = 200; %55; % Because we have changed the stimulus current to emulate diffusion.
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
    population_calibration_biomarkers = [biomarker_apd40; biomarker_apd50; biomarker_apd90; biomarker_tri_90_40; ...
        biomarker_dvdt_max; biomarker_vpeak; biomarker_RMP; biomarker_CTD50; biomarker_CTD90]'; %'
    calibrated_population_biomarkers_index = (population_calibration_biomarkers > calibration_criteria_min) & ( population_calibration_biomarkers < calibration_criteria_max);
    calibrated_population_biomarkers_index = all(calibrated_population_biomarkers_index, 2);
%     calibrated_population_biomarkers = population_calibration_biomarkers(calibrated_population_biomarkers_index,:);
    calibrated_population_vm = shared_vm(calibrated_population_biomarkers_index);
    calibrated_population_time = shared_time(calibrated_population_biomarkers_index);
    calibrated_population_cai = shared_cai(calibrated_population_biomarkers_index);

    % Resample action potentials at 1000 Hz
    max_simulation_time = 0;
    min_simulation_time = 100;
    for cellmodel_i = 1:size(calibrated_population_vm, 2)   % structures have always a 1 in the first dimention
        max_simulation_time = max(max(calibrated_population_time(cellmodel_i).value), max_simulation_time);
        min_simulation_time = min(min(calibrated_population_time(cellmodel_i).value), min_simulation_time);
    end
    disp('min_simulation_time');
    disp(min_simulation_time);
    calibrated_time = 0:ceil(max_simulation_time); % time given by model is in ms.
    calibrated_vm = zeros(size(calibrated_population_vm, 2), length(calibrated_time));
    calibrated_cai = zeros(size(calibrated_population_vm, 2), length(calibrated_time));
    for cellmodel_i = 1:size(calibrated_population_vm, 2)
        simulation_t = calibrated_population_time(cellmodel_i).value;
        idx = knnsearch(simulation_t, calibrated_time'); %' Knnsearch needs second input to be column vector.
        calibrated_vm(cellmodel_i,:) = calibrated_population_vm(cellmodel_i).value(idx);
        calibrated_cai(cellmodel_i,:) = calibrated_population_cai(cellmodel_i).value(idx);
    end

    % Trim end based on last instance of activity
    activity_vm = zeros(size(calibrated_vm));
    activity_vm(:, 2:end) = abs(calibrated_vm(:, 2:end) - calibrated_vm(:, 1:end-1));
    activity_threshold = 50*max(activity_vm(:, end-1));
    activity_cut_index = 1;
    for cellmodel_i = 1:size(calibrated_vm, 1)
        activity_cut_index = max(activity_cut_index, find(activity_vm(cellmodel_i, :) > activity_threshold,1, 'last'));
    end
    calibrated_vm = calibrated_vm(:, 1:activity_cut_index);
    calibrated_time = calibrated_time(1:activity_cut_index);
    calibrated_cai = calibrated_cai(:, 1:activity_cut_index);

    % Trim start based on first instance of activity
    activity_vm = zeros(size(calibrated_vm));
    activity_vm(:, 1:end-1) = abs(calibrated_vm(:, 2:end) - calibrated_vm(:, 1:end-1));
    activity_threshold = 50*max(activity_vm(:, 1));
    activation_index_shift = size(calibrated_vm, 2);
    for cellmodel_i = 1:size(calibrated_vm, 1)
        activation_index_shift = min(activation_index_shift, find(activity_vm(cellmodel_i, :) > activity_threshold,1, 'first'));
    end
    activation_time_shift = activation_index_shift - 1; % time starts at zero but indexes start at 1
    disp('activation_time_shift in ms');
    disp(activation_time_shift);
    calibrated_vm = calibrated_vm(:, activation_index_shift:end);
    calibrated_time = calibrated_time(activation_index_shift:end) - activation_time_shift;
    calibrated_cai = calibrated_cai(:, activation_index_shift:end);
%     activaiton_time_shift = activity_cut_index;

    % Save as table based on integer values of APD90 and APD50 (based on
    % variability) in ms.
    apd40 = biomarker_apd40(calibrated_population_biomarkers_index)';    %'
    apd50 = biomarker_apd50(calibrated_population_biomarkers_index)';    %'
    apd60 = biomarker_apd60(calibrated_population_biomarkers_index)';    %'
    apd70 = biomarker_apd70(calibrated_population_biomarkers_index)';    %'
    apd80 = biomarker_apd80(calibrated_population_biomarkers_index)';    %'
    apd90 = biomarker_apd90(calibrated_population_biomarkers_index)';    %'
    tri_90_40 = biomarker_tri_90_40(calibrated_population_biomarkers_index)';    %'
    dvdt_max = biomarker_dvdt_max(calibrated_population_biomarkers_index)';    %'
    CTD50 = biomarker_CTD50(calibrated_population_biomarkers_index)';    %'
    CTD90 = biomarker_CTD90(calibrated_population_biomarkers_index)';    %'
    vpeak = biomarker_vpeak(calibrated_population_biomarkers_index)';    %'
    RMP = biomarker_RMP(calibrated_population_biomarkers_index)';    %'
    CaT_max = biomarker_CaTmax(calibrated_population_biomarkers_index)';    %'
    CaT_min = biomarker_CaTmin(calibrated_population_biomarkers_index)';    %'
    sf_IKs = lhs_multipliers(calibrated_population_biomarkers_index,1);
    activation_time = biomarker_activation_time(calibrated_population_biomarkers_index)' - activation_time_shift;
    biomarkers = table(activation_time, apd40, apd50, apd60, apd70, apd80, apd90, tri_90_40, dvdt_max, ...
        CTD50, CTD90, vpeak, RMP, CaT_max, CaT_min, sf_IKs);
    writetable(biomarkers, [result_dir '/biomarkers_table_' celltype '.csv'],'WriteRowNames',false)
    writematrix(calibrated_vm, [result_dir '/torord_calibrated_pom_1000Hz_vm_' celltype '.csv']);
    writematrix(calibrated_time, [result_dir '/torord_calibrated_pom_1000Hz_time_' celltype '.csv']);
    writematrix(calibrated_cai, [result_dir '/torord_calibrated_pom_1000Hz_cai_' celltype '.csv']);

    disp('size(calibrated_time)');
    disp(size(calibrated_time));
    figure;
    title(['Celltype: ' celltype ', StimAmp: ' num2str(stimAmp)])
    hold on;
%     oliTraces = load('data/oliTraces.mat');
%     for i = 1:size(oliTraces.tracesRealigned,2)
%         plot(oliTraces.referenceTime, oliTraces.tracesRealigned(:,i), 'k');
%     end
    disp('Time per model: ')
    disp(toc/nb_models)
    for cellmodel_i = 1:size(calibrated_vm, 1)
         plot(calibrated_time, calibrated_vm(cellmodel_i, :), 'b');
    end
    hold off;
    
    figure;
    title(['Celltype: ' celltype ', StimAmp: ' num2str(stimAmp)])
    hold on;
%     oliTraces = load('data/oliTraces.mat');
%     for i = 1:size(oliTraces.tracesRealigned,2)
%         plot(oliTraces.referenceTime, oliTraces.tracesRealigned(:,i), 'k');
%     end
    disp('Time per model: ')
    disp(toc/nb_models)
    for cellmodel_i = 1:size(calibrated_cai, 1)
         plot(calibrated_time, calibrated_cai(cellmodel_i, :), 'b', 'linewidth', 0.5);
    end
    hold off;


    figure;
    histogram(activation_time);
    % apd40_importance = std(APD40./APD90)
    % apd50_importance = std(APD50./APD90)
    % apd60_importance = std(APD60./APD90)
    % apd70_importance = std(APD70./APD90)
    % apd80_importance = std(APD80./APD90)

    %% Plot calibrated population with experimental traces
    figure;
    title(['Calibrated celltype: ' celltype ', StimAmp: ' num2str(stimAmp)])
    hold on;
    oliTraces = load('data/oliTraces.mat');
    for i = 1:size(oliTraces.tracesRealigned,2)
        plot(oliTraces.referenceTime, oliTraces.tracesRealigned(:,i), 'k');
    end
    for i = 1:size(calibrated_vm,1)
        plot(calibrated_vm(i,:), 'b');
    end
    hold off;
    
    figure;
    histogram(apd90);
    title(['Calibrated APD90 histogram: ' celltype])
    %% 
    figure; hold on;
    target_apds = cast(min(apd90), 'int64'):cast(max(apd90), 'int64');
    idx = knnsearch(apd90, target_apds');    %'
    plot(target_apds, apd90(idx), '.');
    xlabel('Target APD90 (ms)');
    ylabel('Provided APD90 (ms)');
    title(['Error in APD90 generation in dictionary ', celltype]);
    hold off;
    min(apd90)
    max(apd90)
    
    %%
    
%     figure;
%     title(['Calibrated celltype: ' celltype ', StimAmp: ' num2str(stimAmp)])
%     hold on;
%     oliTraces = load('data/oliTraces.mat');
%     for i = 1:size(oliTraces.tracesRealigned,2)
%         plot(oliTraces.referenceTime, oliTraces.tracesRealigned(:,i), 'k');
%     end
%     for i = 1:size(calibrated_vm,1)
%         plot(calibrated_vm(i,:), 'g');
%     end
%     hold off;
%     
%     figure;
%     histogram(apd90);
%     title(['Calibrated APD90 histogram: ' celltype ' max: ', num2str(max(apd90)), ' min: ', num2str(min(apd90))])
%     %% 
%     figure; hold on;
%     target_apds = cast(min(apd90), 'int64'):cast(max(apd90), 'int64');
%     idx = knnsearch(apd90, target_apds');    %'
%     plot(target_apds, apd90(idx));
%     xlabel('Target APD90 (ms)');
%     ylabel('Provided APD90 (ms)');
%     title(['Error in APD90 generation in dictionary ', celltype]);
%     hold off;
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
    time_trimmed(end-front_shift_idx:end) = linspace(last_trimmed_time, time(end), front_shift_idx+1)';    %'
end


function [cut_index] = find_ap_activation(time_series, vm, percentage, front_padding_time)
%    vm_trimmed = vm;
%    cai_trimmed = cai;
%    time_trimmed = time_series;
    vm_range = max(vm) - min(vm);
    vm_threshold = vm_range * (1 - percentage/100.0) + min(vm);
    upstroke_index = find(vm > vm_threshold,1);
    time_of_upstroke = time_series(upstroke_index);
    cut_index = find((time_of_upstroke-time_series)<front_padding_time, 1);
    %vm_trimmed = vm(cut_index:end);
    %cai_trimmed = cai(cut_index:end);
    %time_trimmed = time_series(cut_index:end) - time_series(cut_index);
end
