%% Compare step stimulus amplitude on AP morphology
clear
close all
% param.bcl = 800; % basic cycle length in ms
% param.model = @model_ToRORd_step_stimulus_rescaled_IKs; % which model is to be used
% param.verbose = true; % printing numbers of beats simulated.
% stim_amps = [-53, -80, -100];
% figure;
% hold on
% 
% for i = 1:3
%     param.stimAmp = stim_amps(i);
%     param.cellType = 0; %0 endo, 1 epi, 2 mid
%     options = []; % parameters for ode15s - usually empty
%     beats = 50; % number of beats
%     ignoreFirst = beats - 1; % this many beats at the start of the simulations are ignored when extracting the structure of simulation outputs (i.e., beats - 1 keeps the last beat).
%     celltype = 'endo';
%     X0 = getStartingState_ToRORd(['m_',celltype]); % starting state - can be also m_mid or m_epi for midmyocardial or epicardial cells respectively.
% 
%     % Simulation and extraction of outputs
% 
%     [time, X] = modelRunner_ToRORd(X0, options, param, beats, ignoreFirst);
% 
%     currents = getCurrentsStructure_ToRORd(time, X, beats, param, 0);
%     plot(currents.time, currents.V);
% end
% legend('-53', '-80', '-100')
% title('Compare step stimulus amplitude on AP morphology');


%% Compare step vs diffusive  on AP morphology
clear
close all
param.bcl = 800; % basic cycle length in ms
param.model = @model_ToRORd_diffusion_current_rescaled_IKs; % which model is to be used
models = {@model_ToRORd_step_stimulus_rescaled_IKs, @model_ToRORd_diffusion_current_rescaled_IKs};
param.verbose = true; % printing numbers of beats simulated.
stim_amps = [-53, -80, -100];
figure;
hold on

for i = 1:2
    param.model = models{i};
    if i==2 
        t = 0:100;
        Istim = get_diffusion_current(t, 1);
        max_Istim = max(Istim);
    else
        max_Istim = 1;
    end
    if i == 2
        stimAmp = 11;
        param.Istim_sf = stimAmp / max_Istim;
    end
   
    param.cellType = 0; %0 endo, 1 epi, 2 mid
    options = []; % parameters for ode15s - usually empty
    beats = 50; % number of beats
    ignoreFirst = beats - 1; % this many beats at the start of the simulations are ignored when extracting the structure of simulation outputs (i.e., beats - 1 keeps the last beat).
    celltype = 'endo';
    X0 = getStartingState_ToRORd(['m_',celltype]); % starting state - can be also m_mid or m_epi for midmyocardial or epicardial cells respectively.

    % Simulation and extraction of outputs

    [time, X] = modelRunner_ToRORd(X0, options, param, beats, ignoreFirst);

    currents = getCurrentsStructure_ToRORd(time, X, beats, param, 0);
    plot(currents.time, currents.V);
end
legend('step', 'diffusive')
title('Compare step vs diffusive current');


%% Compare diffusive current magnitude on AP morphology
clear
close all
param.bcl = 800; % basic cycle length in ms
% param.model = @model_ToRORd_diffusion_current_rescaled_IKs; % which model is to be used
param.model = @model_ToRORd_diffusion_current;
param.verbose = true; % printing numbers of beats simulated.
stim_amps = [-53, -80, -100];
figure;
hold on
stimAmp = [8, 10, 12, 14];

for i = 1:4
    t = 0:100;
    Istim = get_diffusion_current(t, 1);
    max_Istim = max(Istim);

    param.Istim_sf = stimAmp(i) / max_Istim;
   
    param.cellType = 0; %0 endo, 1 epi, 2 mid
    options = []; % parameters for ode15s - usually empty
    beats = 50; % number of beats
    ignoreFirst = beats - 1; % this many beats at the start of the simulations are ignored when extracting the structure of simulation outputs (i.e., beats - 1 keeps the last beat).
    celltype = 'endo';
    X0 = getStartingState_ToRORd(['m_',celltype]); % starting state - can be also m_mid or m_epi for midmyocardial or epicardial cells respectively.

    % Simulation and extraction of outputs

    [time, X] = modelRunner_ToRORd(X0, options, param, beats, ignoreFirst);

    currents = getCurrentsStructure_ToRORd(time, X, beats, param, 0);
    plot(currents.time, currents.V);
end
legend( '8', '10', '12', '14')
title('Compare diffusive current amplitude on AP');

%%
% ActiveTension = X{1, 1}(:,44)*480; % = XS*Tref/dr - only if lambda=1.
% add Ta as output in getCurrentsStructure otherwise
activeTension = X{1,1}(:,44)*480;
subplot(1,3,1)
plot(currents.time-7, currents.V)
hold on
subplot(1,3,2)
plot(currents.time-7, currents.Cai)
hold on
subplot(1,3,3)
plot(currents.time-7, activeTension)
hold on
V = currents.V;
Cai = currents.Cai;
Ta = activeTension;
time = currents.time;

%%
param.cellType = 0; %0 endo, 1 epi, 2 mid
param.bcl = 800; % basic cycle length in ms
param.model = @model_ToRORd_Land; % which model is to be used
param.verbose = true; % printing numbers of beats simulated.
options = []; % parameters for ode15s - usually empty
beats = 50; % number of beats
ignoreFirst = beats - 1; % this many beats at the start of the simulations are ignored when extracting the structure of simulation outputs (i.e., beats - 1 keeps the last beat).
celltype = 'endo';
X0 = getStartingState(['m_',celltype]); % starting state - can be also m_mid or m_epi for midmyocardial or epicardial cells respectively.

% Simulation and extraction of outputs

[time, X] = modelRunner(X0, options, param, beats, ignoreFirst);

currents = getCurrentsStructure(time, X, beats, param, 0);

% ActiveTension = X{1, 1}(:,44)*480; % = XS*Tref/dr - only if lambda=1.
% add Ta as output in getCurrentsStructure otherwise
activeTension = X{1,1}(:,44)*480;
subplot(1,3,1)
plot(currents.time, currents.V)
hold on
subplot(1,3,2)
plot(currents.time, currents.Cai)
hold on
subplot(1,3,3)
plot(currents.time, activeTension)
hold on
V = currents.V;
Cai = currents.Cai;
Ta = activeTension;
time = currents.time;


