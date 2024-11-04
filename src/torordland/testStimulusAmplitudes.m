%% Test amplitude of stimulus on various aspects of ionic currents in 
% ToR-ORd+Land electro-mechanical model 

%% Setting parameters
clear
clc
close all
param.bcl = 800; % basic cycle length in ms
param.model = @model_ToRORd_Land; % which model is to be used
param.verbose = true; % printing numbers of beats simulated
param.cellType = 0; % 0 endo, 1, epi, 2, mid 

options = []; % parameters for ode15s - usually empty
beats = 500; % number of beats
ignoreFirst = beats - 1; % this many beats at the start of the simulations are ignored when extracting the structure of simulation outputs (i.e., beats - 1 keeps the last beat).

X0 = getStartingState('m_endo'); % starting state 

% Run ToRORd+Land for the following combinations of stimulus amplitude and
% duration
stimAmps = [-40, -53, -80, -100, -110, -120, -130, -150, -100, -40];
stimDurs = [1, 1, 1, 1, 1, 1, 1, 1, 0.5, 1.5];
figure()
subplot(3,2,1)
ylabel('V (mV)')
hold on
subplot(3,2,2)
ylabel('[Ca]^{2+} (mM)')
hold on
subplot(3,2,3)
ylabel('ICaL (\muA/\muF)')
hold on
subplot(3,2,4)
ylabel('Jrel (mM/ms)')
hold on
subplot(3,2,5)
ylabel('CaJRS (mM)')
xlabel('Time (ms)')
hold on
subplot(3,2,6)
ylabel('Ta (kPa)')
xlabel('Time (ms)')
hold on
for i = 1:length(stimAmps)
    param.stimAmp = stimAmps(i);
    param.stimDur = stimDurs(i);
    
    % Simulation and extraction of outputs
    [time, X] = modelRunner(X0, options, param, beats, ignoreFirst);

    currents = getCurrentsStructure(time, X, beats, param, 0);
    activeTension = X{1,1}(:,44)*489; % = XS*Tref/dr - only if lambda = 1
    
    subplot(3,2,1);plot(currents.time, currents.V, 'linewidth',1.2);
    subplot(3,2,2);plot(currents.time, currents.Cai,'linewidth',1.2);
    subplot(3,2,3);plot(currents.time, currents.ICaL,'linewidth',1.2);
    subplot(3,2,4);plot(currents.time, currents.Jrel,'linewidth',1.2);
    subplot(3,2,5);plot(currents.time, currents.CaJSR,'linewidth',1.2);
    subplot(3,2,6);plot(currents.time, activeTension,'linewidth',1.2);   
end
l = {};
for i = 1:length(stimAmps)
    l{end+1} = [num2str(stimAmps(i)),' \muA/\muF, dur=',num2str(stimDurs(i)),' ms'];
end
legend(l)
