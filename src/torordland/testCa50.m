%% Test calcium sensitivity on calcium transient and active tension

%% Setting parameters
clear
clc
close all
param.bcl = 800; % basic cycle length in ms
param.model = @model_ToRORd_Land; % which model is to be used
param.verbose = true; % printing numbers of beats simulated
param.cellType = 1; % 0 endo, 1, epi, 2, mid 

options = []; % parameters for ode15s - usually empty
beats = 200; % number of beats
ignoreFirst = beats - 1; % this many beats at the start of the simulations are ignored when extracting the structure of simulation outputs (i.e., beats - 1 keeps the last beat).

X0 = getStartingState('m_epi'); % starting state 

figure()
subplot(2,1,1)
ylabel('CaT (mM)')
xlabel('Time')
hold on
subplot(2,1,2)
ylabel('Tact (kPa)')
xlabel('Time')
hold on

Ca50s = [0.505,  0.805, 1.105];
for i = 1:length(Ca50s)
    param.Ca50 = Ca50s(i);
    
    % Simulation and extraction of outputs
    [time, X] = modelRunner(X0, options, param, beats, ignoreFirst);

    currents = getCurrentsStructure(time, X, beats, param, 0);
    activeTension = X{1,1}(:,44)*489; % = XS*Tref/dr - only if lambda = 1
    
    subplot(2,1,1);plot(currents.time, currents.Cai, 'linewidth',1.2);
    subplot(2,1,2);plot(currents.time, activeTension,'linewidth',1.2);
end
l = {};
for i = 1:length(Ca50s)
    l{end+1} = ['Ca50: ',num2str(Ca50s(i)),' \muM'];
end
legend(l)

