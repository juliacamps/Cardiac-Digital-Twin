% Run for ToR-ORd+Land electro-mechanical model
% (Margara, F., Wang, Z.J., Levrero-Florencio, F., Santiago, A., Vázquez, M., Bueno-Orovio, A.,
% and Rodriguez, B. (2020). In-silico human electro-mechanical ventricular modelling and simulation for
% drug-induced pro-arrhythmia and inotropic risk assessment. Progress in Biophysics and Molecular Biology).

%% Setting parameters
clear
close all
param.bcl = 800; % basic cycle length in ms
param.model = @model_ToRORd_Land_diffusion_current; % which model is to be used
param.verbose = true; % printing numbers of beats simulated.
figure;
hold on

param.cellType = 0; %0 endo, 1 epi, 2 mid
options = []; % parameters for ode15s - usually empty
beats = 50; % number of beats
ignoreFirst = beats - 1; % this many beats at the start of the simulations are ignored when extracting the structure of simulation outputs (i.e., beats - 1 keeps the last beat).
celltype = 'endo';
X0 = getStartingState(['m_',celltype]); % starting state - can be also m_mid or m_epi for midmyocardial or epicardial cells respectively.

% Simulation and extraction of outputs

[time, X] = modelRunner_ToRORd(X0, options, param, beats, ignoreFirst);

currents = getCurrentsStructure_ToRORd(time, X, beats, param, 0);

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


