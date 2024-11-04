% Run for ToR-ORd+Land electro-mechanical model
% (Margara, F., Wang, Z.J., Levrero-Florencio, F., Santiago, A., Vázquez, M., Bueno-Orovio, A.,
% and Rodriguez, B. (2020). In-silico human electro-mechanical ventricular modelling and simulation for
% drug-induced pro-arrhythmia and inotropic risk assessment. Progress in Biophysics and Molecular Biology).

%% Setting parameters
clear

% % Acute BZ1:
% condition = 'bz1';
% param.Ito_Multiplier = 0.1;
% param.IKs_Multiplier = 0.2;
% param.IK1_Multiplier = 0.3;
% param.IKr_Multiplier = 0.7;
% param.INa_Multiplier = 0.4;
% param.ICaL_Multiplier = 0.64;
% param.IKCa_Multiplier = 1.0;

% % Acute BZ2;
% condition = 'bz2';
% param.IKs_Multiplier = 0.2; 
% param.IKr_Multiplier = 0.3;
% param.INa_Multiplier = 0.38;
% param.ICaL_Multiplier = 0.31;
% param.IKCa_Multiplier = 1.0;

% Acute BZ3;
condition = 'bz3';
param.Ito_Multiplier = 0.0;
param.IK1_Multiplier = 0.6;
param.INa_Multiplier = 0.4;
param.ICaL_Multiplier = 0.64;
param.aCaMK_Multiplier = 1.5;
param.taurelp_Multiplier = 6.0; 
param.ICab_Multiplier = 1.33;


param.bcl = 800; % basic cycle length in ms
param.model = @model_ToRORd_Land; % which model is to be used
param.verbose = true; % printing numbers of beats simulated.
celltypes = [0,1,2];
celltype_names = {'endo', 'epi', 'mid'};
for i = 1:3

    param.cellType = celltypes(i); %0 endo, 1 epi, 2 mid
    options = []; % parameters for ode15s - usually empty
    beats = 200; % number of beats
    ignoreFirst = beats - 1; % this many beats at the start of the simulations are ignored when extracting the structure of simulation outputs (i.e., beats - 1 keeps the last beat).
    celltype = celltype_names{i};
    X0 = getStartingState(['m_',celltype]); % starting state - can be also m_mid or m_epi for midmyocardial or epicardial cells respectively.
    
    % Simulation and extraction of outputs
    
    [time, X] = modelRunner(X0, options, param, beats, ignoreFirst);
    
    currents = getCurrentsStructure(time, X, beats, param, 0);
    
    % ActiveTension = X{1, 1}(:,44)*480; % = XS*Tref/dr - only if lambda=1. 
    % add Ta as output in getCurrentsStructure otherwise
    activeTension = X{1,1}(:,44)*480;
    subplot(1,3,1)
    plot(currents.time, currents.V)
    subplot(1,3,2)
    plot(currents.time, currents.Cai)
    subplot(1,3,3)
    plot(currents.time, activeTension) 
    V = currents.V; 
    Cai = currents.Cai;
    Ta = activeTension;
    time = currents.time;
    save([condition,'_',celltype,'_V.mat'], 'V', '-mat');
    save([condition,'_',celltype,'_Cai.mat'], 'Cai', '-mat');
    save([condition,'_',celltype,'_Ta.mat'], 'Ta', '-mat');
    save([condition,'_',celltype,'_time.mat'], 'time', '-mat');
end