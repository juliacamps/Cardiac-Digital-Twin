function [ time, X, parameters ] = modelRunner_ToRORd( X0, options, parameters, beats, ignoreFirst)

% Parameters are set here
% defaults which may be overwritten

cellType = 0; if (isfield(parameters,'cellType')) cellType = parameters.cellType;end

% if the number of simulated beat is to be printed out.
verbose = false; if (isfield(parameters,'verbose')) verbose = parameters.verbose; end

nao = 140; if (isfield(parameters,'nao')) nao = parameters.nao; end
cao = 1.8;if (isfield(parameters,'cao')) cao = parameters.cao; end
ko = 5;if (isfield(parameters,'ko')) ko = parameters.ko; end
ICaL_fractionSS = 0.8; if (isfield(parameters, 'ICaL_fractionSS')) ICaL_fractionSS = parameters.ICaL_fractionSS; end
INaCa_fractionSS = 0.35; if (isfield(parameters, 'INaCa_fractionSS')) INaCa_fractionSS = parameters.INaCa_fractionSS; end

INa_Multiplier = 1; if (isfield(parameters,'INa_Multiplier')) INa_Multiplier = parameters.INa_Multiplier; end
ICaL_Multiplier = 1; if (isfield(parameters,'ICaL_Multiplier')) ICaL_Multiplier = parameters.ICaL_Multiplier; end
Ito_Multiplier = 1; if (isfield(parameters,'Ito_Multiplier')) Ito_Multiplier = parameters.Ito_Multiplier; end
INaL_Multiplier = 1; if (isfield(parameters,'INaL_Multiplier')) INaL_Multiplier = parameters.INaL_Multiplier; end
IKr_Multiplier = 1; if (isfield(parameters,'IKr_Multiplier')) IKr_Multiplier = parameters.IKr_Multiplier; end
IKs_Multiplier = 1; if (isfield(parameters,'IKs_Multiplier')) IKs_Multiplier = parameters.IKs_Multiplier; end
IK1_Multiplier = 1; if (isfield(parameters,'IK1_Multiplier')) IK1_Multiplier = parameters.IK1_Multiplier; end
IKCa_Multiplier = 0; if (isfield(parameters,'IKCa_Multiplier')) IKCa_Multiplier = parameters.IKCa_Multiplier; end
IKb_Multiplier = 1; if (isfield(parameters,'IKb_Multiplier')) IKb_Multiplier = parameters.IKb_Multiplier; end
INaCa_Multiplier = 1; if (isfield(parameters,'INaCa_Multiplier')) INaCa_Multiplier = parameters.INaCa_Multiplier; end
INaK_Multiplier = 1; if (isfield(parameters,'INaK_Multiplier')) INaK_Multiplier = parameters.INaK_Multiplier; end
INab_Multiplier = 1; if (isfield(parameters,'INab_Multiplier')) INab_Multiplier = parameters.INab_Multiplier; end
ICab_Multiplier = 1; if (isfield(parameters,'ICab_Multiplier')) ICab_Multiplier = parameters.ICab_Multiplier; end
IpCa_Multiplier = 1; if (isfield(parameters,'IpCa_Multiplier')) IpCa_Multiplier = parameters.IpCa_Multiplier; end
ICaCl_Multiplier = 1; if (isfield( parameters, 'ICaCl_Multiplier')) ICaCl_Multiplier =  parameters.ICaCl_Multiplier; end
IClb_Multiplier = 1; if (isfield( parameters, 'IClb_Multiplier')) IClb_Multiplier =  parameters.IClb_Multiplier; end
Jrel_Multiplier = 1; if (isfield(parameters,'Jrel_Multiplier')) Jrel_Multiplier = parameters.Jrel_Multiplier; end
Jup_Multiplier = 1; if (isfield(parameters,'Jup_Multiplier')) Jup_Multiplier = parameters.Jup_Multiplier; end
aCaMK_Multiplier = 1; if (isfield(parameters,'aCaMK_Multiplier')) aCaMK_Multiplier = parameters.aCaMK_Multiplier; end
taurelp_Multiplier = 1; if (isfield(parameters,'taurelp_Multiplier')) taurelp_Multiplier = parameters.taurelp_Multiplier; end
kws_Multiplier = 1; if (isfield(parameters,'kws_Multiplier')) kws_Multiplier = parameters.kws_Multiplier; end
kuw_Multiplier = 1; if (isfield(parameters,'kuw_Multiplier')) kuw_Multiplier = parameters.kuw_Multiplier; end
ksu_Multiplier = 1; if (isfield(parameters,'ksu_Multiplier')) ksu_Multiplier = parameters.ksu_Multiplier; end
ca50_Multiplier = 1; if (isfield(parameters,'ca50_Multiplier')) ca50_Multiplier = parameters.ca50_Multiplier; end
extraParams = []; if (isfield(parameters,'extraParams')) extraParams = parameters.extraParams; end
Istim_sf = 1; if (isfield(parameters,'Istim_sf')) Istim_sf = parameters.Istim_sf;end 
vcParameters = []; if (isfield(parameters, 'vcParameters')) vcParameters = parameters.vcParameters; end
apClamp = []; if (isfield(parameters,'apClamp')) apClamp = parameters.apClamp; end

stimAmp = -53; if (isfield(parameters,'stimAmp')) stimAmp = parameters.stimAmp; end
stimDur = 1; if (isfield(parameters,'stimDur')) stimDur = parameters.stimDur; end

% Ca50 = 0.805; if (isfield(parameters,'Ca50')) Ca50 = parameters.Ca50; end
trpnmax = 0.07; if (isfield(parameters,'trpnmax')) trpnmax = parameters.trpnmax; end 

if(~isfield(parameters,'model'))
    parameters.model = @model11;
end


CL = parameters.bcl;
time = cell(beats,1);
X = cell(beats, 1);
for n=1:beats
    if (verbose)
        disp(['Beat = ' num2str(n)]);
    end
    if ((exist('ode15sTimed')) == 2) % if timed version provided, it is preferred
        [time{n}, X{n}]=ode15sTimed(parameters.model,[0 CL],X0,options,1,  cellType, ICaL_Multiplier, ...
            INa_Multiplier, Ito_Multiplier, INaL_Multiplier, IKr_Multiplier, IKs_Multiplier, IK1_Multiplier, IKCa_Multiplier, IKb_Multiplier,INaCa_Multiplier,...
            INaK_Multiplier, INab_Multiplier, ICab_Multiplier, IpCa_Multiplier, ICaCl_Multiplier, IClb_Multiplier, Jrel_Multiplier,Jup_Multiplier,aCaMK_Multiplier,...
            taurelp_Multiplier,kws_Multiplier, kuw_Multiplier, ksu_Multiplier, ca50_Multiplier, nao,cao,ko,ICaL_fractionSS,INaCa_fractionSS, stimAmp, stimDur, Istim_sf, trpnmax,vcParameters, apClamp, extraParams);
        
    else
        [time{n}, X{n}]=ode15s(parameters.model,[0 CL],X0,options,1,  cellType, ICaL_Multiplier, ...
            INa_Multiplier, Ito_Multiplier, INaL_Multiplier, IKr_Multiplier, IKs_Multiplier, IK1_Multiplier, IKCa_Multiplier, IKb_Multiplier,INaCa_Multiplier,...
            INaK_Multiplier, INab_Multiplier, ICab_Multiplier, IpCa_Multiplier, ICaCl_Multiplier, IClb_Multiplier, Jrel_Multiplier,Jup_Multiplier,aCaMK_Multiplier,...
            taurelp_Multiplier,kws_Multiplier, kuw_Multiplier, ksu_Multiplier, ca50_Multiplier, nao,cao,ko,ICaL_fractionSS,INaCa_fractionSS, stimAmp, stimDur, Istim_sf, trpnmax, vcParameters, apClamp, extraParams);
        
    end
    if isequal(time{n}, -1) % unfinished (unstable) computation - we end here.
        try
            time(1:ignoreFirst) = [];
            X(1:ignoreFirst) = [];
        catch
            time = [];
            X = [];
        end
        parameters.isFailed = 1;
        return;
    end
    
    X0=X{n}(size(X{n},1),:);
    
    %n %output beat number to the screen to monitor runtime progress
end

time(1:ignoreFirst) = [];
X(1:ignoreFirst) = [];
end

