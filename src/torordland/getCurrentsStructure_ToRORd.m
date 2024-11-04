function currents = getCurrentsStructure_ToRORd(time, X, beats, parameters, ignoreFirstSpikes)
% Extraction of parameters
cellType = 0; if (isfield(parameters,'cellType')) cellType = parameters.cellType;end

nao = 140;if (isfield(parameters,'nao')) nao = parameters.nao; end
cao = 1.8;if (isfield(parameters,'cao')) cao = parameters.cao; end
ko = 5;if (isfield(parameters,'ko')) ko = parameters.ko; end
ICaL_fractionSS = 0.8; if (isfield(parameters, 'ICaL_fractionSS')) ICaL_fractionSS = parameters.ICaL_fractionSS; end
INaCa_fractionSS = 0.24; if (isfield(parameters, 'INaCa_fractionSS')) INaCa_fractionSS = parameters.INaCa_fractionSS; end

INa_Multiplier = 1; if (isfield(parameters,'INa_Multiplier')) INa_Multiplier = parameters.INa_Multiplier; end
ICaL_Multiplier = 1; if (isfield(parameters,'ICaL_Multiplier')) ICaL_Multiplier = parameters.ICaL_Multiplier; end
Ito_Multiplier = 1; if (isfield(parameters,'Ito_Multiplier')) Ito_Multiplier = parameters.Ito_Multiplier; end
INaL_Multiplier = 1; if (isfield(parameters,'INaL_Multiplier')) INaL_Multiplier = parameters.INaL_Multiplier; end
IKr_Multiplier = 1; if (isfield(parameters,'IKr_Multiplier')) IKr_Multiplier = parameters.IKr_Multiplier; end
IKs_Multiplier = 1; if (isfield(parameters,'IKs_Multiplier')) IKs_Multiplier = parameters.IKs_Multiplier; end
IK1_Multiplier = 1; if (isfield(parameters,'IK1_Multiplier')) IK1_Multiplier = parameters.IK1_Multiplier; end
IKb_Multiplier = 1; if (isfield(parameters,'IKb_Multiplier')) IKb_Multiplier = parameters.IKb_Multiplier; end
IKCa_Multiplier = 1; if (isfield(parameters,'IKCa_Multiplier')) IKCa_Multiplier = parameters.IKCa_Multiplier; end
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
Istim_sf = 1; if (isfield(parameters,'Istim_sf')) Istim_sf = parameters.Istim_sf; end
extraParams = []; if (isfield(parameters,'extraParams')) extraParams = parameters.extraParams; end

vcParameters = []; if (isfield(parameters, 'vcParameters')) vcParameters = parameters.vcParameters; end
apClamp = []; if (isfield(parameters,'apClamp')) apClamp = parameters.apClamp; end

stimAmp = -53; if (isfield(parameters,'stimAmp')) stimAmp = parameters.stimAmp; end
stimDur = 1; if (isfield(parameters,'stimDur')) stimDur = parameters.stimDur; end

if(~isfield(parameters,'model'))
    parameters.model = @model10;
end

vcParameters = []; if (isfield(parameters, 'vcParameters')) vcParameters = parameters.vcParameters; end
apClamp = []; if (isfield(parameters,'apClamp')) apClamp = parameters.apClamp; end

stimAmp = -53; if (isfield(parameters,'stimAmp')) stimAmp = parameters.stimAmp; end
stimDur = 1; if (isfield(parameters,'stimDur')) stimDur = parameters.stimDur; end
Ca50 = 0.805; if (isfield(parameters,'Ca50')) Ca50 = parameters.Ca50; end
trpnmax = 0.07; if (isfield(parameters,'trpnmax')) trpnmax = parameters.trpnmax; end
if(~isfield(parameters,'model'))
    parameters.model = @model;
end

%
% vcParameters = [];
% % specific treatment for voltage clamp; we check that vc parameters are
% % available
% if (isequal(parameters.model, @modelMagyar))
%     try
%         vcParameters = parameters.vcParameters;
%     catch
%         error('All voltage clamp parameters must be specified: vcHoldingPotential,vcHoldingDuration,vcAfterHoldingPotential');
%     end
% end
% preallocation of currents
nPoints = length(cell2mat(time((ignoreFirstSpikes+1):end)));
currents.time = [];
currents.Cai = cell2mat(time((ignoreFirstSpikes+1):end));
currents.Cass = cell2mat(time((ignoreFirstSpikes+1):end));
currents.V = cell2mat(time((ignoreFirstSpikes+1):end));
currents.V2 = cell2mat(time((ignoreFirstSpikes+1):end));
currents.Irel=zeros(nPoints, 1);
currents.INa=zeros(nPoints, 1);
currents.INaL=zeros(nPoints, 1);
currents.Ito=zeros(nPoints, 1);
currents.ICaL_i=zeros(nPoints, 1);
currents.ICaL_ss=zeros(nPoints, 1);
currents.IKr=zeros(nPoints, 1);
currents.IKs=zeros(nPoints, 1);
currents.IK1=zeros(nPoints, 1);
currents.INaCa_i=zeros(nPoints, 1);
currents.INaCa_ss=zeros(nPoints, 1);
currents.INaK=zeros(nPoints, 1);
currents.IKb=zeros(nPoints, 1);
currents.INab=zeros(nPoints, 1);
currents.ICab=zeros(nPoints, 1);
currents.IpCa=zeros(nPoints, 1);
currents.Jdiff=zeros(nPoints, 1);
currents.JdiffNa=zeros(nPoints, 1);
currents.JdiffK=zeros(nPoints, 1);
currents.Jup=zeros(nPoints, 1);
currents.Jleak=zeros(nPoints, 1);
currents.Jtr=zeros(nPoints, 1);
currents.CaMKa=zeros(nPoints, 1);
currents.Istim=zeros(nPoints, 1);
currents.JSRstart = zeros(nPoints, 1);
currents.JSRend = zeros(nPoints, 1);
currents.CaJSR = zeros(nPoints, 1);
currents.CaNSR = zeros(nPoints, 1);
currents.PhiCaL = zeros(nPoints, 1);
currents.openChannels = zeros(nPoints, 1);
currents.caLeavingNCX = zeros(nPoints, 1);
currents.caLeavingJup = zeros(nPoints, 1);
currents.caLeavingPCa = zeros(nPoints, 1);
currents.IClbk = zeros(nPoints, 1);
currents.IClCa = zeros(nPoints, 1);

currents.ICaL_d = cell2mat(time((ignoreFirstSpikes+1):end));
currents.ICaL_ff = cell2mat(time((ignoreFirstSpikes+1):end));
currents.ICaL_fs = cell2mat(time((ignoreFirstSpikes+1):end));
currents.ICaL_fcaf = cell2mat(time((ignoreFirstSpikes+1):end));
currents.ICaL_fcas = cell2mat(time((ignoreFirstSpikes+1):end));
currents.ICaL_jca = cell2mat(time((ignoreFirstSpikes+1):end));
currents.ICaL_nca = cell2mat(time((ignoreFirstSpikes+1):end));
currents.ICaL_fICaLP = cell2mat(time((ignoreFirstSpikes+1):end));
currents.ICaL_PhiCaLP = cell2mat(time((ignoreFirstSpikes+1):end));

i = 1;

modelNow = parameters.model;

for iBeat = 1:size(time, 1)
    if (iBeat <= ignoreFirstSpikes)
        continue
    end
    currents.time = [currents.time; (iBeat-1)*parameters.bcl + time{iBeat}];
    Xhere = X{iBeat};
    for j=1:size(X{iBeat},1)
        
        
        
        
        IsJs=modelNow(time{iBeat}(j),X{iBeat}(j,:),0, cellType, ICaL_Multiplier, ...
            INa_Multiplier, Ito_Multiplier, INaL_Multiplier, IKr_Multiplier, IKs_Multiplier, IK1_Multiplier,IKCa_Multiplier,IKb_Multiplier,INaCa_Multiplier,...
            INaK_Multiplier, INab_Multiplier, ICab_Multiplier, IpCa_Multiplier, ICaCl_Multiplier, IClb_Multiplier, Jrel_Multiplier,Jup_Multiplier,...
            aCaMK_Multiplier,taurelp_Multiplier,kws_Multiplier, kuw_Multiplier, ksu_Multiplier, ca50_Multiplier, nao,cao,ko,ICaL_fractionSS,...
            INaCa_fractionSS, stimAmp, stimDur,Istim_sf, trpnmax, vcParameters, apClamp, extraParams);
        
        currents.V(i) = X{iBeat}(j,1);
        currents.Cai(i) = X{iBeat}(j,6);
        currents.Cass(i) = X{iBeat}(j,7);
        currents.Jrel(i)=IsJs(21);
        currents.INa(i)=IsJs(1);
        currents.INaL(i)=IsJs(2);
        currents.Ito(i)=IsJs(3);
        currents.ICaL(i)=IsJs(4);
        currents.IKr(i)=IsJs(5);
        currents.IKs(i)=IsJs(6);
        currents.IK1(i)=IsJs(7);
        currents.INaCa_i(i)=IsJs(8);
        currents.INaCa_ss(i)=IsJs(9);
        currents.INaK(i)=IsJs(10);
        currents.IKb(i)=IsJs(11);
        currents.INab(i)=IsJs(12);
        currents.ICab(i)=IsJs(13);
        currents.IpCa(i)=IsJs(14);
        currents.Jdiff(i)=IsJs(15);
        currents.JdiffNa(i)=IsJs(16);
        currents.JdiffK(i)=IsJs(17);
        currents.Jup(i)=IsJs(18);
        currents.Jleak(i)=IsJs(19);
        currents.Jtr(i)=IsJs(20);
        
        currents.CaMKa(i)=IsJs(22);
        currents.Istim(i)=IsJs(23);
        
        
        if (length(IsJs) >= 24)
            currents.fINap(i) = IsJs(24);
            currents.fINaLp(i) = IsJs(25);
            currents.fICaLp(i) = IsJs(26);
            currents.fJrelp(i) = IsJs(27);
            currents.fJupp(i) = IsJs(28);
            
            currents.CaJSR(i) = IsJs(29);
            currents.CaNSR(i) = IsJs(30);
            currents.PhiCaL_ss(i) = IsJs(31);
        end
        if (length(IsJs) >= 32)
            currents.V(i) = IsJs(32);
        end
        if (length(IsJs) >=33) 
            currents.ICaL_i(i) = IsJs(33);
            currents.IClCa(i) = IsJs(34);
            currents.IClbk(i) = IsJs(35);        
            currents.ICaL_tot(i) = IsJs(36);
        end
        
        i = i + 1; % incrementing the index where everything is stored
    end
end

currents.INaCa = currents.INaCa_i+currents.INaCa_ss;
end

