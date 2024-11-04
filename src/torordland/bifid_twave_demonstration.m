% Demonstrate effect of cell AP morphology on T wave morphology: bifid T
% waves due to change in gradients in ToRORd model 
close all
param.bcl = 800;
param.model = @model_ToRORd_step_stimulus;

param.verbose = false;
param.stimAmp = -53;
options = [];
beats = 100;
ignoreFirst = beats-1;

celltype = 'endo';
X0 = getStartingState_ToRORd(['m_', celltype]);

if strcmp(celltype, 'endo')
    param.cellType = 0;
elseif strcmp(celltype, 'epi')
    param.cellType = 1;
elseif strcmp(celltype, 'mid')
    param.cellType = 2;
end

[time, X] = modelRunner_ToRORd(X0, options, param, beats, ignoreFirst);
currents = getCurrentsStructure_ToRORd(time, X, beats, param, 0);
vm = currents.V; 
t = currents.time;

% Show first derivative
dvdt = zeros(1, length(vm));
for i = 1:length(vm) - 2 
    dvdt(i) = (vm(i+2) - vm(i))/(t(i+2) - t(i));
end

figure;
plot(t, dvdt);
xlim([20, 800]);
title('ToRORd dVdt');

% Resample AP to 1 ms resolution 
t_end = max(t);
t_1ms = 0:t_end;
vm_resampled = interp1(t, vm, t_1ms, 'linear');
vm = vm_resampled;
t = t_1ms;
% idx = knnsearch(t, t_1ms');
% vm_1ms = zeros(size(vm));
% for i = 1:length(idx)
%     vm_1ms = (vm(idx[i]-idx[i]-1)
% vm = vm(idx);
% t = t_1ms';

shift_idx = 20; % [ms]
shifted_vm = zeros(size(vm));
shifted_vm(shift_idx:end) = vm(1:end-(shift_idx-1));
shifted_vm(1:shift_idx) = vm(1);

figure;
plot(t, vm, t, shifted_vm, t, shifted_vm - vm);
legend('Vm', 'Shifted Vm', 'shifted vm - vm')
title('ToRORd Unipolar Electrogram');

% Show that the camel hump is missing in the Mitchell Schaefer model 
CL = 800;
y0 = [0, 1]; % Starting states for Vm and h gating
apd = 300;
options = [];
t = 0:CL;
[time, X]= ode15s(@MitchellSchaeffer, t, y0, options, apd);
vm = X(:,1);
shift_idx = 20;
shifted_vm = zeros(size(vm));
shifted_vm(shift_idx:end) = vm(1:end-(shift_idx-1));
shifted_vm(1:shift_idx) = vm(1);
figure;
plot(time, vm, time, shifted_vm, time, shifted_vm - vm);
legend('Vm', 'Shifted Vm', 'shifted vm - vm')
title('Mitchell-Schaeffer Unipolar Electrogram');
        
% Show first derivative
dvdt = zeros(1, length(vm));
for i = 1:length(vm) - 2 
    dvdt(i) = (vm(i+2) - vm(i))/(t(i+2) - t(i));
end

figure;
plot(t, dvdt);
xlim([20, 800]);
title('Mitchell-Schaeffer dVdt');