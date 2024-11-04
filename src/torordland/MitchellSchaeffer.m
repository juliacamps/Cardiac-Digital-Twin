function [dydt] = MitchellSchaeffer(t, vm_h, apd)
 stim_amp = 0.05;
stim_dur = 5;
vm = vm_h(1);
h = vm_h(2);
tau_out = 5.4;
tau_in = 0.3;
tau_open = 80;
tau_close = apd / log(tau_out / (2.9 * tau_in));  % Gilette paper has 4*tau_in, this gave us larger apd errors than 2.9.
V_min = -86.2;
V_max = 40.0;
vm_gate = 0.1;  % Original Mitchel Schaeffer paper has 0.13 for Vm,gate, we changed this to get better apd match.
Cm = 1;
if vm < vm_gate
    dhdt = (1 - h) / tau_open;
else
    dhdt = -h / tau_close;
end
J_in = h * vm ^ 2 * (1 - vm) / tau_in;
J_out = - vm / tau_out;
I_ion = (J_in + J_out);
amp = stim_amp;
duration = stim_dur;
if t <= duration
    Istim = amp;
else
    Istim = 0.0;
end
dVmdt = I_ion + Istim;
dydt = [dVmdt, dhdt]'; 
end