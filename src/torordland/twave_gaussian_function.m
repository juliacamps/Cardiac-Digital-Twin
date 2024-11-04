function y = twave_gaussian_function(t, t_Tend, t_Tpeak, v_Tpeak)
%t = t_Tstart*0.5:0.001:(t_Tpeak - t_Tstart)*2.5+t_Tstart;
%t = 0:0.1:10;
% t  = 0: 0.1: t_Tpeak*2;
y = v_Tpeak * gaussmf(t, [(t_Tend-t_Tpeak)/2, t_Tpeak]);
%y = gaussmf(t, [2 5]);
%y  = v_Tpeak * gaussmf(t, [0+t_Tstart*0.8, (t_Tpeak-t_Tstart)]);
% plot(t,y)
end