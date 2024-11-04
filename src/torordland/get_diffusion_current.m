function [Istim] = get_diffusion_current(t, Istim_sf)
% A1 = 2.12315813e+04;
% A2 = 2.25018921e+04;
% mu1 = 1.40727901e+01;
% mu2 = 1.53482735e+01;
% sigma1 = 1.94690948;
% sigma2 = 1.74285262;
% DTI004: 
A1 = 13.66257845;
A2 = 14.14967003; 
mu1 = 14.14969756;
mu2 = 15.288053 ;
sigma1 = 1.94313551;
sigma2 = 1.77886919;

% % Rodero 05:
% A1 = 7.666;
% A2 = 5.455;
% mu1 = 13.11;
% mu2 = 15.47;
% sigma1 = 1.675;
% sigma2 = 2.16;
Istim = Istim_sf* -(A1 * exp(-(t-mu1).^2/(2*sigma1^2)) - A2 * exp(-(t-mu2).^2/(2*sigma2^2)));
% Istim = -(A1 * exp(-(t-mu1).^2/(2*sigma1^2)) - A2 * exp(-(t-mu2).^2/(2*sigma2^2)));
end