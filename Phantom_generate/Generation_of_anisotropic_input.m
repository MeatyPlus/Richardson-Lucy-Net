%% the corresponding PSF can be isotropic (the lateral and axial stepsize should be the same) or anisotropic
%% the corresponding testing data are anisotropic

clear all;

% ground truth path
pathMain_gt= 'C:\Users\LiY\Desktop\simulation_generate\DATA\ground_truth\';
number_of_simulation=100;

%output path
pathMain_no_noise_input= 'C:\Users\LiY\Desktop\simulation_generate\DATA\input_no_noise\';
pathMain_noisy_input= 'C:\Users\LiY\Desktop\simulation_generate\DATA\input_noise\';
pathMain_anisotropic_GT= 'C:\Users\LiY\Desktop\simulation_generate\DATA\anisotropic_GT\';

% The blur function PSF path
PSFpath='C:\Users\LiY\Desktop\simulation_generate\PSF.tif';
PSF = double(ReadTifStack(PSFpath));
% PSF=imresize3(PSF, [128,128,128]);
%% if PSF is anisotropic, use imresize to make it isotropic first
PSF=PSF./sum(PSF(:));

I=double(ReadTifStack([pathMain_gt,'1.tif']));
[Sx, Sy, Sz] = size(I);

PSF= align_size(PSF, Sx, Sy, Sz);
OTF= fftn(ifftshift(PSF));

for i=1:number_of_simulation
    ratio=5;% control the Poisson noise level
   I=double(ReadTifStack([pathMain_gt,num2str(i),'.tif']));
   I=((1+1*rand())*I+thresh)*ratio; % thresh is used to add some background, rand() control the Poisson noise become more random
   B=ConvFFT3_S(I,OTF);
   B1=poissrnd(B);

   % introducing the anisotropic
   sq_ratio=3; %% the sq_ratio=lateral stepsize / axial stepsize;
   nSZ=round(SZ/sq_ratio);
    B1=imresize3(B1, [Sx,Sy,nSZ]);
    %A=imgaussfilt3(K,0.5);
    A=imresize3(A, [Sx,Sy,nSZ]);   

   E=max(B1(:));
   C=B1./E;
   sigma=0.5;%1.0*(rand()+0.1); % Control the level of Gaussian noise, constant or random value
   C=imnoise(C,'gaussian',0,sigma^2/E^2);
   C=C*E;
   
   WriteTifStack(A, [pathMain_anisotropic_GT,num2str(i),'.tif'], 32);
   WriteTifStack(B, [pathMain_no_noise_input,num2str(i),'.tif'], 32);
   WriteTifStack(C, [pathMain_noisy_input,num2str(i),'.tif'], 32);
end