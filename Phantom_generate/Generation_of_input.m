clear all;

% ground truth path
pathMain_gt= 'C:\Users\LiY\Desktop\simulation_generate\DATA\ground_truth\';
number_of_simulation=100;

%output path
pathMain_no_noise_input= 'C:\Users\LiY\Desktop\simulation_generate\DATA\input_no_noise\';
pathMain_noisy_input= 'C:\Users\LiY\Desktop\simulation_generate\DATA\input_noise\';

% The blur function PSF path
PSFpath='C:\Users\LiY\Desktop\simulation_generate\PSF.tif';

I=double(ReadTifStack([pathMain_gt,'1.tif']));
[Sx, Sy, Sz] = size(I);
PSF = double(ReadTifStack(PSFpath));
PSF=PSF./sum(PSF(:));
PSF= align_size(PSF, Sx, Sy, Sz);
OTF= fftn(ifftshift(PSF));

for i=1:number_of_simulation
    ratio=5;% control the Poisson noise level
   I=double(ReadTifStack([pathMain_gt,num2str(i),'.tif']))*ratio;
   B=ConvFFT3_S(I,OTF);
   B1=poissrnd(B);
   E=max(B1(:));
   C=B1./E;
   sigma=0.5;%1.0*(rand()+0.1); % Control the level of Gaussian noise, constant or random value
   C=imnoise(C,'gaussian',0,sigma^2/E^2);
   C=C*E;

   WriteTifStack(B, [pathMain_no_noise_input,num2str(i),'.tif'], 32);
   WriteTifStack(C, [pathMain_noisy_input,num2str(i),'.tif'], 32);
end