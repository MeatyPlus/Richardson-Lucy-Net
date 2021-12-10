

clear all

% SET OUTPUT PATH
pathMain1= 'C:\Users\LiY\Desktop\simulation_generate\DATA\ground_truth\';

% WITH BACKGROUND, for applying simulated data to biology sample
% you can choose True
is_with_background=true;

% set the Gaussian blur parameter
delta=0.7;

% How many simulation to generate
number_of_simulation=100;

% Set numbers of strctures
n_spheres=100;
n_ellipsoidal=100;
n_dots=400/8;




%% creat Gaussian filter
Ggrid = -floor(5/2):floor(5/2);
[X Y Z] = meshgrid(Ggrid, Ggrid, Ggrid);

% Create Gaussian Mask
GaussM = exp(-(X.^2 + Y.^2 + Z.^2) / (2*delta^2));

% Normalize so that total area (sum of all weights) is 1
GaussM = GaussM/sum(GaussM(:));

%%   spheroid
for tt=1:number_of_simulation
    
    A=zeros(128,128,128);
    B=zeros(128,128,128);
    
    for times=1:n_spheres
        x=floor(110*rand()+9);
        y=floor(110*rand()+9);
        z=floor(110*rand()+9);
        
        r=floor(4*rand()+4);
        
        inten=800*rand()+50;
        
        
        for i=(x-r):(x+r)
            for j=(y-r):(y+r)
                for k=(z-r):(z+r)
                    
                    if(((i-x)^2+(j-y)^2+(k-z)^2)<=(r)^2)
                        A(i,j,k)=inten;
                    end
                end
            end
        end
    end
    
    for times=1:n_ellipsoidal
        x=floor(110*rand()+9);
        y=floor(110*rand()+9);
        z=floor(110*rand()+9);
        
        r1=floor(4*rand()+4);
        r2=floor(4*rand()+4);
        r3=floor(4*rand()+4);
        
        inten=800*rand()+50;
        
        for i=(x-r1):(x+r1)
            for j=(y-r2):(y+r2)
                for k=(z-r3):(z+r3)
                    if((((i-x)^2)/r1^2+((j-y)^2)/r2^2+((k-z)^2)/r3^2)<=1.3 && (((i-x)^2)/r1^2+((j-y)^2)/r2^2+((k-z)^2)/r3^2)>=0.8)
                        A(i,j,k)=inten;
                    end
                end
            end
        end
    end
    
    
    for times=1:n_dots
        x=floor(125*rand()+1);
        y=floor(125*rand()+1);
        z=floor(125*rand()+1);
        
        r=1;
        
        inten=800*rand()+50;
        
        A(x:x+1,y:y+1,z:z+1)=inten;
        
    end
    
    for times=1:n_dots
        x=floor(115*rand()+1);
        y=floor(125*rand()+1);
        z=floor(125*rand()+1);
        
        r=1;
        
        inten=800*rand()+50;
        k=floor(rand()*9)+1;
        
        A(x:x+k,y:y+1,z:z+1)=inten;
    end
    
    for times=1:n_dots
        x=floor(125*rand()+1);
        y=floor(115*rand()+1);
        z=floor(125*rand()+1);
        
        r=1;
        
        inten=800*rand()+50;
        
        k=floor(rand()*9)+1;
        
        A(x:x+1,y:y+k,z:z+1)=inten+50*rand();
    end
    
    for times=1:n_dots
        x=floor(125*rand()+1);
        y=floor(125*rand()+1);
        z=floor(115*rand()+1);
        
        r=1;
        
        inten=800*rand()+50;
        k=floor(rand()*9)+1;
        
        A(x:x+1,y:y+1,z:z+k)=inten;
        
    end
    
    for times=1:n_dots
        x=floor(115*rand()+1);
        y=floor(125*rand()+1);
        z=floor(115*rand()+1);
        
        r=1;
        
        inten=800*rand()+50;
        k1=floor(rand()*9)+1;
        k2=floor(rand()*9)+1;
        
        A(x:x+k1,y:y+1,z:z+k2)=inten;
        
    end
    
    for times=1:n_dots
        x=floor(115*rand()+1);
        y=floor(115*rand()+1);
        z=floor(125*rand()+1);
        
        r=1;
        
        inten=800*rand()+50;
        
        k1=floor(rand()*9)+1;
        k2=floor(rand()*9)+1;
        A(x:x+k1,y:y+k2,z:z+1)=inten;
        
    end
    
    for times=1:n_dots
        x=floor(125*rand()+1);
        y=floor(115*rand()+1);
        z=floor(115*rand()+1);
        
        r=1;
        
        inten=800*rand()+50;
        k1=floor(rand()*9)+1;
        k2=floor(rand()*9)+1;
        A(x:x+1,y:y+k1,z:z+k2)=inten;
        
    end
    
    for times=1:n_dots
        x=floor(120*rand()+1);
        y=floor(120*rand()+1);
        z=floor(120*rand()+1);
        
        r=1;
        
        inten=800*rand()+50;
        k1=floor(rand()*5)+1;
        k2=floor(rand()*5)+1;
        k3=floor(rand()*5)+1;
        
        A(x:x+k1,y:y+k2,z:z+k3)=inten;
        
    end
    
    %WriteTifStack(A, [pathMain1,num2str(tt),'.tif'], 32);
    if is_with_background
        A=A+30;
    end
    
    
    A = convn(A, GaussM, 'same');
    
    WriteTifStack(A, [pathMain1,num2str(tt),'.tif'], 32);
end


