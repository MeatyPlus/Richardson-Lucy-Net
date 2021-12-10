function [outVol] = ConvFFT3_S(inVol,OTF)

% N = size(inVol);
% M = size(OTF);
% if N == M
    outVol = single(real(ifftn(fftn(inVol).*OTF)));  
% else 
%     Q = max(N,M);
%     inVolPad = zeros([Q(1),Q(2),Q(3)],'single','gpuArray') + 0.001;
%     yrange = (M(1)-N(1))/2+1:(M(1)-N(1))/2+N(1);
%     xrange = (M(2)-N(2))/2+1:(M(2)-N(2))/2+N(2);
%     zrange = (M(3)-N(3))/2+1:(M(3)-N(3))/2+N(3); 
%     inVolPad(yrange,xrange,zrange) = inVol;
%     outVolPad = single(real(ifftn(fftn(inVolPad).*OTF)));  
%     outVol = outVolPad(yrange,xrange,zrange);
% end

end
 



