function Stack = ReadTifStack(FileTif)
    InfoImage = imfinfo(FileTif);
    NumberImages = length(InfoImage);
    TifLink = Tiff(FileTif, 'r');
    for i = 1:NumberImages
        TifLink.setDirectory(i);
        Stack(:,:,i) = TifLink.read();
    end
    TifLink.close();
end

%Another faster tiff stack loader
%InfoImage = imfinfo(FileTif);
%mImage = InfoImage(1).Width;
%nImage = InfoImage(1).Height;
%FinalImage=zeros(nImage,mImage,NumberImages,'uint16');
%FileID = tifflib('open',FileTif,'r');
%rps = tifflib('getField',FileID,Tiff.TagID.RowsPerStrip);
 
%for i=1:NumberImages
%   tifflib('setDirectory',FileID,i);
   % Go through each strip of data.
%   rps = min(rps,nImage);
%   for r = 1:rps:nImage
%      row_inds = r:min(nImage,r+rps-1);
%      stripNum = tifflib('computeStrip',FileID,r);
%      FinalImage(row_inds,:,i) = tifflib('readEncodedStrip',FileID,stripNum);
%   end
%end
%tifflib('close',FileID);