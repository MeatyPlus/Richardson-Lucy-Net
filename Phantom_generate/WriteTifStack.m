function WriteTifStack(Stack, Filename, BitsPerSample)

    if nargin < 3
        BitsPerSample = 16;
    end
        
   t = Tiff(Filename, 'w');
   tagstruct.ImageLength = size(Stack, 1);
   tagstruct.ImageWidth = size(Stack, 2);
   tagstruct.Compression = Tiff.Compression.None;
   %tagstruct.Compression = Tiff.Compression.LZW;        % compressed
   
   tagstruct.Photometric = Tiff.Photometric.MinIsBlack;   
   tagstruct.Software = 'MATLAB';
   tagstruct.BitsPerSample =  BitsPerSample; 
   if tagstruct.BitsPerSample == 32
       tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
       Stack = single(Stack);
   else 
       tagstruct.SampleFormat = Tiff.SampleFormat.UInt;
       Stack = uint16(Stack);
   end
   %tagstruct.RowsPerStrip = 256;
   tagstruct.SamplesPerPixel = 1;
   tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
   for k = 1:size(Stack, 3)
       t.setTag(tagstruct)
       t.write(Stack(:, :, k));
       t.writeDirectory();
   end
   t.close();
end