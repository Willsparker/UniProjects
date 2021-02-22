%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compression / Decompression Coursework %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Original code.

clc;
clear all;
close all;
warning off; % If it compiles, I'm not gonna be picky, dammit

%---------- DCT Compression
% Help from: https://www.mathworks.com/help/images/discrete-cosine-transform.html

original_image=rgb2gray(imresize(imread('images/IC2.png'),0.5));

block_size=8;
% An error is flagged if the image size is not divisible by the block size
% to create an integer. i.e. 1512x2016 (the image at half size), is visible
% by 8, but not by 16

DTC_image = im2double(original_image);
dctMatrix = dctmtx(block_size); % Make DCT Matrix of size block_size
dct_func = @(block_struct) dctMatrix * block_struct.data * dctMatrix';

% blockproc will break up DTC_image into blocks, and apply 'dct_func' to
% each block
B = blockproc(DTC_image,[block_size block_size], dct_func);

% Removes 1 - (1/blocksize^2) % of coefficients
% i.e blocksize = 8 therefore removes ~95% of DCT coefficients
mask = zeros(block_size);
mask(1,1)=1; 

% Apply the mask to each block in the image
B2 = blockproc(B,[block_size block_size],@(block_struct) mask * block_struct.data);
% Inverse DCT blockproc function
invdct = @(block_struct) dctMatrix' * block_struct.data * dctMatrix;
% Apply the inverse dctf, so we can see the image
compress_img = blockproc(B2,[block_size block_size], invdct);

%------------ Huffman encoding

[m,n]=size(B2);                                 % Get size of original image
Totalcount=m*n;
cnt=1;
for i=0:255                                     % Find probability of each
  k=uint8(B2)==i;                               % Grey level
  count(cnt)=sum(k(:));
  pro(cnt)=count(cnt)/Totalcount;
  cnt=cnt+1;
end

symbols = [0:255];                              % Only 256 greyvalues
dict = huffmandict(symbols,pro);                % Build the dictionary

B2_vector_string=uint8(B2(:));                  % Convert to Uint8 & Vector
comp1 = huffmanenco(B2_vector_string,dict);     % Huffman Encode!

%------------ Save to file

matlab.io.saveVariablesToScript('compressed_images/dct_huffman_image.m',{'comp1','dict','m','n'})

%------------ Decode Huffman vector

decomp1 = uint8(huffmandeco(comp1,dict));       % recode into vector
decoded_array=reshape(decomp1,m,n);             % Reshape array

%------------ Reapply inverse DCT & Show the results

decoded_compressed_img = blockproc(B2,[block_size block_size], invdct);

figure
subplot(2,2,1), imshow(original_image), title('Original Image')
subplot(2,2,2), imshow(compress_img), title('IMG after Compression')
subplot(2,2,3), imshow(decoded_compressed_img), title('Compressed IMG after Huffman')


%----------- Writing images in PNG, for the GUI

a =  decoded_compressed_img;
b =  compress_img;

imwrite(b,'compressed_images/dct.png');
imwrite(a,'compressed_images/dct_huffman_decoded.png');



