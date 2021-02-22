clc;
clear all;
close all;
warning off; % If it compiles, I'm not gonna be picky, dammit

%---------- DCT Compression
% Help from: https://www.mathworks.com/help/images/discrete-cosine-transform.html

original_image=rgb2gray(imresize(imread('images/IC2.png'),0.5));

block_size=18;
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

% mask = [ 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
%          1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0
%          1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0
%          1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0
%          1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0
%          1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
%          1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
%          1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
%          1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
%          1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
%          1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
%          1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0
%          1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0
%          1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
%          1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
%          1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
%          1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
%          1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ];


% Apply the mask to each block in the image
B2 = blockproc(B,[block_size block_size],@(block_struct) mask * block_struct.data);
% Inverse DCT blockproc function
invdct = @(block_struct) dctMatrix' * block_struct.data * dctMatrix;
% Apply the inverse dctf, so we can see the image
compress_img = blockproc(B2,[block_size block_size], invdct);

imwrite(compress_img, 'compressed_images/dct1818_99.png')
