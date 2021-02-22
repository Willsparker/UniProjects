% Huffman encoding of the original picture
% Original code.

clc;
clear all;
close all;
warning off; % If it compiles, I'm not gonna be picky, dammit

%------------ Huffman encoding

image=imread('images/original_img_half_grey.png');     

[m,n]=size(image);                                 % Get size of original image
Totalcount=m*n;
cnt=1;
for i=0:255                                        % Find probability of each
  k=uint8(image)==i;                               % Grey level
  count(cnt)=sum(k(:));
  pro(cnt)=count(cnt)/Totalcount;
  cnt=cnt+1;
end

symbols = [0:255];                              % Only 256 greyvalues
dict = huffmandict(symbols,pro);                % Build the dictionary

vector_string=uint8(image(:));                  % Convert to Uint8 & Vector
comp1 = huffmanenco(vector_string,dict);        % Huffman Encode!

%------------ Save to file

matlab.io.saveVariablesToScript('compressed_images/huffman_image.m',{'comp1','dict','m','n'})

%------------ Decode Huffman vector

decomp1 = uint8(huffmandeco(comp1,dict));       % recode into vector
decoded_array=reshape(decomp1,m,n);             % Reshape array
isequal(image,decoded_array);                   % Ensure they're equal

