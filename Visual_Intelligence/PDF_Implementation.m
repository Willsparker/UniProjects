clear all;
load('ground_truth.mat')

% Change this for testing
TARGET_SAMPLE_SIZE=20;

random_indicies = zeros(4,TARGET_SAMPLE_SIZE);

% Note, this is randomly getting the points - which may not be the best
% for getting the best probabilities, but it's better than it being a 
% 1.3k line file.

% For each class (row)
for class = 1:4
    % Get list of indexes where the labelled_ground_truth is equal 1-4
    result = find(labelled_ground_truth==class);
    % For TARGET_SAMPLE_SIZE inputs (col)
    for indexCount = 1:TARGET_SAMPLE_SIZE
        % Pick a random value from that list
        rand_index = randperm(length(result),1);
        % Add it to 4xTARGET_SAMPLE_SIZE array
        random_indicies(class,indexCount) = result(rand_index);
    end
end

% Go through each 6 images
r_img = imread('images/r.bmp');
nir_img = imread('images/nir.bmp');
le_img = imread('images/le.bmp');
g_img = imread('images/g.bmp');
fe_img = imread('images/fe.bmp');
b_img = imread('images/b.bmp');

% X axis/rows = class; 
% Y axis/columns = sample per class; 
% Z axis = The values of the pixel in each picture.
results_matrix = zeros(4,TARGET_SAMPLE_SIZE,6);

% For each row in the random_indicies matrix (1-4)
for row=1:size(random_indicies,1)
    % for each column in the random_indicies matrix (1-TARGET_SAMPLE_SIZE)
    for col=1:size(random_indicies,2)
        results_matrix(row,col,1) = r_img(random_indicies(row,col));
        results_matrix(row,col,2) = nir_img(random_indicies(row,col));
        results_matrix(row,col,3) = le_img(random_indicies(row,col));
        results_matrix(row,col,4) = g_img(random_indicies(row,col));
        results_matrix(row,col,5) = fe_img(random_indicies(row,col));
        results_matrix(row,col,6) = b_img(random_indicies(row,col));
    end
end

% This is a better 'implementation' for getting the mean and covariance
% list, but is a bit hard to read, hence commented out
%for classIndex=1:4
%    mean_list(:,classIndex) = mean(squeeze(results_matrix(classIndex,:,:)));
%    cov_list(:,:,classIndex) = cov(squeeze(results_matrix(classIndex,:,:)));
%end

% Class 1 - Building
class_1 = squeeze(results_matrix(1,:,:));
mean_list(:,1) = mean(class_1);
cov_list(:,:,1) = cov(class_1);

% Class 2 - Vegetation
class_2 = squeeze(results_matrix(2,:,:));
mean_list(:,2) = mean(class_2);
cov_list(:,:,2) = cov(class_2);

% Class 3 - Car
class_3 = squeeze(results_matrix(3,:,:));
mean_list(:,3) = mean(class_3);
cov_list(:,:,3) = cov(class_3);

% Class 4 - Ground
class_4 = squeeze(results_matrix(4,:,:));
mean_list(:,4) = mean(class_4);
cov_list(:,:,4) = cov(class_4);

new_image = zeros(size(r_img,1),size(r_img,2));
% For all dimensions of picture
for row = 1:size(r_img,1)
    for col = 1:size(r_img,2)
        pixel_value = [];
        pixel_value(1) = r_img(row,col);
        pixel_value(2) = nir_img(row,col);
        pixel_value(3) = le_img(row,col);
        pixel_value(4) = g_img(row,col);
        pixel_value(5) = fe_img(row,col);
        pixel_value(6) = b_img(row,col);
        % Find all the probabilities of the pixel
        for classIndex = 1:4
            % We need to reshape the mean list as it gives a 6x1 list, not
            % a 1x6 one, like we want.
            prob_vals(classIndex) = PDF(cov_list(:,:,classIndex),reshape(mean_list(:,classIndex),[1,6]),pixel_value);
        end
        maxValue = max(prob_vals);
        new_image(row,col) = find(prob_vals==maxValue);
    end
end

subplot(2,2,1) , imagesc(new_image), title('Predicted Results')
subplot(2,2,2) , imagesc(labelled_ground_truth), title('Ground_Truth')
subplot(2,2,3) , confusionchart(confusionmat(reshape(labelled_ground_truth,1,[]),reshape(new_image,1,[]))), title('Confusion Matrix')

function val = PDF(cov,mean,pixel_value)
    term_1 = 1/((2*pi).^6/2 * sqrt(det(cov)));
    term_2 = exp(-0.5*(pixel_value - mean)*inv(cov)*transpose(pixel_value - mean));
    val = term_1 * term_2;
end
