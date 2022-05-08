
clear all 
close all


%load dataset

%This loads a 500 x 256 matrix called threes. 
% Each line of this matrix is a single 16 by 16 image of a handwritten 3 
% that has been expanded out into a 256 long vector

load ../files/threes.mat -ascii

image_no=1;
colormap('gray');
imagesc(reshape(threes(image_no,:),16,16),[0,1]);
title("first image of the dataset")


%mean(X,VECDIM) operates on the dimensions specified in the vector 
%VECDIM. For example, mean(X,[1 2]) operates on the elements contained
%in the first and second dimensions of X.


%Display the average "3"
figure
average_three = mean(threes,1);
colormap('gray');
imagesc(reshape(average_three,16,16),[0,1]);
title('average three over all image of the dataset')




%Compute the covariance matrix of the whole dataset of 3s . 
% Compute the eigenvalues and eigenvectors of this covariance
% matrix. Plot the eigenvalues (plot(diag(D) where D is the 
% diagonal matrix of eigenvalues).

threes_normalised = threes - mean(threes,2);    

%compute covariance matrix
cov_matrix = cov(threes_normalised'); % input : N data point of dimension "p". 
                          % output : pxp covariance matrix


% Compress the dataset by projecting it onto one, two, three, 
% and four principal components. Now reconstruct the images
% from these compressions and plot some pictures of the four 
% reconstructions.
figure
for q=1:7
    %returns diagonal matrix D containing the eigenvalues on the main diagonal, 
    % and matrix V whose columns are the corresponding eigenvectors. 
    %Specify a second input "q" to compute a specific number of the largest 
    % eigenvalues.
    [V,D]=eigs(cov_matrix,q); %D gives eigen values, V gives corresponding eigenvectors 
    eigen_Values = diag(D); 
    
    %compute reduced set 
    reduced_dataset =  V' * threes_normalised;

    %reconstruct original dataset from reduced one
    reconstructed_dataset = V * reduced_dataset;
    

    image_no = 1;

    subplot(2,4,1)
    colormap('gray');
    imagesc(reshape(threes(image_no,:),16,16),[0,1]);
    title("original image of the dataset")


    subplot(2,4,q+1)
    colormap('gray');
    imagesc(reshape(reconstructed_dataset(image_no,:),16,16),[0,1]);
    title("reconstruction using " +q+ ' principal components')

    %reconstruction error, i.e. RMSE between original datased and
    %reconstructed dataset
    reconstruction_error(q) = sqrt(mean(mean((threes_normalised-reconstructed_dataset).^2)));
end

%plot eigen values
figure
bar(eigen_Values)
title("Eigen values")
xlabel("Indice of the eigen value, starting from the principal component")
ylabel("Value")
alpha(.1)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%compress and compute reconstruction error %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for q=1:50
    %returns diagonal matrix D containing the eigenvalues on the main diagonal, 
    % and matrix V whose columns are the corresponding eigenvectors. 
    %Specify a second input "q" to compute a specific number of the largest 
    % eigenvalues.
    [V,D]=eigs(cov_matrix,q); %D gives eigen values, V gives corresponding eigenvectors 
    eigen_Values = diag(D); 
    
    %compute reduced set 
    reduced_dataset =  V' * threes_normalised;

    %reconstruct original dataset from reduced one
    reconstructed_dataset = V * reduced_dataset;

    %reconstruction error, i.e. RMSE between original datased and
    %reconstructed dataset
    reconstruction_error(q) = sqrt(mean(mean((threes_normalised-reconstructed_dataset).^2)));
end


%plot reconstruction error agains number of eigen values used 
%with also eigenvalues with barplot
plot(reconstruction_error,'LineWidth',2)
title("Reconstruction error as a function of number of eigenvalues used")
xlabel("Number of eigen values used")
ylabel("RMSE")


%reconstruction error for q=256

q = 256
[V,D]=eigs(cov_matrix,q); %D gives eigen values, V gives corresponding eigenvectors 
eigen_Values = diag(D); 

%compute reduced set 
reduced_dataset =  V' * threes_normalised;

%reconstruct original dataset from reduced one
reconstructed_dataset = V * reduced_dataset;

%reconstruction error, i.e. RMSE between original datased and
%reconstructed dataset
reconstruction_error_for_q_256 = sqrt(mean(mean((threes_normalised-reconstructed_dataset).^2)))

