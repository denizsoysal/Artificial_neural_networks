clear all
close all 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%reduce dimensions with PCA on random number %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%so the goal is to go from a dimension of 50 to a lower one 
%goal here is to Examine different reduced datasets for different dimensions. 
% Try to reconstruct the original matrix. Estimate the error with RMSE e.g.
% sqrt(mean(mean((X-Xhat).^2)))

%Generate a 50x500 matrix of Gaussian random numbers 
numbers = randn(50,500); %to consider as 500 datapoints of dimension 50
%zero-mean the data
number_standardised = numbers - mean(numbers,2);

%compute covariance matrix
cov_matrix = cov(number_standardised'); % input : N data point of dimension "p". 
                          % output : pxp covariance matrix

%Create the q x p projection matrix E_transposed from the eigenvectors 
% corresponding to the q largest eigenvalues, and reduce the
%dataset by multiplying it with this matrix.
%we will test for "q" eigenvalues used, with "q" going from 1 to the
%dimensonality of the dataset

for q=1:size(number_standardised,1)
    %returns diagonal matrix D containing the eigenvalues on the main diagonal, 
    % and matrix V whose columns are the corresponding eigenvectors. 
    %Specify a second input "q" to compute a specific number of the largest 
    % eigenvalues.
    [V,D]=eigs(cov_matrix,q); %D gives eigen values, V gives corresponding eigenvectors 
    eigen_Values = diag(D); 
    
    %compute reduced set 
    reduced_dataset =  V' * number_standardised;

    %reconstruct original dataset from reduced one
    reconstructed_dataset = V * reduced_dataset;

    %reconstruction error, i.e. RMSE between original datased and
    %reconstructed dataset
    reconstruction_error(q) = sqrt(mean(mean((number_standardised-reconstructed_dataset).^2)));
end

% %plot reconstruction error agains number of eigen values used 
% plot(reconstruction_error,'LineWidth',2)
% title("Reconstruction error as a function of number of eigenvalues used")
% xlabel("Number of eigen values used")
% ylabel("RMSE")
% 
%plot reconstruction error agains number of eigen values used 
%with also eigenvalues with barplot
hold on
plot(reconstruction_error,'LineWidth',2)
bar(eigen_Values)
hold off
title("Reconstruction error as a function of number of eigenvalues used")
xlabel("Number of eigen values used")
ylabel("RMSE")
alpha(.1)



