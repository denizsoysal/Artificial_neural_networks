clear all
close all 



%PCA reduction of the "choles_all" dataset

%load dataset
%choles_add dataset has several components, but we will just use the "p"
%matrix which is a 21x264 matrix (dimension = 21)
load choles_all ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%reduce dimensions with PCA %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


number_standardised = p - mean(p,2);

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


%plot reconstruction error agains number of eigen values used 
%with also eigenvalues with barplot
subplot(1,2,1)
plot(reconstruction_error,'LineWidth',2)
title("Reconstruction error as a function of number of eigenvalues used")
xlabel("Number of eigen values used")
ylabel("RMSE")
subplot(1,2,2)
bar(eigen_Values)
title("Eigen values")
xlabel("Indice of the eigen value, starting from the principal component")
ylabel("Value")
alpha(.1)
