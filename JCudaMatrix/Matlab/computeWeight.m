function w = computeWeight(stack, row, col, z, row1, col1, z1)
% row,col,z are 0 based

% Matlab related only
row = row + 1;
col = col + 1;
z = z + 1;
row1 = row1 + 1;
col1 = col1 + 1;
z1 = z1 + 1;

sigma_grad=0.1;%0.3; %standard deviation for the intensity gradient weight
sigma_mean=1e6 ; %standard deviation for the difference from mean weight
mean_source = 0.6;

 w = exp(-0.5*((stack(row,col,z)-stack(row1,col1,z1)).^2 / sigma_grad^2) - 0.5*((stack(row,col,z)-mean_source).^2 / sigma_mean^2));
%  w = abs(stack(row,col,z)-stack(row1,col1,z1));
end