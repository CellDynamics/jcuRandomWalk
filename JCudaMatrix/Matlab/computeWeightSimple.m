function w = computeWeightSimple(stack, row, col, z, row1, col1, z1, sigmaGrad)
% row,col,z are 0 based

% Matlab related only
row = row + 1;
col = col + 1;
z = z + 1;
row1 = row1 + 1;
col1 = col1 + 1;
z1 = z1 + 1;

w = exp(-0.5*(stack(row,col,z) - stack(row1,col1,z1)).^2 / sigmaGrad^2);
end