function [r,k,z] = lin20ind(ndx, siz)
%LIN20IND Change linear index to [row, column, z]
%   In contrary to Matlab version this method relies on 0-based indexes and is
%   row-ordered.
k2 = siz(1)*siz(2); % size of the slice
    
vi = mod(ndx, k2); % related to slice
vj = (ndx - vi)/k2;
ndx = vi;
    
% modified for one layer, 0-based and row ordered
vi = mod(ndx, siz(2));
r = double((ndx - vi)/siz(2)); % row
k = double(vi); % column
z = double(vj);
end