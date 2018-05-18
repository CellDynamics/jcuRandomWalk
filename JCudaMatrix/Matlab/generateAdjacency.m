function [A, W] = generateAdjacency(stack, V, E)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

AI = zeros(E, 1);
AJ = zeros(E, 1);
AV = zeros(E, 1);
W = zeros(E, 1);
% assume linear indexing, row order, slice by slice
% i is linear index of each pixel in array [nrows ncols nz] counted from 0
% linear indexing (one loop) for better parallelism
vi = 1; % standard index for arrays where sparse coords are kept
in = 0; % counter for aggregating opposite pairs of neighbouring pixel in one row of incidence matrix. It is incidence matrix row index
BC = false; % false = do not use BC
siz = size(stack);
nrows = siz(1);
ncols = siz(2);
if length(siz)>2
    nz = siz(3);
else
    nz = 1;
end
for i = 0:V-1
    [row, col, z] = lin20ind(i, siz);
    % find 3 edges from current node (assume traveling right|lower|bottom)
%     left = col - 1;
    right = col + 1;
%     upper = row - 1;
    lower = row + 1;
%     top = z - 1;
    bottom = z + 1;
    % apply periodic BC
    if BC==true
%         if left < 0
%             left = ncols - 1;
%         end
        if right >= ncols
            right = 0;
        end
%         if upper < 0
%             upper = nrows - 1;
%         end
        if lower >= nrows
            lower = 0;
        end
%         if top < 0
%             top = nz - 1;
%         end
        if bottom >= nz
            bottom = 0;
        end
    end
    % revert to linear indexing
%     left_lin = ind20lin(row, left, z, siz);
    right_lin = ind20lin(row, right, z, siz);
%     upper_lin = ind20lin(upper, col, z, siz);L(:,seeds)=[];
    lower_lin = ind20lin(lower, col, z, siz);
%     top_lin = ind20lin(row, col, top, siz);
    bottom_lin = ind20lin(row, col, bottom, siz);
    
    % 1). from each current pixel travel only to right|lower|bottom
    % directions to avoid duplicating edges
    % 2). current pixel is positive (1),  right|lower|bottom negative (-1)
    % 3). Order of edges in incidence matrix is vertex realted:
    %   E1 - edge on right of 1st vertex
    %   E2 - edge to bottom from 1st vertexL(:,seeds)=[];
    %   E3 - edge to bottom layer
    %   E4 - edge to right of 2nd vertex
    %   ....

    % edge from current pixel to right    
    if right < ncols % if no BC this can be larger or equal than ncols and then it is skipped
        AI(vi) = in; % row in incidence matrix stored as sparse coordinate
        AJ(vi) = right_lin; % column in incidence matrix stored as sparse coordinate
        AV(vi) = -1; % value in incidence matrix for [AI(vi), AJ(vi)] cell
        vi = vi+1;
    
        AI(vi) = in; % the same row in incidence as it is one edge
        AJ(vi) = i; % current pixel
        AV(vi) = 1;
        vi = vi+1;
        W(in+1) = computeWeight(stack, row, right, z, row, col, z); % +1 only for Matlab
        in = in + 1; % go to next edge (next row in incidence)
    end      
    
    % edge from current pixel to lower
    if lower < nrows % if no BC this can be larger or equal than ncols and then it is skipped
        AI(vi) = in;
        AJ(vi) = lower_lin;
        AV(vi) = -1;
        vi = vi+1;
    
        AI(vi) = in; % the same row in incidence as it is one edge
        AJ(vi) = i; % current pixel
        AV(vi) = 1;
        vi = vi+1;
        W(in+1) = computeWeight(stack, lower, col, z, row, col, z);
        in = in + 1; % go to next edge (next row in incidence)
    end
    
    % edge from current pixel to bottom
    if nz > 1 && bottom < nz % if no BC this can be larger or equal than ncols and then it is skipped
        AI(vi) = in;
        AJ(vi) = bottom_lin;
        AV(vi) = -1;
        vi = vi+1;
    
        AI(vi) = in; % the same row in incidence as it is one edge
        AJ(vi) = i; % current pixel
        AV(vi) = 1;
        vi = vi+1;
        W(in+1) = computeWeight(stack, row, col, bottom, row, col, z);
        in = in + 1; % go to next edge (next row in incidence)
    end    
end

% +1 only to deal with Matlab indexing
A = sparse(AI+1, AJ+1, AV, in, V);

end

