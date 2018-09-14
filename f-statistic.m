H = 1;
M = 2;
L = 3;
IV = [9;7;6.5;8;7.5;7;9.5;8;6.5;7.5;8;6;7;6.5;7.5;8;6;6;6.5;6.5];
DV = [H;H;H;H;H;H;H;H;H;M;M;M;M;M;M;L;L;L;L;L];

errormsg1 = 'IV and DV must have the same size';
if numel(IV) ~= numel(DV)
    error(errormsg1)
end

errormsg2 = 'DV has not enough values';
if numel(DV) <= 1
    error(errormsg2)
end

DV2 = accumarray(DV,1);
%DV2 = [9;6;5];

% A: all the 'H' elements of IV
A = IV(1 : DV2(1));
meanA = mean(A);

% B: all the 'M' elements of IV
B = IV(DV2(1) : DV2(1)+DV2(2));
meanB = mean(B);

% C: all the 'L' elements of IV
C = IV(DV2(1)+DV2(2) : DV2(1)+DV2(2)+DV2(3));
meanC = mean(C);

grandMean = (DV2(1) * meanA + DV2(2) * meanB + DV2(3) * meanC) / (DV2(1) + DV2(2) + DV2(3));

SSB = ((DV2(1)*(meanA - grandMean).^2) + (DV2(2)*(meanB - grandMean).^2) + (DV2(3)*(meanC - grandMean).^2)) / (numel(DV2)-1);
