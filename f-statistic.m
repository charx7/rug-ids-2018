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
