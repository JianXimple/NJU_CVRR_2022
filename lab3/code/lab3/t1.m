P1=[496,353.5];
P2=[953.361,375.855];
P22=WarpH(P1,H);
P11=WarpH(P2,inv(H));
t=WarpH([1,1],inv(H));
tt=WarpH([-300,300],H);
function P2 = WarpH(P1, H)
x = P1(:, 1);
y = P1(:, 2);
p1 = [x'; y'; ones(1, length(x))];
q1 = H*p1;
q1 = q1./[q1(3, :); q1(3,:); q1(3, :)];
 
P2 = q1';
end