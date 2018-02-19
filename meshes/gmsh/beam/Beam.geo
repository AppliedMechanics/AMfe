//
length = 10;
height = 1;
numell = 100+1;
numelh = 10+1;
//
Point(1) = {0,0,0,999};
Point(2) = {length,0,0,999};
Point(3) = {length,height,0,999};
Point(4) = {0,height,0,999};
//
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
//
Line Loop(1) = {1,2,3,4};
//
Plane Surface(1) = 1;
//
Transfinite Line{1,3} = numell;
Transfinite Line{2,4} = numelh;
Transfinite Surface{1};
Recombine Surface{1};
//
Physical Line(2) = {1};
Physical Line(3) = {2};
Physical Line(4) = {3};
Physical Line(5) = {4};
//
Physical Surface(1) = {1};
