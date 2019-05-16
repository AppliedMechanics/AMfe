//
length = 0.500;
height = 0.500;
thickness = 0.002;
numell = 20+1;
numelh = 20+1;
numelt = 3+1;
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
Extrude {0,0,thickness} {
  Surface{1};
  Layers{numelt-1};
  Recombine;
}
//
Physical Volume(1) = {1};
Physical Surface(2) = {26};
//
Physical Point(3) = {1};
Physical Point(4) = {2};
Physical Point(5) = {3};
Physical Point(6) = {4};

