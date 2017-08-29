// Gmsh project created on Wed Sep 21 15:00:42 2016
cl = 0.05;
l1 = 2;
l2 = 1;
height = 0.05;
Point(1) = {0, 0, 0, cl};
Point(2) = {l1, 0, 0, cl};
Point(3) = {l1, l2, 0, cl};
Point(4) = {0,  l2, 0, cl};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(6) = {3, 4, 1, 2};
Plane Surface(7) = {6};

Extrude {0, 0, height} {
  Surface{7};
}

Transfinite Line "*";
Transfinite Surface "*";
Recombine Surface "*";
Transfinite Volume "*";

Physical Volume(30) = {1};
Physical Surface(31) = {28};
Physical Surface(32) = {20};
