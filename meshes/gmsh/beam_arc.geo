// Gmsh project created on Mon Mar 21 18:26:05 2016
h = 0.1;
R = 3.0;
w = 2.0;
h_m = 0.05;

Point(1) = {-w/2, 0, 0, h_m};
Point(2) = {-w/2, h, 0, h_m};
Point(3) = {w/2, h, 0, h_m};
Point(4) = {w/2, 0, 0, h_m};
Point(5) = {0, -R, 0, h_m};
Circle(1) = {2, 5, 3};
Circle(2) = {1, 5, 4};
Line(3) = {2, 1};
Line(4) = {3, 4};
Physical Line(5) = {3};
Physical Line(6) = {4};
Physical Line(7) = {1};
Line Loop(8) = {3, 2, -4, -1};
Plane Surface(9) = {8};
Physical Surface(10) = {9};
