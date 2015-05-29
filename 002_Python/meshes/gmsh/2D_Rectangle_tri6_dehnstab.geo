// Gmsh project created on Wed May 27 15:23:03 2015
Point(1) = {0, 00, 0, 1.0};
Point(2) = {1, 00, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};

Point(5) = {0.5, 0.5, 0, 1.0};
Point(6) = {0.6, 0.5, 0, 1.0};
Point(7) = {0.5, 0.6, 0, 1.0};
Point(8) = {0.4, 0.5, 0, 1.0};
Point(9) = {0.5, 0.4, 0, 1.0};
Circle(1) = {6, 5, 7};
Circle(2) = {7, 5, 8};
Circle(3) = {8, 5, 9};
Circle(4) = {9, 5, 6};
Line(5) = {3, 4};
Line(6) = {4, 1};
Line(7) = {1, 2};
Line(8) = {3, 2};
Line Loop(9) = {5, 6, 7, -8};
Line Loop(10) = {2, 3, 4, 1};
Plane Surface(11) = {9, 10};
