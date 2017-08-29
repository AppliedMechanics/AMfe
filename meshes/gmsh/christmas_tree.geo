// Gmsh project created on Wed Nov  2 08:39:44 2016
p = 0.05;
Point(1) = {0, 0, 0, p};
Point(2) = {0, 2, 0, p};
Point(3) = {0.35, 1.61, 0, p};
Point(4) = {0.19, 1.62, 0, p};
Point(5) = {0.47, 1.25, 0, p};
Point(6) = {0.28, 1.26, 0, p};
Point(7) = {0.62, 0.82, -0, p};
Point(8) = {0.41, 0.83, -0, p};
Point(9) = {0.81, 0.3, -0, p};
Point(10) = {0.13, 0.33, -0, p};
Point(11) = {0.16, 0, -0, p};
Point(12) = {0.14, 1.81, 0, p};
Point(13) = {0.26, 1.6, 0, p};
Point(14) = {0.31, 1.4, 0, p};
Point(15) = {0.36, 1.24, 0, p};
Point(16) = {0.4, 1, 0, p};
Point(17) = {0.49, 0.82, -0, p};
Point(18) = {0.52, 0.55, -0, p};
Point(19) = {0.41, 0.29, -0, p};
Point(20) = {0.12, 0.15, -0, p};
BSpline(1) = {2, 12, 3};
BSpline(2) = {3, 13, 4};
BSpline(3) = {4, 14, 5};
BSpline(4) = {5, 15, 6};
BSpline(5) = {6, 16, 7};
BSpline(6) = {7, 17, 8};
BSpline(7) = {8, 18, 9};
BSpline(8) = {9, 19, 10};
BSpline(9) = {10, 20, 11};
Line(10) = {11, 1};

Rotate {{0, 1, 0}, {0, 0, 0}, Pi} {
  Duplicata { Line{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}; }
}

// Line Loop(21) = {8, 9, 10, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, 1, 2, 3, 4, 5, 6, 7};
Line Loop(21) = {-7, -6, -5, -4, -3, -2, -1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, -10, -9, -8};
Plane Surface(22) = {21};

Physical Surface(23) = {22}; // main mesh
Physical Line(24) = {10, 20}; // bottom root
Physical Line(25) = {1}; // top branch
