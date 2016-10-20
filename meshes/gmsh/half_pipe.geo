r1 = 0.5;
r2 = 0.7;
t =  0.08;
h = 6.0;
f = 0.01;
no_of_points_quarter = 15;
no_of_points_thickness = 2;
no_of_points_height = 60;

Point(1) = {0, 0, 0, f};
Point(2) = {r1, 0, 0, f};
Point(3) = {0, r2, 0, f};
Point(4) = {-r1, 0, 0, f};
Point(5) = {0, -r2, 0, f};
Point(6) = {r1+t, 0, 0, f};
Point(7) = {0, r2+t, 0, f};
Point(8) = {-r1+t, 0, 0, f};
Point(9) = {-0, -r2+t, 0, f};
Ellipse(1) = {2, 1, 2, 9};
Ellipse(2) = {3, 1, 3, 2};
Ellipse(3) = {7, 1, 1, 6};
Ellipse(4) = {6, 1, 1, 5};
Delete {
  Point{8, 4};
}
Line(5) = {7, 3};
Line(6) = {9, 5};
Line(7) = {6, 2};
Line Loop(8) = {3, 7, -2, -5};
Plane Surface(9) = {8};
Line Loop(10) = {4, -6, -1, -7};
Plane Surface(11) = {10};
Extrude {0, 0, 1} {
  Surface{9, 11};
}
Extrude {0, 0, 0.1} {
  Surface{33, 55};
}
Extrude {0, 0, 2} {
  Surface{77, 99};
}

Transfinite Line {3, 2, 1, 4, 37, 81, 35, 79, 57, 13, 59, 15, 103, 101, 123, 125} = no_of_points_quarter Using Progression 1;
Transfinite Line {6, 36, 80, 124, 104, 60, 16, 5, 7, 14, 58, 102} = no_of_points_thickness Using Progression 1;
Transfinite Line {27, 18, 19, 23, 41, 45} = 14 Using Progression 1;
Transfinite Line {85, 89, 71, 62, 63, 67} = 2 Using Progression 1;
Transfinite Line {129, 133, 107, 111, 106, 115} = 25 Using Progression 1;

Transfinite Surface "*";
Recombine Surface "*";
Transfinite Volume "*";
Physical Surface(144) = {11, 9}; //side left
Physical Surface(145) = {121, 143};n //side right
Physical Surface(146) = {120, 76, 32}; // bottom front
Physical Surface(147) = {134, 90, 46}; // bottom back
Physical Surface(148) = {64}; // rip front
Physical Surface(149) = {86}; // rip back
