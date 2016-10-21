r1 = 0.5;
r2 = 0.7;
t =  0.08;
h = 6.0;
f = 0.08;
no_of_points_quarter = 12;
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

Ellipse(1) = {9, 1, 9, 2};
Ellipse(2) = {2, 1, 2, 3};
Ellipse(3) = {3, 1, 3, 8};
Ellipse(4) = {8, 1, 8, 9};
Ellipse(5) = {5, 1, 5, 6};
Ellipse(6) = {6, 1, 6, 7};
Ellipse(7) = {7, 1, 7, 4};
Ellipse(8) = {4, 1, 4, 5};

Line(9) = {7, 3};
Line(10) = {6, 2};
Line(11) = {9, 5};
Line(12) = {8, 4};

Line Loop(13) = {7, -12, -3, -9};
Plane Surface(14) = {13};
Line Loop(15) = {12, 8, -11, -4};
Plane Surface(16) = {15};
Line Loop(17) = {11, 5, 10, -1};
Plane Surface(18) = {17};
Line Loop(19) = {10, 2, -9, -6};
Plane Surface(20) = {19};

Extrude {0, 0, 1} {
  Surface{14, 16, 18, 20};
}
Extrude {0, 0, 0.1} {
  Surface{42, 108, 86, 64};
}
Extrude {0, 0, 2} {
  Surface{152, 130, 196, 174};
}

Transfinite Line {137, 138, 160, 159, 120, 116, 124, 115} = 3;
//Transfinite Line {76, 80, 54, 58, 36, 27, 32, 28} = 15;
Transfinite Line {11, 12, 10, 9, 46, 154, 23, 111, 25, 113, 68, 132, 244, 221, 200, 198} = 2;
Transfinite Line {5, 1, 8, 4, 3, 7, 2, 6, 45, 47, 177, 179, 67, 155, 69, 157, 135, 133, 91, 89, 24, 112, 22, 110, 243, 245, 265, 267, 199, 201, 222, 220} = 12;

Transfinite Surface "*";
Recombine Surface "*";
Transfinite Volume "*";

Physical Volume(285) = {4, 1, 5, 10, 9, 6, 12, 7, 3, 2, 8, 11};
Physical Surface(286) = {14, 20, 18, 16}; // left side
Physical Surface(287) = {262, 240, 218, 284}; // right side
Physical Surface(288) = {151}; // top rip
