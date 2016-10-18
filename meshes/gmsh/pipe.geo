r1 = 0.5;
r2 = 0.7;
t =  0.08;
h = 6.0;
f = 0.01;
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

Point(11) = {0, 0, h, f};
Point(12) = {r1, 0, h, f};
Point(13) = {0, r2, h, f};
Point(14) = {-r1, 0, h, f};
Point(15) = {0, -r2, h, f};
Point(16) = {r1+t, 0, h, f};
Point(17) = {0, r2+t, h, f};
Point(18) = {-r1+t, 0, h, f};
Point(19) = {-0, -r2+t, h, f};

Ellipse(11) = {19, 11, 19, 12};
Ellipse(12) = {12, 11, 12, 13};
Ellipse(13) = {13, 11, 13, 18};
Ellipse(14) = {18, 11, 18, 19};
Ellipse(15) = {15, 11, 15, 16};
Ellipse(16) = {16, 11, 16, 17};
Ellipse(17) = {17, 11, 17, 14};
Ellipse(18) = {14, 11, 14, 15};

Line(19) = {14, 18};
Line(20) = {15, 19};
Line(21) = {16, 12};
Line(22) = {17, 13};
Line(23) = {7, 3};
Line(24) = {4, 8};
Line(25) = {5, 9};
Line(26) = {6, 2};

Line(27) = {4, 14};
Line(28) = {8, 18};
Line(29) = {2, 12};
Line(30) = {6, 16};
Line(31) = {9, 19};
Line(32) = {5, 15};
Line(33) = {3, 13};
Line(34) = {7, 17};

Transfinite Line {12, 16, 13, 17, 18, 14, 11, 15} = no_of_points_quarter Using Progression 1;
Transfinite Line {4, 8, 1, 5, 6, 2, 3, 7} = no_of_points_quarter Using Progression 1;
Transfinite Line {22, 21, 20, 19, 23, 26, 25, 24} = no_of_points_thickness Using Progression 1;
Transfinite Line {34, 33, 29, 30, 31, 32, 27, 28} = no_of_points_height Using Progression 1;

Line Loop(35) = {23, 3, -24, -7};
Plane Surface(36) = {35};
Line Loop(37) = {24, 4, -25, -8};
Plane Surface(38) = {37};
Line Loop(39) = {25, 1, -26, -5};
Plane Surface(40) = {39};
Line Loop(41) = {26, 2, -23, -6};
Plane Surface(42) = {41};
Line Loop(43) = {19, 14, -20, -18};
Plane Surface(44) = {43};
Line Loop(45) = {20, 11, -21, -15};
Plane Surface(46) = {45};
Line Loop(47) = {21, 12, -22, -16};
Plane Surface(48) = {47};
Line Loop(49) = {22, 13, -19, -17};
Plane Surface(50) = {49};
Line Loop(51) = {24, 28, -19, -27};
Plane Surface(52) = {51};
Line Loop(53) = {25, 31, -20, -32};
Plane Surface(54) = {53};
Line Loop(55) = {21, -29, -26, 30};
Plane Surface(56) = {55};
Line Loop(57) = {23, 33, -22, -34};
Plane Surface(58) = {57};

Line Loop(59) = {16, -34, -6, 30};
Ruled Surface(60) = {59};
Line Loop(61) = {2, 33, -12, -29};
Ruled Surface(62) = {61};
Line Loop(63) = {1, 29, -11, -31};
Ruled Surface(64) = {63};
Line Loop(65) = {5, 30, -15, -32};
Ruled Surface(66) = {65};
Line Loop(67) = {4, 31, -14, -28};
Ruled Surface(68) = {67};
Line Loop(69) = {8, 32, -18, -27};
Ruled Surface(70) = {69};
Line Loop(71) = {3, 28, -13, -33};
Ruled Surface(72) = {71};
Line Loop(73) = {7, 27, -17, -34};
Ruled Surface(74) = {73};

Surface Loop(75) = {60, 48, 62, 42, 58, 56};
Volume(76) = {75};
Surface Loop(77) = {66, 40, 64, 46, 56, 54};
Volume(78) = {77};
Surface Loop(79) = {70, 38, 68, 44, 54, 52};
Volume(80) = {79};
Surface Loop(81) = {74, 36, 72, 50, 52, 58};
Volume(82) = {81};

Transfinite Surface "*";
Recombine Surface "*";
Transfinite Volume "*";

Physical Surface(83) = {42, 40, 38, 36};
Physical Volume(84) = {82, 76, 78, 80};
Physical Surface(85) = {50};
