// Gmsh localized nonlinearities example structure

lc=2;
H=10;
L=50;
lF=5;

dh=0.1;
ls=L/5;
hs=H/2;

Point(1) = {0, 0, 0, lc};
Point(2) = {0, hs, 0, lc};
Point(3) = {ls, hs, 0, lc};
Point(4) = {ls, 0, 0, lc};

Point(5) = {ls*2, 0, 0, lc};
Point(6) = {ls*2, hs, 0, lc};

Point(7) = {ls*3, 0, 0, lc};
Point(8) = {ls*3, hs, 0, lc};

Point(9) = {ls*4, 0, 0, lc};
Point(10) = {ls*4, hs, 0, lc};

Point(11) = {L, 0, 0, lc};
Point(12) = {L, hs, 0, lc};

Point(13) = {0, H, 0, lc};
Point(14) = {ls, H, 0, lc};

Point(15) = {ls*2, H, 0, lc};

Point(16) = {ls*3, H, 0, lc};

Point(17) = {ls*4, H, 0, lc};

Point(18) = {L-lF, H, 0, lc};
Point(19) = {L, H, 0, lc};

Line(1) = {1, 4};
Line(2) = {4, 3};
Line(3) = {3, 2};
Line(4) = {2, 1};

Line(5) = {4, 5};
Line(6) = {5, 6};
Line(7) = {6, 3};
Line(8) = {3, 4};

Line(9) = {5, 7};
Line(10) = {7, 8};
Line(11) = {8, 6};
Line(12) = {6, 5};

Line(13) = {7, 9};
Line(14) = {9, 10};
Line(15) = {10, 8};
Line(16) = {8, 7};

Line(17) = {9, 11};
Line(18) = {11, 12};
Line(19) = {12, 10};
Line(20) = {10, 9};

Line(21) = {2, 3};
Line(22) = {3, 14};
Line(23) = {14, 13};
Line(24) = {13, 2};

Line(25) = {3, 6};
Line(26) = {6, 15};
Line(27) = {15, 14};
Line(28) = {14, 3};

Line(29) = {6, 8};
Line(30) = {8, 16};
Line(31) = {16, 15};
Line(32) = {15, 6};

Line(33) = {8, 10};
Line(34) = {10, 17};
Line(35) = {17, 16};
Line(36) = {16, 8};

Line(37) = {10, 12};
Line(38) = {12, 19};
Line(39) = {19, 18};
Line(40) = {18, 17};
Line(41) = {17, 10};


Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {5, 6, 7, 8};
Line Loop(3) = {9, 10, 11, 12};
Line Loop(4) = {13, 14, 15, 16};
Line Loop(5) = {17, 18, 19, 20};
Line Loop(6) = {21, 22, 23, 24};
Line Loop(7) = {25, 26, 27, 28};
Line Loop(8) = {29, 30, 31, 32};
Line Loop(9) = {33, 34, 35, 36};
Line Loop(10) = {37, 38, 39, 40, 41};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};
Plane Surface(6) = {6};
Plane Surface(7) = {7};
Plane Surface(8) = {8};
Plane Surface(9) = {9};
Plane Surface(10) = {10};

Physical Surface(101) = {1};
Physical Surface(102) = {2};
Physical Surface(103) = {3};
Physical Surface(104) = {4};
Physical Surface(105) = {5};
Physical Surface(106) = {6};
Physical Surface(107) = {7};
Physical Surface(108) = {8};
Physical Surface(109) = {9};
Physical Surface(110) = {10};

Physical Point("xy dirichlet point",1) = {1};
Physical Line("x dirichlet line",2) = {4, 24};
Physical Line("x neumann",3) = {18, 38};
Physical Line("z neumann",4) = {39};

Physical Surface("surface material steel",5) = {1, 3, 4, 5, 6, 7, 8, 10};
Physical Surface("surface material elastomer",6) = {2, 9};

Coherence;