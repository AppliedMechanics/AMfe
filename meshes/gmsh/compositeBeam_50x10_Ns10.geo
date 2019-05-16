// Gmsh localized nonlinearities example structure

lc=0.7;
H=10;
L=50;
l_force=5;

nf=3;
nm=2;
dh_f=2*lc;
dh_m=(H-nf*dh_f)/nm;

Point(1) = {0, 0, 0, lc};
Point(2) = {0, dh_f, 0, lc};
Point(3) = {0, dh_f+dh_m, 0, lc};
Point(4) = {0, 2*dh_f+dh_m, 0, lc};

Point(5) = {0, 2*dh_f+2*dh_m, 0, lc};
Point(6) = {0, H, 0, lc};

Point(7) = {L, 0, 0, lc};
Point(8) = {L, dh_f, 0, lc};
Point(9) = {L, dh_f+dh_m, 0, lc};
Point(10) = {L, 2*dh_f+dh_m, 0, lc};

Point(11) = {L, 2*dh_f+2*dh_m, 0, lc};
Point(12) = {L, H, 0, lc};

Point(13) = {L-l_force, H, 0, lc};


Line(1) = {1, 2};
Line(2) = {2, 8};
Line(3) = {8, 7};
Line(4) = {7, 1};

Line(5) = {2, 8};
Line(6) = {8, 9};
Line(7) = {9, 3};
Line(8) = {3, 2};

Line(9) = {9, 3};
Line(10) = {3, 4};
Line(11) = {4, 10};
Line(12) = {10, 9};

Line(13) = {4, 10};
Line(14) = {10, 11};
Line(15) = {11, 5};
Line(16) = {5, 4};

Line(17) = {11, 5};
Line(18) = {5, 6};
Line(19) = {6, 13};
Line(20) = {13, 12};
Line(21) = {12, 11};


Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {5, 6, 7, 8};
Line Loop(3) = {9, 10, 11, 12};
Line Loop(4) = {13, 14, 15, 16};
Line Loop(5) = {17, 18, 19, 20, 21};


Plane Surface(1) = {-1};
Plane Surface(2) = {2};
Plane Surface(3) = {-3};
Plane Surface(4) = {4};
Plane Surface(5) = {-5};

Physical Point("xy dirichlet point",1) = {1};
Physical Line("x dirichlet line",2) = {1, 8, 10, 16, 18};
Physical Line("x neumann",3) = {3, 6, 12, 14, 21};
Physical Line("z neumann",4) = {20};

Physical Surface("surface material steel",5) = {1, 3, 5};
Physical Surface("surface material elastomer",6) = {2, 4};

Coherence;