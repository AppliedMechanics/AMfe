// Gmsh project created on Wed Sep 21 15:00:42 2016
cl = 0.5;
l1 = 2;
l2 = 1;
height = 0.4;

x_dist = 1.08;
y_dist = 0.28;
l3 = 0.5;
l4 = 0.5;


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

Point(5) = {x_dist, y_dist, 0, cl};
Point(6) = {x_dist + l3, y_dist, 0, cl};
Point(7) = {x_dist + l3, y_dist + l4, 0, cl};
Point(8) = {x_dist, y_dist + l4, 0, cl};

Line(8) = {8, 7};
Line(9) = {7, 6};
Line(10) = {6, 5};
Line(11) = {5, 8};
Line Loop(12) = {8, 9, 10, 11};
Plane Surface(13) = {12};

Extrude {0, 0, -height} {
  Surface{7};
}

Extrude {0, 0, height} {
  Surface{13};
}


// Transfinite Line "*";
Transfinite Surface "*";
Recombine Surface "*";
Transfinite Volume "*";

Physical Volume('box') = {2};
Physical Volume('platform') = {1};
Physical Surface('fixature') = {26};
Physical Surface('box_side') = {56};
Physical Surface('master_area') = {7};
Physical Surface('slave_area') = {13};
