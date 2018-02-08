//
height = 10;
diameter = 1;
thickness = 0.01;
//
Point(1) = {0,diameter/2,0,999};
Point(2) = {-diameter/2,0,0,999};
Point(3) = {0,-diameter/2,0,999};
Point(4) = {diameter/2,0,0,999};
Point(5) = {0,diameter/2-thickness,0,999};
Point(6) = {-diameter/2+thickness,0,0,999};
Point(7) = {0,-diameter/2+thickness,0,999};
Point(8) = {diameter/2-thickness,0,0,999};
Point(99) = {0,0,0,999};
//
Circle(1) = {1,99,2};
Circle(2) = {2,99,3};
Circle(3) = {3,99,4};
Circle(4) = {4,99,1};
Circle(5) = {5,99,8};
Circle(6) = {8,99,7};
Circle(7) = {7,99,6};
Circle(8) = {6,99,5};
//
Line Loop(1) = {1,2,3,4,5,6,7,8};
//
Plane Surface(1) = 1;
//
Transfinite Line{1,2,3,4,5,6,7,8} = 3;
Transfinite Surface{1};
Recombine Surface{1};
//
Extrude {0,0,height} {
  Surface{1};
  Layers{20};
  Recombine;
}
//
Physical Surface(2) = {1};
Physical Surface(3) = {21};
Physical Surface(4) = {25};
Physical Surface(5) = {29};
Physical Surface(6) = {33};
//
Physical Volume(1) = {1};
