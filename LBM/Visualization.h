#ifndef Visualization_h
#define Visualization_h

#include<Gl/glut.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include <iostream>
#include <fstream>
#include "LBM0.h"

void Initialize();
void DrawLBM1();
void DrawLBM2();
void DrawAtmosHor();
void DrawAtmosVer();
void Timer(int value);

#endif // !Visualization_h
