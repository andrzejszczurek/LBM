#ifndef Visualization_h
#define Visualization_h

#include<Gl/glut.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>
#include <iostream>
#include <fstream>

#include "LBM0.h"

void DrawAtmosHor(char rgba[Npic * Ny][Npic * Nx][3]);
void DrawAtmosVer(char rgba[Npic * Ny][Npic * Nx][3]);
void DrawTempG(char rgba[Ny][Nx][3]);

void BMPout(char[Npic * Ny][Npic * Nx][3], char* bmpfile);
void DrawLine(int x0, int y0, int x1, int y1, char rgbp[Npic * Ny][Npic * Nx][3]);


#endif
