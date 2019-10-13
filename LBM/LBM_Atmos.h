#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Gl/glut.h>
#include "LBM0.h"
#include "Visualization.h"
#include <string.h>
#include <iostream>
#include <fstream>
#include "Obstacles.h"


void DrawAtmosHor(char rgba[Npic * Ny][Npic * Nx][3])
{
   GLfloat CellRed, CellGreen, CellBlue;
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   for (int i = 0; i < Nx; i++)
   {
      for (int j = 0; j < Ny; j++)
      {
         CellRed = CellBlue = CellGreen = 1.f;
         if (AtmosVx[i][j] < -0.0005) 
         { 
             CellRed = 1 + 20.f * AtmosVx[i][j];
             CellBlue = 1.f;
             CellGreen = CellRed; 
         }

         if (AtmosVx[i][j] > 0.0005) 
         { 
             CellBlue = 1 - 20.f * AtmosVx[i][j];
             CellRed = 1.f;
             CellGreen = CellBlue;
         }

         // kształtowanie zmiennych rgba - potrzebnych do zapisy wyniku jako bitmapa
         for (int jj = 0; jj < Npic; jj++)
         {
            for (int ii = 0; ii < Npic; ii++)
            {
               rgba[Npic * j + jj][Npic * i + ii][0] = 255 * CellBlue;
               rgba[Npic * j + jj][Npic * i + ii][1] = 255 * CellGreen;
               rgba[Npic * j + jj][Npic * i + ii][2] = 255 * CellRed;
            }
         }

         // rysowanie przegrody
         if (AtmosState[i][j] == 1)
             CellBlue = CellGreen = CellRed = .1;

         glColor3f(CellRed, CellGreen, CellBlue);
         glBegin(GL_QUADS);
         glVertex3f(i, j, 0.0);
         glVertex3f(i, j + 1, 0.0);
         glVertex3f(i + 1, j + 1, 0.0);
         glVertex3f(i + 1, j, 0.0);
         glEnd();
      }
   }

   // rysowanie lini toku
   glLineWidth(1.);
   glColor3f(0.0, 0.0, 0.0);
   for (int i = 10; i < Nx; i = i + 10) 
   {
      for (int j = 10; j < Ny; j = j + 10) 
      {
         glBegin(GL_LINES);
         glVertex3f(i, j, 0.);
         glVertex3f(i + 500 * AtmosVx[i][j], j + 500 * AtmosVy[i][j], 0);
         glEnd();
         DrawLine(Npic * (i + 0.5), Npic * (j + 0.5), Npic * (i + 0.5 + 500 * AtmosVx[i][j]), Npic * (j + 0.5 + 500 * AtmosVy[i][j]), rgba);
      }
   }
   glutSwapBuffers();
}


void DrawAtmosVer(char rgba[Npic * Ny][Npic * Nx][3])
{
   GLfloat  CellRed, CellGreen, CellBlue;
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   for (int i = 0; i < Nx; i++) 
   {
      for (int j = 0; j < Ny; j++) 
      {
         CellRed = CellBlue = CellGreen = 1.f;
         if (AtmosVy[i][j] < -0.0002)
         { 
             CellRed = 1 + 20.f * AtmosVy[i][j];
             CellBlue = 1.f;
             CellGreen = CellRed; 
         }
         if (AtmosVy[i][j] > 0.0002)
         {
             CellBlue = 1 - 20.f * AtmosVy[i][j];
             CellRed = 1.f;
             CellGreen = CellBlue;
         }

         // rysowanie przegrody
         if (AtmosState[i][j] == 1)
             CellBlue = CellGreen = CellRed = .1;

         // kształtowanie zmiennych rgba - potrzebnych do zapisy wyniku jako bitmapa
         for (int jj = 0; jj < Npic; jj++)
         {
             for (int ii = 0; ii < Npic; ii++)
             {
                 rgba[Npic * j + jj][Npic * i + ii][0] = 255 * CellBlue;
                 rgba[Npic * j + jj][Npic * i + ii][1] = 255 * CellGreen;
                 rgba[Npic * j + jj][Npic * i + ii][2] = 255 * CellRed;
             }
         }

         glColor3f(CellRed, CellGreen, CellBlue);
         glBegin(GL_QUADS);
         glVertex3f(i, j, 0.0);
         glVertex3f(i, j + 1, 0.0);
         glVertex3f(i + 1, j + 1, 0.0);
         glVertex3f(i + 1, j, 0.0);
         glEnd();
      }
   }
   glLineWidth(1.);
   glColor3f(0.0, 0.0, 0.0);
   for (int i = 10; i < Nx; i = i + 10) {
      for (int j = 10; j < Ny; j = j + 10) {
         glBegin(GL_LINES);
         glVertex3f(i, j, 0.);
         glVertex3f(i + 500 * AtmosVx[i][j], j + 500 * AtmosVy[i][j], 0);
         glEnd();
         DrawLine(Npic * (i + 0.5), Npic * (j + 0.5), Npic * (i + 0.5 + 500 * AtmosVx[i][j]), Npic * (j + 0.5 + 500 * AtmosVy[i][j]), rgba);
      }
   }
   glutSwapBuffers();
}

__global__ void InitialAtmos(bool NewSim)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Initial density and velocity, equilibrium and input distribution functions 

    if (NewSim)
    {
        if (x == 0 || x == Nx - 1 || y == 0 || y == Ny - 1) {
            AtmosVx[x][y] = (1.0f * y / Ny) * 0.05; // velocity from 0 to 0.05
        }
        else
        {
            AtmosVx[x][y] = 0.0f;
        }
        AtmosVy[x][y] = 0.0f;
        AtmosRho[x][y] = 1.0f;
    }

    float vx2 = AtmosVx[x][y] * AtmosVx[x][y];
    float vy2 = AtmosVy[x][y] * AtmosVy[x][y];
    float eu2 = 1.f - 1.5f * (vx2 + vy2);
    float Rho36 = AtmosRho[x][y] / 36.0f;
    Atmoseq[x][y].fC = 16.0f * Rho36 * eu2;
    Atmoseq[x][y].fN = 4.0f * Rho36 * (eu2 + 3.0f * AtmosVy[x][y] + 4.5f * vy2);
    Atmoseq[x][y].fE = 4.0f * Rho36 * (eu2 + 3.0f * AtmosVx[x][y] + 4.5f * vx2);
    Atmoseq[x][y].fS = 4.0f * Rho36 * (eu2 - 3.0f * AtmosVy[x][y] + 4.5f * vy2);
    Atmoseq[x][y].fW = 4.0f * Rho36 * (eu2 - 3.0f * AtmosVx[x][y] + 4.5f * vx2);
    Atmoseq[x][y].fNE = Rho36 * (eu2 + 3.0f * (AtmosVx[x][y] + AtmosVy[x][y]) + 4.5f * (AtmosVx[x][y] + AtmosVy[x][y]) * (AtmosVx[x][y] + AtmosVy[x][y]));
    Atmoseq[x][y].fSE = Rho36 * (eu2 + 3.0f * (AtmosVx[x][y] - AtmosVy[x][y]) + 4.5f * (AtmosVx[x][y] - AtmosVy[x][y]) * (AtmosVx[x][y] - AtmosVy[x][y]));
    Atmoseq[x][y].fSW = Rho36 * (eu2 + 3.0f * (-AtmosVx[x][y] - AtmosVy[x][y]) + 4.5f * (AtmosVx[x][y] + AtmosVy[x][y]) * (AtmosVx[x][y] + AtmosVy[x][y]));
    Atmoseq[x][y].fNW = Rho36 * (eu2 + 3.0f * (-AtmosVx[x][y] + AtmosVy[x][y]) + 4.5f * (-AtmosVx[x][y] + AtmosVy[x][y]) * (-AtmosVx[x][y] + AtmosVy[x][y]));

    Atmosin[x][y].fC = Atmoseq[x][y].fC;
    Atmosin[x][y].fE = Atmoseq[x][y].fE;  Atmosin[x][y].fN = Atmoseq[x][y].fN;
    Atmosin[x][y].fS = Atmoseq[x][y].fS;  Atmosin[x][y].fW = Atmoseq[x][y].fW;
    Atmosin[x][y].fNE = Atmoseq[x][y].fNE; Atmosin[x][y].fNW = Atmoseq[x][y].fNW;
    Atmosin[x][y].fSE = Atmoseq[x][y].fSE; Atmosin[x][y].fSW = Atmoseq[x][y].fSW;
}

__global__ void EquiRelaxAtmos()
{
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;
   // macroscopic density and velosity for the current cell
   AtmosRho[x][y] = Atmosin[x][y].fC + Atmosin[x][y].fE + Atmosin[x][y].fW + Atmosin[x][y].fS + Atmosin[x][y].fN + Atmosin[x][y].fNE + Atmosin[x][y].fNW + Atmosin[x][y].fSE + Atmosin[x][y].fSW;
   AtmosVx[x][y] = (Atmosin[x][y].fE + Atmosin[x][y].fSE + Atmosin[x][y].fNE - Atmosin[x][y].fW - Atmosin[x][y].fSW - Atmosin[x][y].fNW) / AtmosRho[x][y];
   AtmosVy[x][y] = (Atmosin[x][y].fN + Atmosin[x][y].fNW + Atmosin[x][y].fNE - Atmosin[x][y].fS - Atmosin[x][y].fSW - Atmosin[x][y].fSE) / AtmosRho[x][y];

   // equillibrum function
   float vx2 = AtmosVx[x][y] * AtmosVx[x][y];
   float vy2 = AtmosVy[x][y] * AtmosVy[x][y];
   float eu2 = 1.f - 1.5f * (vx2 + vy2);
   float Rho36 = AtmosRho[x][y] / 36.0f;
   Atmoseq[x][y].fC = 16.0f * Rho36 * eu2;
   Atmoseq[x][y].fN = 4.0f * Rho36 * (eu2 + 3.0f * AtmosVy[x][y] + 4.5f * vy2);
   Atmoseq[x][y].fE = 4.0f * Rho36 * (eu2 + 3.0f * AtmosVx[x][y] + 4.5f * vx2);
   Atmoseq[x][y].fS = 4.0f * Rho36 * (eu2 - 3.0f * AtmosVy[x][y] + 4.5f * vy2);
   Atmoseq[x][y].fW = 4.0f * Rho36 * (eu2 - 3.0f * AtmosVx[x][y] + 4.5f * vx2);
   Atmoseq[x][y].fNE = Rho36 * (eu2 + 3.0f * (AtmosVx[x][y] + AtmosVy[x][y]) + 4.5f * (AtmosVx[x][y] + AtmosVy[x][y]) * (AtmosVx[x][y] + AtmosVy[x][y]));
   Atmoseq[x][y].fSE = Rho36 * (eu2 + 3.0f * (AtmosVx[x][y] - AtmosVy[x][y]) + 4.5f * (AtmosVx[x][y] - AtmosVy[x][y]) * (AtmosVx[x][y] - AtmosVy[x][y]));
   Atmoseq[x][y].fSW = Rho36 * (eu2 + 3.0f * (-AtmosVx[x][y] - AtmosVy[x][y]) + 4.5f * (AtmosVx[x][y] + AtmosVy[x][y]) * (AtmosVx[x][y] + AtmosVy[x][y]));
   Atmoseq[x][y].fNW = Rho36 * (eu2 + 3.0f * (-AtmosVx[x][y] + AtmosVy[x][y]) + 4.5f * (-AtmosVx[x][y] + AtmosVy[x][y]) * (-AtmosVx[x][y] + AtmosVy[x][y]));

   // relaxation - output function
   Atmosout[x][y].fC = Atmosin[x][y].fC + (Atmoseq[x][y].fC - Atmosin[x][y].fC) / tauAtmos;
   Atmosout[x][y].fE = Atmosin[x][y].fE + (Atmoseq[x][y].fE - Atmosin[x][y].fE) / tauAtmos;
   Atmosout[x][y].fW = Atmosin[x][y].fW + (Atmoseq[x][y].fW - Atmosin[x][y].fW) / tauAtmos;
   Atmosout[x][y].fS = Atmosin[x][y].fS + (Atmoseq[x][y].fS - Atmosin[x][y].fS) / tauAtmos;
   Atmosout[x][y].fN = Atmosin[x][y].fN + (Atmoseq[x][y].fN - Atmosin[x][y].fN) / tauAtmos;
   Atmosout[x][y].fSE = Atmosin[x][y].fSE + (Atmoseq[x][y].fSE - Atmosin[x][y].fSE) / tauAtmos;
   Atmosout[x][y].fNE = Atmosin[x][y].fNE + (Atmoseq[x][y].fNE - Atmosin[x][y].fNE) / tauAtmos;
   Atmosout[x][y].fSW = Atmosin[x][y].fSW + (Atmoseq[x][y].fSW - Atmosin[x][y].fSW) / tauAtmos;
   Atmosout[x][y].fNW = Atmosin[x][y].fNW + (Atmoseq[x][y].fNW - Atmosin[x][y].fNW) / tauAtmos;
}

__global__ void StreamingAtmos()
{
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;

   // streaming
   Atmosin[x][y].fC = Atmosout[x][y].fC;
   if (x < Nx - 1) // EW
   {
      Atmosin[x][y].fW = Atmosout[x + 1][y].fW; Atmosin[x + 1][y].fE = Atmosout[x][y].fE;
   }
   if (y < Ny - 1) // NS
   {
      Atmosin[x][y].fS = Atmosout[x][y + 1].fS; Atmosin[x][y + 1].fN = Atmosout[x][y].fN;
   }
   if (x < Nx - 1 && y < Ny - 1) //D1
   {
      Atmosin[x][y].fSW = Atmosout[x + 1][y + 1].fSW; Atmosin[x + 1][y + 1].fNE = Atmosout[x][y].fNE;
   }
   if (x > 0 && y < Ny - 1) // D2
   {
      Atmosin[x][y].fSE = Atmosout[x - 1][y + 1].fSE; Atmosin[x - 1][y + 1].fNW = Atmosout[x][y].fNW;
   }

   // boundary conditions
   if (x == 0 || x == Nx - 1 || y == 0 || y == Ny - 1) {
      Atmosin[x][y].fE = Atmosout[x][y].fE; Atmosin[x][y].fNE = Atmosout[x][y].fNE; Atmosin[x][y].fSE = Atmosout[x][y].fSE;
      Atmosin[x][y].fW = Atmosout[x][y].fW; Atmosin[x][y].fSW = Atmosout[x][y].fSW; Atmosin[x][y].fNW = Atmosout[x][y].fNW;
      Atmosin[x][y].fN = Atmosout[x][y].fN; Atmosin[x][y].fS = Atmosout[x][y].fS;
   }
}
