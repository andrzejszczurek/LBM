#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "LBM_Atmos.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <Gl/glut.h>
#include "LBM0.h"
#include "Visualization.h"
#include <string.h>

void DrawAtmosHor(char rgba[Ny][Nx][3])
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
         if (AtmosVx[i][j] > 0.001) { CellRed = 10.f * AtmosVx[i][j]; CellBlue = 0.f;  CellGreen = 0.f; }
         if (AtmosVx[i][j] < -0.001) { CellBlue = -10.f * AtmosVx[i][j]; CellRed = 0.f;  CellGreen = 0.f; }

         rgba[j][i][0] = 255 * CellBlue; rgba[j][i][1] = 255 * CellGreen; rgba[j][i][2] = 255 * CellRed;

         glColor3f(CellRed, CellGreen, CellBlue);
         glBegin(GL_QUADS);
         glVertex3f(i, j, 0.0);
         glVertex3f(i, j + 1, 0.0);
         glVertex3f(i + 1, j + 1, 0.0);
         glVertex3f(i + 1, j, 0.0);
         glEnd();
      }
   }
   glutSwapBuffers();

}

void DrawAtmosVer()
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
         if (AtmosVy[i][j] > 0.001) { CellRed = 10.f * AtmosVy[i][j]; CellBlue = 0.f;  CellGreen = 0.f; }
         if (AtmosVy[i][j] < -0.001) { CellBlue = -10.f * AtmosVy[i][j]; CellRed = 0.f;  CellGreen = 0.f; }

         glColor3f(CellRed, CellGreen, CellBlue);
         glBegin(GL_QUADS);
         glVertex3f(i, j, 0.0);
         glVertex3f(i, j + 1, 0.0);
         glVertex3f(i + 1, j + 1, 0.0);
         glVertex3f(i + 1, j, 0.0);
         glEnd();
      }
   }
   glutSwapBuffers();
}

__global__ void InitialAtmos()
{
   unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
   // Initial density and velocity, equilibrium and input distribution functions 
   if (x == 0 || x == Nx - 1 || y == 0 || y == Ny - 1) {
      AtmosVx[x][y] = (1.0f * y / Ny) * 0.10; // velocity from 0 to 0.1
   }
   else { AtmosVx[x][y] = 0.0f; }
   AtmosVy[x][y] = 0.0f;
   AtmosRho[x][y] = 1.0f;

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

   // macroscopic concentration of Nb and C for the current cell
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
