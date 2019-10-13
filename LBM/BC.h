#ifndef BC_h
#define BC_h
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

#include "LBM0.h"


#define BoundaryCEast 4 // 1- bounce-back, 2 - symmetry, 3 - velocity, 4 - rho
#define BoundaryCWest 3
#define BoundaryCNord 3
#define BoundaryCSouth 1


__global__ void BoundaryEast()
{
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    Atmosin[Nx - 1][y].fC = Atmosout[Nx - 1][y].fC;
    Atmosin[Nx - 1][y].fE = Atmosout[Nx - 2][y].fE;

    if (y < Ny - 1)
    {
        Atmosin[Nx - 1][y].fS = Atmosout[Nx - 1][y + 1].fS;
        Atmosin[Nx - 1][y].fSE = Atmosout[Nx - 2][y + 1].fSE;
    }
    else
    {
        Atmosin[Nx - 1][y].fS = Atmosout[Nx - 1][y - 1].fN;
        Atmosin[Nx - 1][y].fSE = Atmosout[Nx - 2][y - 1].fNE;
    }

    if (y > 0)
    {
        Atmosin[Nx - 1][y].fN = Atmosout[Nx - 1][y - 1].fN;
        Atmosin[Nx - 1][y].fNE = Atmosout[Nx - 2][y - 1].fNE;
    }
    else
    {
        Atmosin[Nx - 1][y].fN = Atmosout[Nx - 1][y + 1].fS;
        Atmosin[Nx - 1][y].fNE = Atmosout[Nx - 2][y + 1].fSE;
    }

    int BoundaryCode = BoundaryCEast;
    float RhoExit, Ux, Uy;
    switch (BoundaryCode)
    {
    case 1:	////////////////// bounce-back
        Atmosin[Nx - 1][y].fW = Atmosin[Nx - 1][y].fE;
        Atmosin[Nx - 1][y].fSW = Atmosin[Nx - 1][y].fNE;
        Atmosin[Nx - 1][y].fNW = Atmosin[Nx - 1][y].fSE;
        return;
    case 2:	////////////////// symmetry
        Atmosin[Nx - 1][y].fW = Atmosin[Nx - 1][y].fE;
        Atmosin[Nx - 1][y].fSW = Atmosin[Nx - 1][y].fSE;
        Atmosin[Nx - 1][y].fNW = Atmosin[Nx - 1][y].fNE;
        return;
    case 3:	////////////////// velosity
        Ux = 0.02, Uy = 0.0;
        RhoExit = (Atmosin[Nx - 1][y].fC + Atmosin[Nx - 1][y].fS + Atmosin[Nx - 1][y].fN + 2.0 * (Atmosin[Nx - 1][y].fE + Atmosin[Nx - 1][y].fSE + Atmosin[Nx - 1][y].fNE)) / (1.0 + Ux);
        break;
    case 4:	///////////////// open (rho)
        RhoExit = 1.0; Uy = 0.0;
        Ux = (Atmosin[Nx - 1][y].fC + Atmosin[Nx - 1][y].fS + Atmosin[Nx - 1][y].fN + 2.0 * (Atmosin[Nx - 1][y].fE + Atmosin[Nx - 1][y].fSE + Atmosin[Nx - 1][y].fNE)) / RhoExit - 1.0;
        break;
    }
    Uy = 6.f * (Atmosin[Nx - 1][y].fN - Atmosin[Nx - 1][y].fS + Atmosin[Nx - 1][y].fNE - Atmosin[Nx - 1][y].fSE) / RhoExit / (5 - 3 * Ux);

    Atmosin[Nx - 1][y].fW = Atmosin[Nx - 1][y].fE - 2.0 / 3.0 * Ux * RhoExit;
    Atmosin[Nx - 1][y].fSW = Atmosin[Nx - 1][y].fNE - (Ux - Uy) / 6.0 * RhoExit;
    Atmosin[Nx - 1][y].fNW = Atmosin[Nx - 1][y].fSE - (Ux + Uy) / 6.0 * RhoExit;
}

__global__ void BoundaryWest()
{
   int y = threadIdx.y + blockIdx.y * blockDim.y;

   Atmosin[0][y].fC = Atmosout[0][y].fC;
   Atmosin[0][y].fW = Atmosout[1][y].fW;
   if (y < Ny - 1) { Atmosin[0][y].fS = Atmosout[0][y + 1].fS; Atmosin[0][y].fSW = Atmosout[1][y + 1].fSW; }
   else { Atmosin[0][y].fS = Atmosout[0][y - 1].fN; Atmosin[0][y].fSW = Atmosout[1][y - 1].fNW; }
   if (y > 0) { Atmosin[0][y].fN = Atmosout[0][y - 1].fN; Atmosin[0][y].fNW = Atmosout[1][y - 1].fNW; }
   else { Atmosin[0][y].fN = Atmosout[0][y + 1].fS; Atmosin[0][y].fNW = Atmosout[1][y + 1].fSW; }

   int BoundaryCode = BoundaryCWest;
   float RhoExit, Ux, Uy;
   switch (BoundaryCode)
   {
   case 1:	////////////////// bounce-back
      Atmosin[0][y].fE = Atmosin[0][y].fW; Atmosin[0][y].fNE = Atmosin[0][y].fSW; Atmosin[0][y].fSE = Atmosin[0][y].fNW;
      return;
   case 2:	////////////////// symmetry
      Atmosin[0][y].fE = Atmosin[0][y].fW; Atmosin[0][y].fNE = Atmosin[0][y].fNW; Atmosin[0][y].fSE = Atmosin[0][y].fSW;
      return;
   case 3:	////////////////// velosity
      Ux = (1.0f * y / Ny) * 0.05; // velocity from 0 to 0.05
      RhoExit = (Atmosin[0][y].fC + Atmosin[0][y].fS + Atmosin[0][y].fN + 2.0 * (Atmosin[0][y].fW + Atmosin[0][y].fSW + Atmosin[0][y].fNW)) / (1.0 - Ux);
      break;
   case 4:	///////////////// open (rho)
      RhoExit = 1.0; Uy = 0.0;
      Ux = 1.0 - (Atmosin[0][y].fC + Atmosin[0][y].fS + Atmosin[0][y].fN + 2.0 * (Atmosin[0][y].fW + Atmosin[0][y].fSW + Atmosin[0][y].fNW)) / RhoExit;
      break;
   }
   Uy = 6.f * (Atmosin[0][y].fN - Atmosin[0][y].fS + Atmosin[0][y].fNW - Atmosin[0][y].fSW) / RhoExit / (5 + 3 * Ux);
   Atmosin[0][y].fE = Atmosin[0][y].fW + 2.0 / 3.0 * Ux * RhoExit;
   Atmosin[0][y].fNE = Atmosin[0][y].fSW + (Ux - Uy) / 6.0 * RhoExit;
   Atmosin[0][y].fSE = Atmosin[0][y].fNW + (Ux + Uy) / 6.0 * RhoExit;
}

__global__ void BoundarySouth()
{
   int x = threadIdx.y + blockIdx.y * blockDim.y;

   Atmosin[x][0].fC = Atmosout[x][0].fC;
   Atmosin[x][0].fS = Atmosout[x][1].fS;
   if (x < Nx - 1) { Atmosin[x][0].fW = Atmosout[x + 1][0].fW; Atmosin[x][0].fSW = Atmosout[x + 1][1].fSW; }
   else { return; }
   if (x > 0) { Atmosin[x][0].fE = Atmosout[x - 1][0].fE; Atmosin[x][0].fSE = Atmosout[x - 1][1].fSE; }
   else { return; }

   int BoundaryCode = BoundaryCSouth;
   float RhoExit, Ux, Uy;
   switch (BoundaryCode)
   {
   case 1:	////////////////// bounce-back
      Atmosin[x][0].fN = Atmosout[x][0].fS; Atmosin[x][0].fNW = Atmosout[x][0].fSE; Atmosin[x][0].fNE = Atmosout[x][0].fSW;
      return;
   case 2:	////////////////// symmetry
      Atmosin[x][0].fN = Atmosout[x][0].fS; Atmosin[x][0].fNW = Atmosout[x][0].fSW; Atmosin[x][0].fNE = Atmosout[x][0].fSE;
      return;
   case 3:	////////////////// velosity
      Ux = 0.02; Uy = 0.0;
      RhoExit = (Atmosin[x][0].fC + Atmosin[x][0].fE + Atmosin[x][0].fW + 2.0 * (Atmosin[x][0].fS + Atmosin[x][0].fSW + Atmosin[x][0].fSE)) / (1.0 - Uy);
      break;
   case 4:	////////////////// open
      RhoExit = 1.0; Uy = 0.0;
      Uy = 1.0 - (Atmosin[x][0].fC + Atmosin[x][0].fE + Atmosin[x][0].fW + 2.0 * (Atmosin[x][0].fS + Atmosin[x][0].fSW + Atmosin[x][0].fSE)) / RhoExit;
      break;
   }
   Atmosin[x][0].fN = Atmosin[x][0].fS + 2.0 / 3.0 * Uy * RhoExit;
   Atmosin[x][0].fNW = Atmosin[x][0].fSE + 0.5 * (Atmosin[x][0].fE - Atmosin[x][0].fW) + Uy * RhoExit / 6.0 - 0.5 * Ux * RhoExit;
   Atmosin[x][0].fNE = Atmosin[x][0].fSW - 0.5 * (Atmosin[x][0].fE - Atmosin[x][0].fW) + Uy * RhoExit / 6.0 + 0.5 * Ux * RhoExit;
}

__global__ void BoundaryNord()
{
   int x = threadIdx.y + blockIdx.y * blockDim.y;

   Atmosin[x][Ny - 1].fC = Atmosout[x][Ny - 1].fC;
   Atmosin[x][Ny - 1].fN = Atmosout[x][Ny - 2].fN;
   if (x < Nx - 1) { Atmosin[x][Ny - 1].fW = Atmosout[x + 1][Ny - 1].fW; Atmosin[x][Ny - 1].fNW = Atmosout[x + 1][Ny - 2].fNW; }
   else { return; }
   if (x > 0) { Atmosin[x][Ny - 1].fE = Atmosout[x - 1][Ny - 1].fE; Atmosin[x][Ny - 1].fNE = Atmosout[x - 1][Ny - 2].fNE; }
   else { return; }

   int BoundaryCode = BoundaryCNord;
   float RhoExit, Ux, Uy;
   switch (BoundaryCode)
   {
   case 1:	////////////////// bounce-back
      Atmosin[x][Ny - 1].fS = Atmosout[x][Ny - 1].fN; Atmosin[x][Ny - 1].fSW = Atmosout[x][Ny - 1].fNE; Atmosin[x][Ny - 1].fSE = Atmosout[x][Ny - 1].fNW;
      return;
   case 2:	////////////////// symmetry
      Atmosin[x][Ny - 1].fS = Atmosout[x][Ny - 1].fN; Atmosin[x][Ny - 1].fSW = Atmosout[x][Ny - 1].fNW; Atmosin[x][Ny - 1].fSE = Atmosout[x][Ny - 1].fNE;
      return;
   case 3:	////////////////// velosity
      Ux = 0.05; Uy = 0.0;
      RhoExit = (Atmosin[x][Ny - 1].fC + Atmosin[x][Ny - 1].fE + Atmosin[x][Ny - 1].fW + 2.0 * (Atmosin[x][Ny - 1].fN + Atmosin[x][Ny - 1].fNW + Atmosin[x][Ny - 1].fNE)) / (1.0 + Uy);
      break;
   case 4:	////////////////// open (rho)
      RhoExit = 1.0; Uy = 0.0;
      Uy = -1.0 + (Atmosin[x][Ny - 1].fC + Atmosin[x][Ny - 1].fE + Atmosin[x][Ny - 1].fW + 2.0 * (Atmosin[x][Ny - 1].fN + Atmosin[x][Ny - 1].fNW + Atmosin[x][Ny - 1].fNE)) / RhoExit;
      break;
   }
   Atmosin[x][Ny - 1].fS = Atmosin[x][Ny - 1].fN - 2.0 / 3.0 * Uy * RhoExit;
   Atmosin[x][Ny - 1].fSW = Atmosin[x][Ny - 1].fNE - (Ux + Uy) / 6.0 * RhoExit;
   Atmosin[x][Ny - 1].fSE = Atmosin[x][Ny - 1].fNW + (Ux - Uy) / 6.0 * RhoExit;
}





#endif // BC_h