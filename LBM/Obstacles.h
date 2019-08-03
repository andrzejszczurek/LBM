#ifndef Obstacles_h
#define Obstacles_h
#include "cuda_runtime.h"
#include "LBM0.h"

extern __device__ __managed__  int  AtmosState[Nx][Ny]; // 0 - air, 1 - obstacle, 2  - near obstacles

struct Obst { int x, y, typ1, typ2; }; // cell near obstacles: coordinates, configuration, orientation
extern __device__ __managed__ Obst CellO[4 * Nx];

int InitialObstacles()
{
   int x, y;
   int i = 0;
   int x0 = x = Nx / 2 - 5, x1 = Nx / 2 + 5;
   int y0 = 0, y1 = 80;

   for (x = x0; x < x1; x++) {
      for (y = y0; y < y1; y++)
      {
         AtmosState[x][y] = 1;
      };
   };

   y = y0;
   for (int x = 1; x < x0 - 1; x++) {
      AtmosState[x][0] = 2; CellO[i].x = x; CellO[i].y = y; CellO[i].typ1 = 1; CellO[i].typ2 = 4; i++;
   };

   x = x0 - 1; y = y0; AtmosState[x][y] = 2;
   CellO[i].x = x; CellO[i].y = y; CellO[i].typ1 = 5; CellO[i].typ2 = 3; i++;
   x = x0 - 1;
   for (int y = y0 + 1; y < y1 - 1; y++) {
      AtmosState[x][y] = 2; CellO[i].x = x; CellO[i].y = y; CellO[i].typ1 = 1; CellO[i].typ2 = 1; i++;
   };
   x = x0 - 1; y = y1 - 1; AtmosState[x][y] = 2;
   CellO[i].x = x; CellO[i].y = y; CellO[i].typ1 = 2; CellO[i].typ2 = 2; i++;
   x = x0 - 1; y = y1; AtmosState[x][y] = 2;
   CellO[i].x = x; CellO[i].y = y; CellO[i].typ1 = 4; CellO[i].typ2 = 3; i++;
   x = x0; y = y1; AtmosState[x][y] = 2;
   CellO[i].x = x; CellO[i].y = y; CellO[i].typ1 = 2; CellO[i].typ2 = 7; i++;
   y = y1;
   for (int x = x0 + 1; x < x1 - 1; x++) {
      AtmosState[x][y] = 2; CellO[i].x = x; CellO[i].y = y; CellO[i].typ1 = 1; CellO[i].typ2 = 4; i++;
   };
   x = x1 - 1; y = y1; AtmosState[x][y] = 2;
   CellO[i].x = x; CellO[i].y = y; CellO[i].typ1 = 2; CellO[i].typ2 = 8; i++;
   x = x1; y = y1; AtmosState[x][y] = 2;
   CellO[i].x = x; CellO[i].y = y; CellO[i].typ1 = 4; CellO[i].typ2 = 4; i++;
   x = x1; y = y1 - 1; AtmosState[x][y] = 2;
   CellO[i].x = x; CellO[i].y = y; CellO[i].typ1 = 2; CellO[i].typ2 = 4; i++;
   x = x1;
   for (int y = y0 + 1; y < y1 - 1; y++) {
      AtmosState[x][y] = 2; CellO[i].x = x; CellO[i].y = y; CellO[i].typ1 = 1; CellO[i].typ2 = 2; i++;
   };
   x = x1; y = y0; AtmosState[x][y] = 2;
   CellO[i].x = x; CellO[i].y = y; CellO[i].typ1 = 5; CellO[i].typ2 = 4; i++;
   y = y0;
   for (int x = x1 + 1; x < Nx - 2; x++)
   {
      AtmosState[x][0] = 2;
      CellO[i].x = x; CellO[i].y = y; CellO[i].typ1 = 1; CellO[i].typ2 = 4; i++;
   };

   return i;
}

__global__ void Obstacles(int a_nob)
{
   int i = threadIdx.y + blockIdx.y * blockDim.y;
   if (i >= a_nob) return;
   float muf = .0;  //5f;
   int x = CellO[i].x, y = CellO[i].y, typ1 = CellO[i].typ1, typ2 = CellO[i].typ2;
//
//   Atmosin[x][y].fC = Atmosout[x][y].fC;
//
//   switch (typ1) {
//   case 0: return; // obstacle
//   case 1:	// 	f2=f1;	f6=f5; f7=f8
//      switch (typ2) {
//      case 1: // east wall
//         Atmosin[x][y].fN = Atmosout[x][y - 1].fN; Atmosin[x][y].fS = Atmosout[x][y + 1].fS;
//         Atmosin[x][y].fE = Atmosin[x][y].fW = Atmosout[x - 1][y].fE;
//         Atmosin[x][y].fNE = Atmosout[x - 1][y - 1].fNE; Atmosin[x][y].fSE = Atmosout[x - 1][y + 1].fSE;
//         Atmosin[x][y].fNW = (1 - muf) * Atmosout[x - 1][y - 1].fNE + muf * Atmosout[x - 1][y + 1].fSE;
//         Atmosin[x][y].fSW = (1 - muf) * Atmosout[x - 1][y + 1].fSE + muf * Atmosout[x - 1][y - 1].fNE;
//         return;
//      case 2: // west wall
//         Atmosin[x][y].fN = Atmosout[x][y - 1].fN; Atmosin[x][y].fS = Atmosout[x][y + 1].fS;
//         Atmosin[x][y].fE = Atmosin[x][y].fW = Atmosout[x + 1][y].fW;
//         Atmosin[x][y].fNW = Atmosout[x + 1][y - 1].fNW; Atmosin[x][y].fSW = Atmosout[x + 1][y + 1].fSW;
//         Atmosin[x][y].fNE = (1 - muf) * Atmosout[x + 1][y - 1].fNW + muf * Atmosout[x + 1][y + 1].fSW;
//         Atmosin[x][y].fSE = (1 - muf) * Atmosout[x + 1][y + 1].fSW + muf * Atmosout[x + 1][y - 1].fNW;
//         return;
//      case 3: // nord wall
//         Atmosin[x][y].fE = Atmosout[x - 1][y].fE; Atmosin[x][y].fW = Atmosout[x + 1][y].fW;
//         Atmosin[x][y].fN = Atmosin[x][y].fS = Atmosout[x][y - 1].fN;
//         Atmosin[x][y].fNE = Atmosout[x - 1][y - 1].fNE; Atmosin[x][y].fNW = Atmosout[x + 1][y - 1].fNW;
//         Atmosin[x][y].fSE = (1 - muf) * Atmosout[x - 1][y - 1].fNE + muf * Atmosout[x + 1][y - 1].fNW;
//         Atmosin[x][y].fSW = (1 - muf) * Atmosout[x + 1][y - 1].fNW + muf * Atmosout[x - 1][y - 1].fNE;
//         return;
//      case 4: // south wall
//         Atmosin[x][y].fE = Atmosout[x - 1][y].fE; Atmosin[x][y].fW = Atmosout[x + 1][y].fW;
//         Atmosin[x][y].fN = Atmosin[x][y].fS = Atmosout[x][y + 1].fS;
//         Atmosin[x][y].fSE = Atmosout[x - 1][y + 1].fSE; Atmosin[x][y].fSW = Atmosout[x + 1][y + 1].fSW;
//         Atmosin[x][y].fNE = (1 - muf) * Atmosout[x - 1][y + 1].fSE + muf * Atmosout[x + 1][y + 1].fSW;
//         Atmosin[x][y].fNW = (1 - muf) * Atmosout[x + 1][y + 1].fSW + muf * Atmosout[x - 1][y + 1].fSE; ;
//         return;
//      }
//   case 2: // f2=f1+f4-f3+4(f6-f8); f7=f6-f8+f5+(f4-f3)/2; // f2=f1; f7=f5;
//      switch (typ2) {
//      case 1: // upper east wall
//         Atmosin[x][y].fE = Atmosout[x - 1][y].fE; Atmosin[x][y].fN = Atmosout[x][y - 1].fN; Atmosin[x][y].fS = Atmosout[x][y + 1].fS;
//         Atmosin[x][y].fW = Atmosout[x - 1][y].fE + (1 - muf) * (Atmosout[x][y + 1].fS - Atmosout[x][y - 1].fN + 4.0 * (Atmosout[x + 1][y - 1].fNW - Atmosout[x - 1][y + 1].fSE));
//         Atmosin[x][y].fSE = Atmosout[x - 1][y + 1].fSE; Atmosin[x][y].fNE = Atmosout[x - 1][y - 1].fNE;
//         Atmosin[x][y].fNW = Atmosout[x + 1][y - 1].fNW;
//         Atmosin[x][y].fSW = Atmosout[x - 1][y - 1].fNE + (1 - muf) * (Atmosout[x + 1][y - 1].fNW - Atmosout[x - 1][y + 1].fSE + (Atmosout[x][y + 1].fS - Atmosout[x][y - 1].fN) / 2.0);
//         return;
//      case 2: // bottom east wall
//         Atmosin[x][y].fE = Atmosout[x - 1][y].fE; Atmosin[x][y].fN = Atmosout[x][y - 1].fN; Atmosin[x][y].fS = Atmosout[x][y + 1].fS;
//         Atmosin[x][y].fW = Atmosout[x - 1][y].fE + (1 - muf) * (Atmosout[x][y - 1].fN - Atmosout[x][y + 1].fS + 4.0 * (Atmosout[x + 1][y + 1].fSW - Atmosout[x - 1][y + 1].fNE));
//         Atmosin[x][y].fNE = Atmosout[x - 1][y - 1].fNE;	Atmosin[x][y].fSE = Atmosout[x - 1][y + 1].fSE;
//         Atmosin[x][y].fSW = Atmosout[x + 1][y + 1].fSW;
//         Atmosin[x][y].fNW = Atmosout[x - 1][y + 1].fSE + (1 - muf) * (Atmosout[x + 1][y + 1].fSW - Atmosout[x - 1][y - 1].fNE + (Atmosout[x][y - 1].fN - Atmosout[x][y + 1].fS) / 2.0);
//         return;
//      case 3: // upper west wall
//         Atmosin[x][y].fW = Atmosout[x + 1][y].fW; Atmosin[x][y].fN = Atmosout[x][y - 1].fN; Atmosin[x][y].fS = Atmosout[x][y + 1].fS;
//         Atmosin[x][y].fE = Atmosout[x + 1][y].fW + (1 - muf) * (Atmosout[x][y + 1].fS - Atmosout[x][y - 1].fN + 4.0 * (Atmosout[x - 1][y - 1].fNE - Atmosout[x + 1][y + 1].fSW));
//         Atmosin[x][y].fSW = Atmosout[x + 1][y + 1].fSW; Atmosin[x][y].fNW = Atmosout[x + 1][y - 1].fNW;
//         Atmosin[x][y].fNE = Atmosout[x - 1][y - 1].fNE;
//         Atmosin[x][y].fSE = Atmosout[x + 1][y - 1].fNW + (1 - muf) * (Atmosout[x - 1][y - 1].fNE - Atmosout[x + 1][y + 1].fSW + (Atmosout[x][y + 1].fS - Atmosout[x][y - 1].fN) / 2.0);
//         return;
//      case 4:  // bottom west wall
//         Atmosin[x][y].fW = Atmosout[x + 1][y].fW; Atmosin[x][y].fN = Atmosout[x][y - 1].fN; Atmosin[x][y].fS = Atmosout[x][y + 1].fS;
//         Atmosin[x][y].fE = Atmosout[x + 1][y].fW + (1 - muf) * (Atmosout[x][y - 1].fN - Atmosout[x][y + 1].fS + 4.0 * (Atmosout[x - 1][y + 1].fSE - Atmosout[x + 1][y - 1].fNW));
//         Atmosin[x][y].fNW = Atmosout[x + 1][y - 1].fNW; Atmosin[x][y].fSW = Atmosout[x + 1][y + 1].fSW;
//         Atmosin[x][y].fSE = Atmosout[x - 1][y + 1].fSE;
//         Atmosin[x][y].fNE = Atmosout[x + 1][y + 1].fSW + (1 - muf) * (Atmosout[x - 1][y + 1].fSE - Atmosout[x + 1][y - 1].fNW + (Atmosout[x][y - 1].fN - Atmosout[x][y + 1].fS) / 2.0);
//         return;
//      case 5:  // right nord wall
//         Atmosin[x][y].fE = Atmosout[x - 1][y].fE; Atmosin[x][y].fN = Atmosout[x][y - 1].fN; Atmosin[x][y].fW = Atmosout[x + 1][y].fW;
//         Atmosin[x][y].fS = Atmosout[x][y - 1].fN + (1 - muf) * (Atmosout[x + 1][y].fW - Atmosout[x - 1][y].fE + 4.0 * (Atmosout[x - 1][y + 1].fSE - Atmosout[x + 1][y - 1].fNW));
//         Atmosin[x][y].fSE = Atmosout[x - 1][y + 1].fSE; Atmosin[x][y].fSW = Atmosout[x + 1][y + 1].fSW;
//         Atmosin[x][y].fNW = Atmosout[x + 1][y - 1].fNW;
//         Atmosin[x][y].fNE = Atmosout[x + 1][y + 1].fSW + (1 - muf) * (Atmosout[x - 1][y + 1].fSE - Atmosout[x + 1][y - 1].fNW + (Atmosout[x + 1][y].fW - Atmosout[x - 1][y].fE) / 2.0);
//         return;
//      case 6:  // left nord wall
//         Atmosin[x][y].fE = Atmosout[x - 1][y].fE; Atmosin[x][y].fN = Atmosout[x][y - 1].fN; Atmosin[x][y].fW = Atmosout[x + 1][y].fW;
//         Atmosin[x][y].fS = Atmosout[x][y - 1].fN + (1 - muf) * (Atmosout[x - 1][y].fE - Atmosout[x + 1][y].fW + 4.0 * (Atmosout[x + 1][y + 1].fSW - Atmosout[x - 1][y - 1].fNE));
//         Atmosin[x][y].fSE = Atmosout[x - 1][y + 1].fSE; Atmosin[x][y].fSW = Atmosout[x + 1][y + 1].fSW;
//         Atmosin[x][y].fNE = Atmosout[x - 1][y - 1].fNE;
//         Atmosin[x][y].fNW = Atmosout[x - 1][y + 1].fSE + (1 - muf) * (Atmosout[x + 1][y + 1].fSW - Atmosout[x - 1][y - 1].fNE + (Atmosout[x - 1][y].fE - Atmosout[x + 1][y].fW) / 2.0);
//         return;
//      case 7:  // right south wall
//         Atmosin[x][y].fE = Atmosout[x - 1][y].fE;  Atmosin[x][y].fW = Atmosout[x + 1][y].fW; Atmosin[x][y].fS = Atmosout[x][y + 1].fS;
//         Atmosin[x][y].fN = Atmosout[x][y + 1].fS + (1 - muf) * (Atmosout[x + 1][y].fW - Atmosout[x - 1][y].fE + 4.0 * (Atmosout[x - 1][y - 1].fNE - Atmosout[x + 1][y + 1].fSW));
//         Atmosin[x][y].fNE = Atmosout[x - 1][y - 1].fNE; Atmosin[x][y].fNW = Atmosout[x + 1][y - 1].fNW;
//         Atmosin[x][y].fSW = Atmosout[x + 1][y + 1].fSW;
//         Atmosin[x][y].fSE = Atmosout[x + 1][y - 1].fNW + (1 - muf) * (Atmosout[x - 1][y - 1].fNE - Atmosout[x + 1][y + 1].fSW + (Atmosout[x + 1][y].fW - Atmosout[x - 1][y].fE) / 2.0);
//         return;
//      case 8:  // left south wall
//         Atmosin[x][y].fE = Atmosout[x - 1][y].fE;  Atmosin[x][y].fW = Atmosout[x + 1][y].fW; Atmosin[x][y].fS = Atmosout[x][y + 1].fS;
//         Atmosin[x][y].fN = Atmosout[x][y + 1].fS + (1 - muf) * (Atmosout[x - 1][y].fE - Atmosout[x + 1][y].fW + 4.0 * (Atmosout[x + 1][y - 1].fNW - Atmosout[x - 1][y + 1].fSE));
//         Atmosin[x][y].fSE = Atmosout[x - 1][y + 1].fSE; Atmosin[x][y].fNE = Atmosout[x - 1][y - 1].fNE;
//         Atmosin[x][y].fNW = Atmosout[x + 1][y - 1].fNW;
//         Atmosin[x][y].fSW = Atmosout[x - 1][y - 1].fNE + (1 - muf) * (Atmosout[x + 1][y - 1].fNW - Atmosout[x - 1][y + 1].fSE + (Atmosout[x - 1][y].fE - Atmosout[x + 1][y].fW) / 2.0);
//         return;
//      }
   }
#endif