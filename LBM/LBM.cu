#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Gl/glut.h>
#include <string.h>

#include "LBM0.h"
#include "Visualization.h"
#include "LBM_Atmos.h"
#include "BC.h"
#include "Obstacles.h"
#include "Temperature.h"
#include "EnergySource.h"


std::fstream infile;	// file with input data
float timev;			// real time from cycle beginning
int npasses, ipass;		// number of cycles and cycle number
float timecycle[10000];		//duration of cycle
static bool newSim = true;
int obstaclesI = 0;

int mainLBM(bool FirstCycle)
{
   using namespace std;
   dim3 block(1, 8, 1); // it could be different
   dim3 grid(Nx / block.x, Ny / block.y, 1);

   // dla warunków brzegowych
   dim3 gridX(1, Nx / block.y, 1);
   dim3 gridY(1, Ny / block.y, 1);

   if (FirstCycle)
   {
      FirstCycle = false;

      if (!newSim)
      {
          newSim = false;
          InRes();
      }
      obstaclesI = InitialObstacles();
      InitialAtmos <<< grid, block >>>(newSim);
      InitialTempG << < grid, block >> > ();
      cudaDeviceSynchronize();
      //EnergySourceX << < gridX, block >> > ();
      //EnergySourceY << < gridY, block >> > ();
      //cudaDeviceSynchronize();

      timem = 0.0f; 
      timev = 0.0f;

      //infile.open("Schedule.txt", ios::in);
      //infile >> npasses; ipass = 0;	// read number of cycles
      //for (int i = 0; i < npasses; i++) { infile >> timecycle[i]; }j

      //infile.close();

      // alternatywa dla cykli z pliku
      ipass = 0;
      npasses = 10000;
      for (int i = 0; i < npasses; i++)
      {
         timecycle[i] = 1.0f/250;
      }

      return 0;
   }
   else
   {
      while ((timev < timecycle[ipass]) && (ipass < npasses))
      {
         stept = 0.01;	// time step
         EquiRelaxAtmos << < grid, block >> > ();
         cudaDeviceSynchronize();
         EquiRelaxTempG << < grid, block >> > ();
         cudaDeviceSynchronize();

         //EnergySourceX << < gridX, block >> > ();
         //EnergySourceY << < gridY, block >> > ();
         //cudaDeviceSynchronize();

         StreamingAtmos << < grid, block >> > ();
         StreamingTempG << < grid, block >> > ();
         cudaDeviceSynchronize();

         BoundaryEast << < gridX, block >> > ();
         BoundaryWest << < gridX, block >> > ();
         BoundarySouth << < gridY, block >> > ();
         BoundaryNord << < gridY, block >> > ();
         cudaDeviceSynchronize();

         /*dim3 gridOb(1, obstaclesI + 2 / block.y, 1);
         Obstacles << < gridOb, block >> > (obstaclesI);
         cudaDeviceSynchronize();*/

         BoundaryEastTemp << < gridX, block >> > ();
         BoundaryWestTemp << < gridX, block >> > ();
         BoundarySouthTemp << < gridY, block >> > ();
         BoundaryNordTemp << < gridY, block >> > ();
         cudaDeviceSynchronize();

         timem += stept; 
         timev += stept;
      }
      { timev = 0.0f; ipass++; }
   }
   if (ipass >= npasses) 
   { 
       // zapis wyniku ostatniego cyklu do pliku
       OutRes();
       return 1;
   };	// exit if it is the last cycle
   return 0;
}

void OutRes()
{
    std::ofstream ResultsOut;
    ResultsOut.open("ResultsOut.txt", std::ios::out | std::ios::trunc);
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            ResultsOut << AtmosRho[i][j] << "    " << AtmosVx[i][j] << "    " << AtmosVy[i][j] << std::endl;
        }
    }
    ResultsOut.close();
}

void InRes()
{
    std::fstream ResultsOut;
    ResultsOut.open("ResultsOut.txt", std::ios::in);
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            ResultsOut >> AtmosRho[i][j] >> AtmosVx[i][j] >> AtmosVy[i][j];	// read data
        }
    }
    ResultsOut.close();
}
