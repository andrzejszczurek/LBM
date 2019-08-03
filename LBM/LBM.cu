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


std::fstream infile;	// file with input data
float timev;			// real time from cycle beginning
int npasses, ipass;		// number of cycles and cycle number
float timecycle[40];		//duration of cycle

int mainLBM(bool FirstCycle)
{
   using namespace std;
   dim3 block(8, 8, 1); // it could be different
   dim3 grid(Nx / block.x, Ny / block.y, 1);

   dim3 gridX(1, Nx / block.y, 1);
   dim3 gridY(1, Ny / block.y, 1);

   int Nob = 10;

   if (FirstCycle)
   {
      FirstCycle = false;
      InitialAtmos << < grid, block >> > ();
      cudaDeviceSynchronize();
      Nob = InitialObstacles();
      timem = 0.0f; timev = 0.0f;

      //infile.open("Schedule.txt", ios::in);
      //infile >> npasses; ipass = 0;	// read number of cycles
      //for (int i = 0; i < npasses; i++) { infile >> timecycle[i]; }
      //infile.close();

      ipass = 0;
      npasses = 100;
      for (int i = 0; i < npasses; i++)
      {
         timecycle[i] = 1.0f/5;
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

         Obstacles << < gridY, block >> > (Nob);
         cudaDeviceSynchronize();

         StreamingAtmos << < grid, block >> > ();
         cudaDeviceSynchronize();

         BoundaryEast << < gridX, block >> > ();
         BoundaryWest << < gridX, block >> > ();
         BoundarySouth << < gridY, block >> > ();
         BoundaryNord << < gridY, block >> > ();
         cudaDeviceSynchronize();

         timem += stept; timev += stept;
      }
      { timev = 0.0f; ipass++; }
   }
   if (ipass >= npasses) { return 1; };	// exit if it is the last cycle
   return 0;
}
