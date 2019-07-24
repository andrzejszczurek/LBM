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


std::ofstream myfile;	// file with output data
std::fstream infile;	// file with input data
float timev;			// real time from cycle beginning
int npasses, ipass;		// number of cycles and cycle number
float timecycle[40], temp1[40], stress[40], Trate[40];	//duration of cycle; temperature; stress; cooling rate ????????????????

int mainLBM(bool FirstCycle)
{
   using namespace std;
   dim3 block(8, 8, 1);
   dim3 grid(Nx / block.x, Ny / block.y, 1);

   if (FirstCycle)
   {
      FirstCycle = false;
      InitialAtmos << < grid, block >> > ();
      cudaDeviceSynchronize();
      timem = 0.0f; timev = 0.0f;

      //ofstream myfile;
      myfile.open("time.txt", ios::out | ios::trunc);
      myfile << "time     temperature   N_Per   krok czasowy    D_Nb" << endl;
      //myfile.close();
      infile.open("Schedule.txt", ios::in);
      infile >> npasses; ipass = 0;	// read number of cycles
      infile >> temp1[0]; temp1[0] = temp1[0] + 273; // read initial temperature
      for (int i = 0; i < npasses; i++)
      {
         infile >> timecycle[i] >> temp1[i + 1] >> stress[i]; temp1[i + 1] = temp1[i + 1] + 273; //read duration, final temperature and stress for the cycle
         Trate[i] = (temp1[i + 1] - temp1[i]) / timecycle[i]; // cooling rate
      }
      infile.close();
   }
   else
   {
      //while (timev < TimeViz)
      while ((timev < timecycle[ipass]) && (ipass < npasses))
         //		for (int it = 0; it < NViz; it++)
      {
         Temp_Per = 1173.f;
         Temp_Per = 1223.f - 2.0f * timem;
         Temp_Per = temp1[ipass] + Trate[ipass] * timev; // temperature
         if (Temp_Per < 1023.f) { exit(0); };	// exit if temperature is below transformation temperature
         RT = 8.314 * Temp_Per, kBolT = k_Bol * Temp_Per;
         D_Nb = 0.00094 * expf(-290000 / RT);	// coefficient of niobium diffusion
         stept = (tauNb - 0.5) * Lcell * Lcell / 2.0 / D_Nb;	// time step
// diffusion and precipitation calculations
         dP_convex = 2 * sigma_Nb * V_NbC / RT;
         float N_0 = (stress[ipass] / 4. / 2.59 * 2.5e7);
         N_0 = 0.5f * N_0 * N_0 * N_0;
         NucSites = N_0; // number of possible nucleation sites [m-3]
         lks0 = logf(10.f) * (2.26 - 6770 / Temp_Per);
         ks0 = expf(lks0);
         EquiRelaxCon << < grid, block >> > ();
         cudaDeviceSynchronize();
         FS1 = Control1; FL1 = Control2;
         StreamingCon << < grid, block >> > ();
         cudaDeviceSynchronize();
         timem += stept; timev += stept;
      }
      { timev = 0.0f; ipass++; }
   }
   myfile << timem << "    " << Temp_Per - 273 << "       " << NPer << "     " << stept << "    " << D_Nb << endl;
   if (ipass >= npasses) { exit(0); };	// exit if it is the last cycle
   return 0;
}
