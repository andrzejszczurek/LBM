#ifndef LBM0_h
#define LBM0_h
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

#define Nx 256	// horizontal size of the space
#define Ny 256	// vertical size of the space
#define tauAtmos 1.0f	// relaxation time constant

extern __device__ __managed__  float timem, stept; // overall time, time step
struct Dist { float fC, fE, fW, fS, fN, fSE, fSW, fNE, fNW; }; // distribution function D2Q9
extern __device__ __managed__  float AtmosRho[Nx][Ny]; // density
extern __device__ __managed__  float AtmosVx[Nx][Ny], AtmosVy[Nx][Ny]; // horizontal and vertical velocities
extern __device__ __managed__  Dist  Atmosin[Nx][Ny], Atmosout[Nx][Ny], Atmoseq[Nx][Ny]; // input, output and equilibrium distribution functions

__global__ void InitialAtmos();
__global__ void StreamingAtmos();
__global__ void EquiRelaxAtmos();
int mainLBM(bool FirstCycle);
#endif // LBM0_h
