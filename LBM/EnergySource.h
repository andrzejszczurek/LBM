#ifndef EnergySource_h
#define EnergySource_h

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "LBM0.h"
#include "Temperature.h"

#define TempH1 0.2
#define TempH2 1.0
#define TempH3 1.0

__global__ void EnergySourceX()
{
    //  Temperature
    int x = threadIdx.y + blockIdx.y * blockDim.y;
    int y = 0;

    if (x >= Nx / 2 + 5)
        Temp[x][y] = TempH1;

    if (x < Nx / 2 + 5 && x >= Nx / 2 - 5)
        y = 80; Temp[x][y] = TempH3; 

    if (x <= Nx / 2 - 5) 
        y = 0; Temp[x][y] = TempH2; 

    //	distributions
    Teq[x][y].fC = Temp[x][y] * 4.0f / 9.0f;
    Teq[x][y].fE = Teq[x][y].fN = Teq[x][y].fS = Teq[x][y].fW = Temp[x][y] / 9.0f;
    Teq[x][y].fNE = Teq[x][y].fNW = Teq[x][y].fSE = Teq[x][y].fSW = Temp[x][y] / 36.0f;
    Tin[x][y].fC = Teq[x][y].fC;
    Tin[x][y].fE = Tin[x][y].fN = Tin[x][y].fS = Tin[x][y].fW = Teq[x][y].fE;
    Tin[x][y].fNE = Tin[x][y].fNW = Tin[x][y].fSE = Tin[x][y].fSW = Teq[x][y].fNE;
    Tout[x][y].fC = Teq[x][y].fC;
    Tout[x][y].fE = Tout[x][y].fN = Tout[x][y].fS = Tout[x][y].fW = Teq[x][y].fE;
    Tout[x][y].fNE = Tout[x][y].fNW = Tout[x][y].fSE = Tout[x][y].fSW = Teq[x][y].fNE;
}

__global__ void EnergySourceY()
{
    //  Temperature
    //int y = threadIdx.x + blockIdx.x * blockDim.x;
    //int x = 0;

    //if (y >= Ny / 2 + 5)
    //    Temp[x][y] = TempH1;

    //if (y < Ny / 2 + 5 && y >= Ny / 2 - 5)
    //{
    //    x = 80;
    //    Temp[x][y] = TempH3;
    //}

    //if (y <= Ny / 2 - 5)
    //{
    //    x = 0;
    //    Temp[x][y] = TempH2;
    //}

    ////	distributions
    //Teq[x][y].fC = Temp[x][y] * 4.0 / 9.0;
    //Teq[x][y].fE = Teq[x][y].fN = Teq[x][y].fS = Teq[x][y].fW = Temp[x][y] / 9.0f;
    //Teq[x][y].fNE = Teq[x][y].fNW = Teq[x][y].fSE = Teq[x][y].fSW = Temp[x][y] / 36.0f;
    //Tin[x][y].fC = Teq[x][y].fC;
    //Tin[x][y].fE = Tin[x][y].fN = Tin[x][y].fS = Tin[x][y].fW = Teq[x][y].fE;
    //Tin[x][y].fNE = Tin[x][y].fNW = Tin[x][y].fSE = Tin[x][y].fSW = Teq[x][y].fNE;
    //Tout[x][y].fC = Teq[x][y].fC;
    //Tout[x][y].fE = Tout[x][y].fN = Tout[x][y].fS = Tout[x][y].fW = Teq[x][y].fE;
    //Tout[x][y].fNE = Tout[x][y].fNW = Tout[x][y].fSE = Tout[x][y].fSW = Teq[x][y].fNE;
}


#endif