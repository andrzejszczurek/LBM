#ifndef Temperature_h
#define Temperature_h
#include "cuda_runtime.h"
#include "LBM0.h"

#define tauT 1.0f

extern __device__ __managed__  float Temp[Nx][Ny];
extern __device__ __managed__  Dist  Tin[Nx][Ny];
extern __device__ __managed__  Dist Tout[Nx][Ny];
extern __device__ __managed__  Dist Teq[Nx][Ny];


__global__ void InitialTempG()
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    Temp[x][y] = .5f; // Initial Temperature
    //	distributions
    Teq[x][y].fC = Temp[x][y] * 4.0f / 9.0f;
    Teq[x][y].fE = Teq[x][y].fN = Teq[x][y].fS = Teq[x][y].fW = Temp[x][y] / 9.0f;
    Teq[x][y].fNE = Teq[x][y].fNW = Teq[x][y].fSE = Teq[x][y].fSW = Temp[x][y] / 36.0f;

    Tin[x][y].fC = Teq[x][y].fC;
    Tin[x][y].fE = Tin[x][y].fN = Tin[x][y].fS = Tin[x][y].fW = Teq[x][y].fE;
    Tin[x][y].fNE = Tin[x][y].fNW = Tin[x][y].fSE = Tin[x][y].fSW = Teq[x][y].fNE;
}

__global__ void EquiRelaxTempG()
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // macroscopic Temperature for the current cell
    Temp[x][y] = Tin[x][y].fC + Tin[x][y].fE + Tin[x][y].fW + Tin[x][y].fS + Tin[x][y].fN + Tin[x][y].fNE + Tin[x][y].fNW + Tin[x][y].fSE + Tin[x][y].fSW;

    // equillibrum
    float Temp36 = Temp[x][y] / 36.0f;
    Teq[x][y].fC = 16.0f * Temp36;
    Teq[x][y].fN = 4.0f * Temp36 * (1.0f + 3.0f * AtmosVy[x][y]);
    Teq[x][y].fE = 4.0f * Temp36 * (1.0f + 3.0f * AtmosVx[x][y]);
    Teq[x][y].fS = 4.0f * Temp36 * (1.0f - 3.0f * AtmosVy[x][y]);
    Teq[x][y].fW = 4.0f * Temp36 * (1.0f - 3.0f * AtmosVx[x][y]);
    Teq[x][y].fNE = Temp36 * (1.0f + 3.0f * (AtmosVx[x][y] + AtmosVy[x][y]));
    Teq[x][y].fSE = Temp36 * (1.0f + 3.0f * (AtmosVx[x][y] - AtmosVy[x][y]));
    Teq[x][y].fSW = Temp36 * (1.0f + 3.0f * (-AtmosVx[x][y] - AtmosVy[x][y]));
    Teq[x][y].fNW = Temp36 * (1.0f + 3.0f * (-AtmosVx[x][y] + AtmosVy[x][y]));

    // relaxation
    Tout[x][y].fC = Tin[x][y].fC + (Teq[x][y].fC - Tin[x][y].fC) / tauT;
    Tout[x][y].fE = Tin[x][y].fE + (Teq[x][y].fE - Tin[x][y].fE) / tauT;
    Tout[x][y].fW = Tin[x][y].fW + (Teq[x][y].fW - Tin[x][y].fW) / tauT;
    Tout[x][y].fS = Tin[x][y].fS + (Teq[x][y].fS - Tin[x][y].fS) / tauT;
    Tout[x][y].fN = Tin[x][y].fN + (Teq[x][y].fN - Tin[x][y].fN) / tauT;
    Tout[x][y].fSE = Tin[x][y].fSE + (Teq[x][y].fSE - Tin[x][y].fSE) / tauT;
    Tout[x][y].fNE = Tin[x][y].fNE + (Teq[x][y].fNE - Tin[x][y].fNE) / tauT;
    Tout[x][y].fSW = Tin[x][y].fSW + (Teq[x][y].fSW - Tin[x][y].fSW) / tauT;
    Tout[x][y].fNW = Tin[x][y].fNW + (Teq[x][y].fNW - Tin[x][y].fNW) / tauT;
}

__global__ void StreamingTempG()
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    Tin[x][y].fC = Tout[x][y].fC;

    if (x < Nx - 1) // EW
    {
        Tin[x][y].fW = Tout[x + 1][y].fW;
        Tin[x + 1][y].fE = Tout[x][y].fE;
    }

    if (y < Ny - 1) // NS
    {
        Tin[x][y].fS = Tout[x][y + 1].fS;
        Tin[x][y + 1].fN = Tout[x][y].fN;
    }

    if (x < Nx - 1 && y < Ny - 1) //D1
    {
        Tin[x][y].fSW = Tout[x + 1][y + 1].fSW;
        Tin[x + 1][y + 1].fNE = Tout[x][y].fNE;
    }

    if (x > 0 && y < Ny - 1) // D2
    {
        Tin[x][y].fSE = Tout[x - 1][y + 1].fSE;
        Tin[x - 1][y + 1].fNW = Tout[x][y].fNW;
    }
}

__global__ void BoundaryEastTemp()
{
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = Nx - 1;
    float TempConst = (1.0f - 1.0f * y / Ny) * .2f / 18.0f; //temperature from 0 to 0.2
    float alphaT = 0.f;

    Tin[x][y].fC = Tout[x][y].fC;
    Tin[x][y].fE = Tout[x - 1][y].fE;

    if (y < Ny - 1)
    {
        Tin[x][y].fS = Tout[x][y + 1].fS;
        Tin[x][y].fSE = Tout[x - 1][y + 1].fSE;
    }
    else
    {
        Tin[x][y].fS = Tout[x][y - 1].fN;
        Tin[x][y].fSE = Tout[x - 1][y - 1].fNE;
    }

    if (y > 0)
    {
        Tin[x][y].fN = Tout[x][y - 1].fN;
        Tin[x][y].fNE = Tout[x - 1][y - 1].fNE;
    }
    else
    {
        Tin[x][y].fN = Tout[x][y + 1].fS;
        Tin[x][y].fNE = Tout[x - 1][y + 1].fSE;
    }

    Tin[x][y].fW = (1.f - alphaT) * Tin[x][y].fE + alphaT * (4.0f * TempConst - Tin[x][y].fE);
    Tin[x][y].fSW = (1.f - alphaT) * Tin[x][y].fNE + alphaT * (TempConst - Tin[x][y].fNE);
    Tin[x][y].fNW = (1.f - alphaT) * Tin[x][y].fSE + alphaT * (TempConst - Tin[x][y].fSE);
    //return;
    Temp[x][y] = (Tin[0][y].fC + Tin[0][y].fE + Tin[0][y].fS + 2 * (Tin[0][y].fE + Tin[0][y].fNE + Tin[0][y].fSE)) / (1 - AtmosVx[0][y]);
    Tin[0][y].fW = Tin[0][y].fE + 2.f / 3.f * Temp[0][y] * AtmosVx[0][y];
    Tin[0][y].fNW = Tin[0][y].fNE + Temp[0][y] * AtmosVx[0][y] / 6.f;
    Tin[0][y].fSW = Tin[0][y].fSE + Temp[0][y] * AtmosVx[0][y] / 6.f;
    return;
}

__global__ void BoundaryWestTemp()
{
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int x = 0;
    float TempConst = (1.0f - 1.0f * (Ny-y) / Ny) * 0.1f / 18.f;//temperature from 0 to 0.2
    float alphaT = 1.f;

    Tin[0][y].fC = Tout[0][y].fC;
    Tin[0][y].fW = Tout[1][y].fW;
    if (y < Ny - 1) 
    { 
       Tin[0][y].fS = Tout[0][y + 1].fS;
       Tin[0][y].fSW = Tout[1][y + 1].fSW; 
    }
    else 
    { 
       Tin[0][y].fS = Tout[0][y - 1].fN;
       Tin[0][y].fSW = Tout[1][y - 1].fNW; 
    }

    if (y > 0) 
    { 
       Tin[0][y].fN = Tout[0][y - 1].fN;
       Tin[0][y].fNW = Tout[1][y - 1].fNW; 
    }
    else 
    { 
       Tin[0][y].fN = Tout[0][y + 1].fS;
       Tin[0][y].fNW = Tout[1][y + 1].fSW; 
    }

    Tin[0][y].fE = (1.f - alphaT) * Tin[0][y].fW + alphaT * (4.0f * TempConst - Tin[0][y].fW);
    Tin[0][y].fNE = (1.f - alphaT) * Tin[0][y].fSW + alphaT * (TempConst - Tin[0][y].fSW);
    Tin[0][y].fSE = (1.f - alphaT) * Tin[0][y].fNW + alphaT * (TempConst - Tin[0][y].fNW);
    return;

    Temp[x][y] = (Tin[0][y].fC + Tin[0][y].fN + Tin[0][y].fS + 2 * (Tin[0][y].fW + Tin[0][y].fNW + Tin[0][y].fSW)) / (1 - AtmosVx[0][y]);
    Tin[0][y].fE = Tin[0][y].fW + 2.f / 3.f * Temp[0][y] * AtmosVx[0][y];
    Tin[0][y].fNE = Tin[0][y].fNW + Temp[0][y] * AtmosVx[0][y] / 6.f;
    Tin[0][y].fSE = Tin[0][y].fSW + Temp[0][y] * AtmosVx[0][y] / 6.f;
    return;
}

__global__ void BoundarySouthTemp()
{
    int x = threadIdx.y + blockIdx.y * blockDim.y;
    int y = 0;
    float TempConst = (1.0f * x / Nx) * 1.0f / 18.0f; // temperature from 0 to 1
    float alphaT = 0.f;

    Tin[x][0].fC = Tout[x][0].fC;
    Tin[x][0].fS = Tout[x][1].fS;
    if (x < Nx - 1) 
    {
       Tin[x][0].fW = Tout[x + 1][0].fW;
       Tin[x][0].fSW = Tout[x + 1][1].fSW; 
    }
    else { return; }

    if (x > 0) 
    { 
       Tin[x][0].fE = Tout[x - 1][0].fE;
       Tin[x][0].fSE = Tout[x - 1][1].fSE; 
    }
    else { return; }

    Tin[x][0].fN = (1.f - alphaT) * Tin[x][0].fS + alphaT * (4.0f * TempConst - Tin[x][0].fS);
    Tin[x][0].fNW = (1.f - alphaT) * Tin[x][0].fSE + alphaT * (TempConst - Tin[x][0].fSE);
    Tin[x][0].fNE = (1.f - alphaT) * Tin[x][0].fSW + alphaT * (TempConst - Tin[x][0].fSW);
    return;
}

__global__ void BoundaryNordTemp()
{
    int x = threadIdx.y + blockIdx.y * blockDim.y;
    int y = Ny - 1;
    float TempConst = 0.0f;
    float alphaT = 1.f;

    Tin[x][y].fC = Tout[x][y].fC;
    Tin[x][y].fN = Tout[x][y - 1].fN;
    if (x < Nx - 1)
    {
       Tin[x][y].fW = Tout[x + 1][y].fW;
       Tin[x][y].fNW = Tout[x + 1][y - 1].fNW;
    }
    else { return; }
    if (x > 0) 
    {
       Tin[x][y].fE = Tout[x - 1][y].fE;
       Tin[x][y].fNE = Tout[x - 1][y - 1].fNE; 
    }
    else { return; }

    Tin[x][y].fS = (1.f - alphaT) * Tin[x][y].fN + alphaT * (4.0f * TempConst - Tin[x][y].fN);
    Tin[x][y].fSW = (1.f - alphaT) * Tin[x][y].fNE + alphaT * (TempConst - Tin[x][y].fNE);
    Tin[x][y].fSE = (1.f - alphaT) * Tin[x][y].fNW + alphaT * (TempConst - Tin[x][y].fNW);
    // return
    Temp[x][y] = (Tin[x][y].fC + Tin[x][y].fE + Tin[x][y].fW + 2 * (Tin[x][y].fN + Tin[x][y].fNW + Tin[x][y].fNE)) / (1 + AtmosVy[x][y]);
    Tin[x][y].fS = Tin[x][y].fN - 2.f / 3.f * Temp[x][y] * AtmosVy[x][y];
    Tin[x][y].fSE = Tin[x][y].fNE - Temp[x][y] * AtmosVy[x][y] / 6.f;
    Tin[x][y].fSW = Tin[x][y].fNW - Temp[x][y] * AtmosVy[x][y] / 6.f;
    return;
}


#endif