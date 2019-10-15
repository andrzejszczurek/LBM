#ifndef LBM0_h
#define LBM0_h
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>


// warunki brzegowe: 1 - bounce-back, 2 - symmetry, 3 - velocity, 4 - rho
#define BoundaryCEast 4
#define BoundaryCWest 3
#define BoundaryCNord 3
#define BoundaryCSouth 1

// czy uwzględniać grawitację
#define UseGravityMode true

// Czy rysować linie toku
#define DrawLines true;

// Grawitacja
#define grav 0.000025f

// Powiększenie (na potrzeby zapisu do pliku)
#define Npic 3	

// Poziomy rozmiar przestrzeni
#define Nx 256

// Pionowy rozmiar przestrzeni
#define Ny 256

// Stała czasowa relaksacji
#define tauAtmos 1.0f

// Całkowity czas
extern __device__ __managed__ float timem;

// Krok czasu
extern __device__ __managed__ float stept;

// Funkcja rozkładu D2Q9
struct Dist 
{ 
    float fC;
    float fE;
    float fW; 
    float fS; 
    float fN;
    float fSE;
    float fSW;
    float fNE;
    float fNW;
};

// Tablica gęstości
extern __device__ __managed__  float AtmosRho[Nx][Ny];

// Tablica poziomych prędkości
extern __device__ __managed__  float AtmosVx[Nx][Ny];

// Tablica pionowych prędkości
extern __device__ __managed__  float AtmosVy[Nx][Ny];

// input, output and equilibrium distribution functions
extern __device__ __managed__ Dist Atmosin[Nx][Ny], Atmosout[Nx][Ny], Atmoseq[Nx][Ny]; 

__global__ void InitialAtmos(bool NewSim);
__global__ void StreamingAtmos();
__global__ void EquiRelaxAtmos();
int mainLBM(bool FirstCycle);

void OutRes();

void InRes();

#endif
