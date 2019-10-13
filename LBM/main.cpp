#include <Gl/glut.h>
#include <math.h>
#include <stdlib.h>
#include "Visualization.h"
#include "LBM0.h"

void KeyboardEsc(unsigned char key, int x, int y);
void Timer(int value);
void Initialize();
void DrawLBM_Vy();
void DrawLBM_Vx();
void DrawLBM_Temp();
void TimeEvent(int te);

int WinId;
int win_vy;
int win_vx;
int win_temp;
bool FirstCycle = true;
bool LastCycle = false;
int icycle = 0;


int main(int argc, char** argv)
{
   // ogólna konfiguracja okien
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
   glutInitWindowSize(768, 768);
   glutTimerFunc(10, TimeEvent, 1);
   //glutTimerFunc(50, Timer, 0);

   // okno dla składowej y prędkości
   glutInitWindowPosition(0, 20);
   win_vy = glutCreateWindow("LBM-CUDA [Vy]");
   glutDisplayFunc(DrawLBM_Vy);
   Initialize();

   // okno dla składowej x prędkości
   glutInitWindowPosition(768, 20);
   win_vx = glutCreateWindow("LBM-CUDA [Vx]");
   glutDisplayFunc(DrawLBM_Vx);
   Initialize();

   // okno dla rozkładu temperatury
   glutInitWindowPosition(1536, 20);
   win_temp = glutCreateWindow("LBM-CUDA [Temp]");
   glutDisplayFunc(DrawLBM_Temp);
   Initialize();

   glutKeyboardFunc(KeyboardEsc);
   glutMainLoop();
   return 0;
}


void Initialize()
{
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-10, Nx + 10, -10, Ny + 10, -200.0, 200.0);
    glMatrixMode(GL_MODELVIEW);
    //glutTimerFunc(50, Timer, 0);
}

static void TimeEvent(int te)
{
    glutSetWindow(win_vy);
    glutPostRedisplay();

    glutSetWindow(win_vx);
    glutPostRedisplay();

    glutSetWindow(win_temp);
    glutPostRedisplay();

    glutTimerFunc(10, TimeEvent, 1);  // Reset our timer.
}

void Timer(int value)
{
    glColor3f(1.0, 1.0, 1.0);
    glutPostRedisplay();
    glutTimerFunc(50, Timer, 0);
}

void KeyboardEsc(unsigned char key, int x, int y)
{
   switch (key)
   {
   case 27: //Escape key
      glutDestroyWindow(WinId);
      exit(0);
      break;
   }
   glutPostRedisplay();
}

void DrawLBM_Vy()
{
   auto rgb = new char[Npic * Ny][Npic * Nx][3];
   if (mainLBM(FirstCycle) == 1) LastCycle = true;
   DrawAtmosVer(rgb);
   const auto bmpfilelen = 11;
   char bmpfile[bmpfilelen];
   snprintf(bmpfile, bmpfilelen, "Uy%04d.bmp", icycle);
   BMPout(rgb, bmpfile);
   delete[] rgb;
   FirstCycle = false;
}

void DrawLBM_Temp()
{
    DrawTempG(NULL);
}

void DrawLBM_Vx()
{
   auto rgb = new char[Npic * Ny][Npic * Nx][3];
   if (mainLBM(FirstCycle) == 1) LastCycle = true;
   DrawAtmosHor(rgb);
   const auto bmpfilelen = 11;
   char bmpfile[bmpfilelen];
   snprintf(bmpfile, bmpfilelen, "Ux%04d.bmp", icycle);
   BMPout(rgb, bmpfile);
   delete[] rgb;
   FirstCycle = false;

   if (LastCycle) 
      exit(0);
}

