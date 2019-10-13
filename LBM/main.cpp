#include <Gl/glut.h>
#include <math.h>
#include <stdlib.h>
#include "Visualization.h"
#include "LBM0.h"

void KeyboardEsc(unsigned char key, int x, int y);
void Timer(int value);
void Initialize();
void DrawLBM1();
void DrawLBM2();

int WinId, win1, win2;
bool FirstCycle = true;
bool LastCycle = false;
int icycle = 0;

static void TimeEvent(int te) /////////////////////////////////////////////////
{
   glutSetWindow(win1);
   glutPostRedisplay();

   glutSetWindow(win2);
   glutPostRedisplay();

   glutTimerFunc(10, TimeEvent, 1);  // Reset our timer.
}


int main(int argc, char** argv) ///////////////////////////////////////////////
{
   int a;
   std::cin >> a;

   // Initialization
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
   glutInitWindowSize(768, 768);
   glutInitWindowPosition(0, 20);
   glutTimerFunc(10, TimeEvent, 1);
   glutTimerFunc(50, Timer, 0);
   win1 = glutCreateWindow("LBM-CUDA [Vy]");
   // Registration
   glutDisplayFunc(DrawLBM1);
   Initialize();
   glutInitWindowPosition(768, 20);
   win2 = glutCreateWindow("LBM-CUDA [Vx]");
   glutDisplayFunc(DrawLBM2);
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

void Timer(int value)
{
    glColor3f(1.0, 1.0, 1.0);
    glutPostRedisplay();
    glutTimerFunc(50, Timer, 0);
}

void KeyboardEsc(unsigned char key, int x, int y) //////////////////////
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

void DrawLBM1() /////////////////////////////////////////////////////////
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

void DrawLBM2() ///////////////////////////////////////////////////////
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

