#include <Gl/glut.h>
#include <math.h>
#include <stdlib.h>
#include "Visualization.h"
#include "LBM0.h"

void KeyboardEsc(unsigned char key, int x, int y);

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
   // Initialization
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
   glutInitWindowSize(768, 768);
   glutInitWindowPosition(0, 20);
   glutTimerFunc(10, TimeEvent, 1);
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
   if (mainLBM(FirstCycle) == 1) LastCycle = true;
   DrawAtmosVer();
   FirstCycle = false;
}

void DrawLBM2() ///////////////////////////////////////////////////////
{
   DrawAtmosHor();
   FirstCycle = false;
   if (LastCycle) { exit(0); }
}

