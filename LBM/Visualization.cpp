#include<Gl/glut.h>
#include<math.h>
#include<stdlib.h>
#include "Visualization.h"

void Initialize()
{
   glClearColor(0.0, 0.0, 0.0, 1.0);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(-10, Nx + 10, -10, Ny + 10, -200.0, 200.0);
   glMatrixMode(GL_MODELVIEW);
}

void Timer(int value)
{
   glColor3f(1.0, 1.0, 1.0);
   glutPostRedisplay();
   glutTimerFunc(50, Timer, 0);
}
