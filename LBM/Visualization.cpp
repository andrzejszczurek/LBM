#include<Gl/glut.h>
#include<math.h>
#include<stdlib.h>
#include <algorithm>
#include "Visualization.h"
using namespace std;

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

void BMPout(char rgbp[Npic * Ny][Npic * Nx][3], char* bmpfile)
{
   const int width = Npic * Nx;
   const int height = Npic * Ny;
   using namespace std;
   FILE* pFile = fopen(bmpfile, "wb"); // wb -> w: writable b: binary, open as writable and binary
   if (pFile == NULL) { return; }

   BITMAPINFOHEADER BMIH;                         // BMP header
   BMIH.biSizeImage = width * height * 3;
   // Create the bitmap for this OpenGL context
   BMIH.biSize = sizeof(BITMAPINFOHEADER);
   BMIH.biWidth = width;
   BMIH.biHeight = height;
   BMIH.biPlanes = 1;
   BMIH.biBitCount = 24;
   BMIH.biCompression = BI_RGB;
   BMIH.biSizeImage = width * height * 3;

   BITMAPFILEHEADER bmfh;                         // Other BMP header
   int nBitsOffset = sizeof(BITMAPFILEHEADER) + BMIH.biSize;
   LONG lImageSize = BMIH.biSizeImage;
   LONG lFileSize = nBitsOffset + lImageSize;
   bmfh.bfType = 'B' + ('M' << 8);
   bmfh.bfOffBits = nBitsOffset;
   bmfh.bfSize = lFileSize;
   bmfh.bfReserved1 = bmfh.bfReserved2 = 0;

   // Write the bitmap file header               // Saving the first header to file
   UINT nWrittenFileHeaderSize = fwrite(&bmfh, 1, sizeof(BITMAPFILEHEADER), pFile);

   // And then the bitmap info header            // Saving the second header to file
   UINT nWrittenInfoHeaderSize = fwrite(&BMIH, 1, sizeof(BITMAPINFOHEADER), pFile);

   // Finally, write the image data itself
   //-- the data represents our drawing          // Saving the file content in lpBits to file
   UINT nWrittenDIBDataSize = fwrite(rgbp, 1, lImageSize, pFile);
   fclose(pFile); // closing the file.
}


void DrawLine(int x0, int y0, int x1, int y1, char rgbp[Npic * Ny][Npic * Nx][3])
{
   x0 = max(0, min(Npic * Nx - 1, x0));
   x1 = max(0, min(Npic * Nx - 1, x1));
   y0 = max(0, min(Npic * Ny - 1, y0));
   y1 = max(0, min(Npic * Ny - 1, y1));
   int dx = x1 - x0, dy = y1 - y0;
   if (abs(dy) > abs(dx))
   {
      if (y1 > y0)
      {
         for (int y = y0; y <= y1; y++)
         {
            int x = x0 + 1.0 * (x1 - x0) / (y1 - y0) * (y - y0);
            rgbp[y][x][0] = 0; rgbp[y][x][1] = 0; rgbp[y][x][2] = 0;
         }
      }
      else
      {
         for (int y = y1; y <= y0; y++)
         {
            int x = x1 + 1.0 * (x1 - x0) / (y1 - y0) * (y - y1);
            rgbp[y][x][0] = 0; rgbp[y][x][1] = 0; rgbp[y][x][2] = 0;
         }
      }
   }
   else
   {
      if (x1 > x0)
      {
         for (int x = x0; x <= x1; x++)
         {
            int y = y0 + 1.0 * (y1 - y0) / (x1 - x0) * (x - x0);
            rgbp[y][x][0] = 0; rgbp[y][x][1] = 0; rgbp[y][x][2] = 0;
         }
      }
      else
      {
         if (x0 == x1) return;
         for (int x = x1; x <= x0; x++)
         {
            int y = y1 + 1.0 * (y1 - y0) / (x1 - x0) * (x - x1);
            rgbp[y][x][0] = 0; rgbp[y][x][1] = 0; rgbp[y][x][2] = 0;
         }
      }
   }
}
