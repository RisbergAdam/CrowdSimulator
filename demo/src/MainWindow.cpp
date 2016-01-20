///////////////////////////////////////////////////
// Low Level Parallel Programming 2016.
//
//     ==== Don't change this file! ====
// 
#include "MainWindow.h"

#include <QtGui>
#include <QBrush>
#include <QImage>
#include <iostream>
#include <algorithm>

MainWindow::MainWindow(const Ped::Model &model) : model(model)
{
  // The Window 
  graphicsView = new QGraphicsView();

  setCentralWidget(graphicsView);

  // A surface for managing a large number of 2D graphical items
  scene = new QGraphicsScene(QRect(0,0,800,600),this);
  
  // Connect
  graphicsView->setScene(scene);

  for (int x=0; x<=800; x+=cellsizePixel)
  {
    scene->addLine(x,0,x,800, QPen(Qt::gray));
  }

 // Now add the horizontal lines, paint them green
  for (int y=0; y<=800; y+=cellsizePixel)
  {
    scene->addLine(0,y,800,y, QPen(Qt::gray));
  }

  // Create viewAgents with references to the position of the model counterparts
  auto &agents = model.getAgents();
  for(auto agent : agents)
  {
    viewAgents.push_back(new ViewAgent(agent, scene));
  }

#ifdef ASSIGNMENT_4
  const int heatmapSize = model.getHeatmapSize();
  QPixmap pixmapDummy = QPixmap(heatmapSize, heatmapSize);
  pixmap = scene->addPixmap(pixmapDummy);
#endif
  
  paint();
  graphicsView->show(); // Redundant? 
}

void MainWindow::paint() {
#ifdef ASSIGNMENT_4
  const int heatmapSize = model.getHeatmapSize();
  QImage image((uchar*) *model.getHeatmap(), heatmapSize, heatmapSize, heatmapSize * sizeof(int), QImage::Format_ARGB32);
  pixmap->setPixmap(QPixmap::fromImage(image));
#endif

  for(auto a : viewAgents)
  {
    a->paint();
  }
}

int MainWindow::cellToPixel(int val)
{
  return val*cellsizePixel;
}
