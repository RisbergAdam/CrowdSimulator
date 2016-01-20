///////////////////////////////////////////////////
// Low Level Parallel Programming 2016.
//
//     ==== Don't change this file! ====
// 
#include "PedSimulation.h"
#include <iostream>
#include <QApplication>

using namespace std;

PedSimulation::PedSimulation(Ped::Model &model_,MainWindow &window_) : model(model_), window(window_), maxSimulationSteps(-1)
{}

void PedSimulation::simulateOneStep()
{
  model.tick();
  window.paint();
  if(maxSimulationSteps-- == 0)
    {
      QApplication::quit();
    }
}

void PedSimulation::runSimulationWithQt(int maxNumberOfStepsToSimulate)
{
  maxSimulationSteps = maxNumberOfStepsToSimulate;
  auto movetimer = new QTimer(); 
  QObject::connect(movetimer, SIGNAL(timeout()), this, SLOT(simulateOneStep()));
  movetimer->start();
}

void PedSimulation::runSimulationWithoutQt(int maxNumberOfStepsToSimulate)
{
  maxSimulationSteps = maxNumberOfStepsToSimulate;
  for(int i = 0; i < maxSimulationSteps; i++)
  {
    model.tick();
  }
}
