///////////////////////////////////////////////////
// Low Level Parallel Programming 2016.
//
//     ==== Don't change this file! ====
//
// The main starting point for the crowd simulation.
//

#include "ped_model.h"
#include "MainWindow.h"
#include "ParseScenario.h"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QApplication>
#include <QTimer>
#include <thread>

#include <unistd.h>

#include "PedSimulation.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <cstring>

int main(int argc, char*argv[]) { 
  bool timing_mode = 0;
  int i = 1;
  QString scenefile = "scenario.xml";

  // Argument handling
  while(i < argc)
  {
    if(argv[i][0] == '-' && argv[i][1] == '-')
    {
      if(strcmp(&argv[i][2],"timing-mode") == 0)
      {
	cout << "Timing mode on\n";
	timing_mode = true;
      }
      else if(strcmp(&argv[i][2],"help") == 0)
      {
	cout << "Usage: " << argv[0] << " [--help] [--timing-mode] [scenario]" << endl;
	return 0;
      }
      else
      {
	cerr << "Unrecognized command: \"" << argv[i] << "\". Ignoring ..." << endl;
      }
    }
    else // Assume it is a path to scenefile
    {
      scenefile = argv[i];
    }
    
    i+=1;
  }

  // Reading the scenario file and setting up the crowd simulation model
  Ped::Model model;
  ParseScenario parser(scenefile);
  model.setup(parser.getAgents());

  // GUI related set ups
  QApplication app(argc, argv);
  MainWindow mainwindow(model);

  // Default number of steps to simulate
  const int maxNumberOfStepsToSimulate = 10000;
  PedSimulation *simulation = new PedSimulation(model, mainwindow);

  cout << "Demo setup complete, running ..." << endl;
  int retval = 0;
  // Timing of simulation
  std::chrono::time_point<std::chrono::system_clock> start,stop;
  start = std::chrono::system_clock::now();

  if(timing_mode)
  {
    // Simulation mode to use when profiling (without any GUI)
    simulation->runSimulationWithoutQt(maxNumberOfStepsToSimulate);
  }
  else
  {
    // Simulation mode to use when visualizing
    mainwindow.show();
    simulation->runSimulationWithQt(maxNumberOfStepsToSimulate);
    retval = app.exec();
  }

  // End timing
  stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = stop-start;
  cout << "Time: " << elapsed_seconds.count() << " seconds." << endl;

  cout << "Done" << endl;
  delete (simulation);
  return retval;
}
