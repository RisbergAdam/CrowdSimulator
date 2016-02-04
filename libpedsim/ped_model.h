//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2016
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_
#include <vector>
#include <map>

#include <pthread.h>
#include <semaphore.h>
#include <omp.h>

#include "ped_tree.h"
#include "ped_agent.h"


namespace Ped{
  class Tagent;
  class Ttree;

  // The implementation modes for Assignment 1 + 2:
  // chooses which implementation to use for tick()
  enum IMPLEMENTATION {CUDA, VECTOR, OMP, PTHREAD, SEQ};

  struct ThreadArg {
    int agentFrom, agentTo;
    int isRunning;
    std::vector<Tagent*> agents;
    sem_t waitForModel;
    sem_t waitForThread;
  };
  
  class Model
  {
  public:

    // Sets everything up
    void setup(std::vector<Tagent*> agentsInScenario,int imp);

    // Coordinates a time step in the scenario: move all agents by one step (if applicable).
    void tick();

    // Returns the agents of this scenario
    const std::vector<Tagent*> getAgents() const { return agents; };

    // Updates the treehash, which maps each agent to the current tree node that contains it
    void setResponsibleTree(Ped::Ttree *tree, const Ped::Tagent *agent);
   
    // Adds an agent to the tree structure
    void placeAgent(const Ped::Tagent *a);

    // Cleans up the tree and restructures it. Worth calling every now and then.
    void cleanup();
    ~Model();

    // Returnst the heatmap visualizing the density of agents
    int const * const * getHeatmap() const { return blurred_heatmap; };
    int getHeatmapSize() const;
    
  private:

    // Denotes which implementation (sequential, parallel implementations..)
    // should be used for calculating the desired positions of
    // agents (Assignment 1)
    IMPLEMENTATION implementation;

    // The agents in this scenario
    std::vector<Tagent*> agents;

    // Moves an agent towards its next position
    void move(Ped::Tagent *agent);

    // PTHREAD
    static const int threadCount = 6;
    Ped::ThreadArg * threadArgs[threadCount];
    pthread_t threads[threadCount];

    // VECTOR
    float * xVector, * yVector, * xDest, * yDest;
    
    
    ////////////
    /// Everything below here won't be relevant until Assignment 3
    ///////////////////////////////////////////////

    // The maximum quadtree depth
    static const int treeDepth = 10;    

    // Quadtree that keeps track of the positions of each agent
    // for faster neighbor search in 
    Ped::Ttree *tree;

    // Maps the agent to the tree node containing it. Convenience data structure
    // in order to update the tree in case the agent moves.
    std::map<const Ped::Tagent*, Ped::Ttree*> *treehash;

    // Returns the set of neighboring agents for the specified position
    set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;
    void getNeighbors(list<const Ped::Tagent*>& neighborList, int x, int y, int d) const;
    
    ////////////
    /// Everything below here won't be relevant until Assignment 4
    ///////////////////////////////////////////////
    #define SIZE 1024
    #define CELLSIZE 5
    #define SCALED_SIZE SIZE*CELLSIZE

    // The heatmap representing the density of agents
    int ** heatmap;

    // The scaled heatmap that fits to the view
    int ** scaled_heatmap;

    // The final heatmap: blurred and scaled to fit the view
    int ** blurred_heatmap;
    
    void setupHeatmapSeq();
    void updateHeatmapSeq();
  };
}
#endif
