//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2015
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "cuda_dummy.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include <xmmintrin.h>

// forward declaration
void * StartThread(void *);

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, int imp)
{
 
  agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());
  treehash = new std::map<const Ped::Tagent*, Ped::Ttree*>();

  // Create a new quadtree containing all agents
  tree = new Ttree(NULL, treehash, 0, treeDepth, 0, 0, 1000, 800);
  for (std::vector<Ped::Tagent*>::iterator it = agents.begin(); it != agents.end(); ++it)
    {
      tree->addAgent(*it);
    }
  
  if (imp == 1){
    implementation = SEQ;
  } else if (imp = 2){
    implementation = OMP;
  } else if (imp = 3){
    implementation = PTHREAD;
  } else if (imp = 4) {
    cout << "vector waddup\n" << imp;
    implementation = VECTOR;
  }

  implementation = VECTOR;

  // Set up heatmap (relevant for Assignment 4)
  setupHeatmapSeq();

  if (implementation == PTHREAD) {
    // setup threads and their arguments
    for (int i = 0; i < threadCount; i++) {
      threadArgs[i] = new Ped::ThreadArg;
      Ped::ThreadArg * arg = threadArgs[i];

      arg->isRunning = 1;
      sem_init(&(arg->waitForModel), 0, 0);
      sem_init(&(arg->waitForThread), 0, 0);

      pthread_create(&threads[i], NULL, StartThread, (void *) arg);
    }
  } else if (implementation == VECTOR) {
    cout << "vector2\n";
    xVector = new float[agents.size()];
    yVector = new float[agents.size()];
    xDest = new float[agents.size()];
    yDest = new float[agents.size()];

    for (int i = 0; i < agents.size(); i++) {
      Ped::Tagent * agent = agents[i];
      xVector[i] = (float) agent->getX();
      yVector[i] = (float) agent->getY();
    }
  }

}

void Ped::Model::tick()
{

  if (implementation == SEQ) {
    // get agent list
    const std::vector<Ped::Tagent*> agents = getAgents();
    // iterate over agent list and update each position
    for (int i = 0; i < agents.size(); i++) {
      Ped::Tagent * agent = agents[i];
      agent->computeNextDesiredPosition();
      agent->setX(agent->getDesiredX());
      agent->setY(agent->getDesiredY());
    }
  } else if (implementation == PTHREAD) {
    const std::vector<Ped::Tagent*> agents = getAgents();
    int totalAgentCount = agents.size();
	  
    for (int i = 0; i < threadCount; i++) {
      int agentCount = totalAgentCount / threadCount;
	    
      threadArgs[i]->agentFrom = i * agentCount;
      threadArgs[i]->agentTo = (i + 1) * agentCount;
      threadArgs[i]->agents = agents;
	    
      if (i == threadCount - 1) {
	threadArgs[i]->agentTo = totalAgentCount;
      }
	    
      sem_post(&(threadArgs[i]->waitForModel));
    }

    for (int i = 0; i < threadCount; i++) {
      sem_wait(&(threadArgs[i]->waitForThread));
    }
	  
  } else if (implementation == OMP) {
    const std::vector<Ped::Tagent*> agents = getAgents();
    int i;
    omp_set_num_threads(6);
    #pragma omp parallel for private(i)
    for (int i = 0; i < agents.size(); i++) {
      Ped::Tagent * agent = agents[i];
      agent->computeNextDesiredPosition();
      agent->setX(agent->getDesiredX());
      agent->setY(agent->getDesiredY());
    }
  } else if (implementation == VECTOR) {
    __m128 xReg, xDestReg, xDiffReg;
    __m128 yReg, yDestReg, yDiffReg;
    __m128 xSquaredReg, ySquaredReg, squaredSumReg;
    __m128 len, lenAdd;

    for (int i = 0;i < agents.size();i += 4) {
      for (int j = 0;j < 4;j++) {
	Ped::Tagent * agent = agents[i + j];
	Twaypoint* dest = agent->getNextDestNotNull();
	//xDest[i + j] = (float) dest->getx();
	//yDest[i + j] = (float) dest->gety();
	xDest[i + j] = (float) agent->getX();
	yDest[i + j] = (float) agent->getY();
      }

      float z[4] = {0,0,0,0};
      __m128 zero = _mm_load_ps(z);
      
      xReg = _mm_load_ps(&xVector[i]);
      xDestReg = _mm_load_ps(&xDest[i]);
      xDiffReg = _mm_sub_ps(xDestReg, xReg);

      yReg = _mm_load_ps(&yVector[i]);
      yDestReg = _mm_load_ps(&yDest[i]);
      yDiffReg = _mm_sub_ps(yDestReg, yReg);

      xSquaredReg = _mm_mul_ps(xDiffReg, xDiffReg);
      ySquaredReg = _mm_mul_ps(yDiffReg, yDiffReg);

      squaredSumReg = _mm_add_ps(xSquaredReg, ySquaredReg);

      len = _mm_sqrt_ps(squaredSumReg);
      lenAdd = _mm_cmpeq_ps(len, zero);
      //len = _mm_add_ps(len, lenAdd);

      xDiffReg = _mm_div_ps(xDiffReg, len);
      yDiffReg = _mm_div_ps(yDiffReg, len);

      _mm_store_ps(z, len);
      std::cout << z[0] << "\n";
      
      xReg = _mm_add_ps(xDiffReg, xReg);
      yReg = _mm_add_ps(yDiffReg, yReg);

      //xReg = _mm_round_ps(xReg, 0);
      //yReg = _mm_round_ps(yReg, 0);

      //_mm_store_ps(&xVector[i], xReg);
      //_mm_store_ps(&yVector[i], yReg);

      for (int j = 0;j < 4;j++) {
	Ped::Tagent * agent = agents[i + j];
	agent->setX((int) (xVector[i + j] + 0.5));
	agent->setY((int) (yVector[i + j] + 0.5));
      }
    }
    
  }
}

void * StartThread(void * arg) {
  Ped::ThreadArg * tArg = (Ped::ThreadArg *) arg;
  while (tArg->isRunning) {
    sem_wait(&(tArg->waitForModel));
    
    //std::cout << "from: " << tArg->agentFrom << " to: " << tArg->agentTo << " " << &(tArg->waitForThread) << "\n";

    const std::vector<Ped::Tagent*> agents = tArg->agents;
    // iterate over agent list and update each position
    for (int i = tArg->agentFrom; i < tArg->agentTo; i++) {
      Ped::Tagent * agent = agents[i];
      agent->computeNextDesiredPosition();
      agent->setX(agent->getDesiredX());
      agent->setY(agent->getDesiredY());
    }
    
    sem_post(&(tArg->waitForThread));
  }
  
  return NULL;
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////
void Ped::Model::move( Ped::Tagent *agent)
{
  // Search for neighboring agents
  set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

  // Retrieve their positions
  std::vector<std::pair<int, int> > takenPositions;
  for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
    std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
    takenPositions.push_back(position);
  }

  // Compute the three alternative positions that would bring the agent
  // closer to his desiredPosition, starting with the desiredPosition itself
  std::vector<std::pair<int, int> > prioritizedAlternatives;
  std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
  prioritizedAlternatives.push_back(pDesired);

  int diffX = pDesired.first - agent->getX();
  int diffY = pDesired.second - agent->getY();
  std::pair<int, int> p1, p2;
  if (diffX == 0 || diffY == 0)
    {
      // Agent wants to walk straight to North, South, West or East
      p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
      p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
    }
  else {
    // Agent wants to walk diagonally
    p1 = std::make_pair(pDesired.first, agent->getY());
    p2 = std::make_pair(agent->getX(), pDesired.second);
  }
  prioritizedAlternatives.push_back(p1);
  prioritizedAlternatives.push_back(p2);

  // Find the first empty alternative position
  for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

    // If the current position is not yet taken by any neighbor
    if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

      // Set the agent's position
      agent->setX((*it).first);
      agent->setY((*it).second);

      // Update the quadtree
      (*treehash)[agent]->moveAgent(agent);
      break;
    }
  }
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {
  // if there is no tree, return all agents
  if (tree == NULL)
    return set<const Ped::Tagent*>(agents.begin(), agents.end());

  // create the output list
  list<const Ped::Tagent*> neighborList;
  getNeighbors(neighborList, x, y, dist);

  // copy the neighbors to a set
  return set<const Ped::Tagent*>(neighborList.begin(), neighborList.end());
}

/// Populates the list of neighbors that can be found around x/y./// This triggers a cleanup of the tree structure. Unused leaf nodes are collected in order to
/// save memory. Ideally cleanup() is called every second, or about every 20 timestep.
/// \date    2012-01-28
void Ped::Model::cleanup() {
  if (tree != NULL)
    tree->cut();
}

/// \date    2012-01-29
/// \param   the list to populate
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
void Ped::Model::getNeighbors(list<const Ped::Tagent*>& neighborList, int x, int y, int dist) const {
  stack<Ped::Ttree*> treestack;

  treestack.push(tree);
  while (!treestack.empty()) {
    Ped::Ttree *t = treestack.top();
    treestack.pop();
    if (t->isleaf) {
      t->getAgents(neighborList);
    }
    else {
      if (t->tree1->intersects(x, y, dist)) treestack.push(t->tree1);
      if (t->tree2->intersects(x, y, dist)) treestack.push(t->tree2);
      if (t->tree3->intersects(x, y, dist)) treestack.push(t->tree3);
      if (t->tree4->intersects(x, y, dist)) treestack.push(t->tree4);
    }
  }
}

Ped::Model::~Model()
{
  if (tree != NULL)
    {
      delete tree;
      tree = NULL;
    }
  if (treehash != NULL)
    {
      delete treehash;
      treehash = NULL;
    }
}
