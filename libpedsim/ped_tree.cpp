//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2015
//
#include "ped_tree.h"

#include <cassert>
#include <cstddef>
#include <algorithm>

using namespace std;

/// Description: set intial values
/// \author  chgloor
/// \date    2012-01-28
Ped::Ttree::Ttree(Ped::Ttree *root,std::map<const Ped::Tagent*, Ped::Ttree*> *treehash, int pdepth, int pmaxDepth, double px, double py, double pw, double ph) {
  // more initializations here. not really necessary to put them into the initializator list, too.
  this->root = root != NULL ? root: this;
  this->treehash = treehash;
  isleaf = true;
  x = px;
  y = py;
  w = pw;
  h = ph;
  depth = pdepth;
  maxDepth = pmaxDepth;

  // Tree contains of 4 subtrees
  tree1 = NULL;
  tree2 = NULL;
  tree3 = NULL;
  tree4 = NULL;
};

/// Destructor. Deleted this node and all its children. If there are any agents left, they are removed first (not deleted).
/// \author  chgloor
/// \date    2012-01-28
Ped::Ttree::~Ttree() {
  clear();
}

void Ped::Ttree::clear() {
  if(isleaf) {
    agents.clear();
  }
  else {
    tree1->clear();
    delete tree1;
    tree2->clear();
    delete tree2;
    tree3->clear();
    delete tree3;
    tree4->clear();
    delete tree4;
    isleaf = true;
  }
}

bool cmp(const Ped::Tagent *a, const Ped::Tagent *b) {
  return a->getX() == b->getX() && a->getY() == b->getY();
}

/// \author  chgloor
/// \date    2012-01-28
/// \param   *a The agent to add
void Ped::Ttree::addAgent(const Ped::Tagent *a) {
  if (isleaf) {
    agents.insert(a);
    (*treehash)[a] = this;
  }
  else {
    if ((a->getX() >= x+w/2) && (a->getY() >= y+h/2)) tree3->addAgent(a); // 3
    if ((a->getX() <= x+w/2) && (a->getY() <= y+h/2)) tree1->addAgent(a); // 1
    if ((a->getX() >= x+w/2) && (a->getY() <= y+h/2)) tree2->addAgent(a); // 2
    if ((a->getX() <= x+w/2) && (a->getY() >= y+h/2)) tree4->addAgent(a); // 4
  }

  if (agents.size() > 8 && depth < maxDepth) {
    isleaf = false;
    addChildren();
    while (!agents.empty()) {
      const Ped::Tagent *a = (*agents.begin());
      if ((a->getX() >= x+w/2) && (a->getY() >= y+h/2)) tree3->addAgent(a); // 3
      if ((a->getX() <= x+w/2) && (a->getY() <= y+h/2)) tree1->addAgent(a); // 1
      if ((a->getX() >= x+w/2) && (a->getY() <= y+h/2)) tree2->addAgent(a); // 2
      if ((a->getX() <= x+w/2) && (a->getY() >= y+h/2)) tree4->addAgent(a); // 4
      agents.erase(a);
    }
  }
}

void Ped::Ttree::addChildren() {
  tree1 = new Ped::Ttree(root,treehash, depth+1, maxDepth,  x, y, w/2, h/2);
  tree2 = new Ped::Ttree(root,treehash, depth+1, maxDepth,  x+w/2, y, w/2, h/2);
  tree3 = new Ped::Ttree(root,treehash, depth+1, maxDepth,  x+w/2, y+h/2, w/2, h/2);
  tree4 = new Ped::Ttree(root,treehash, depth+1, maxDepth,  x, y+h/2, w/2, h/2);
}

Ped::Ttree* Ped::Ttree::getChildByPosition(double xIn, double yIn) {
  if((xIn <= x+w/2) && (yIn <= y+h/2))
    return tree1;
  if((xIn >= x+w/2) && (yIn <= y+h/2))
    return tree2;
  if((xIn >= x+w/2) && (yIn >= y+h/2))
    return tree3;
  if((xIn <= x+w/2) && (yIn >= y+h/2))
    return tree4;

  // this should never happen
  return NULL;
}

void Ped::Ttree::moveAgent(const Ped::Tagent *a) {
  if ((a->getX() < x) || (a->getX() > (x+w)) || (a->getY() < y) || (a->getY() > (y+h))) {
    agents.erase(a);
    root->addAgent(a);
  }
}

bool Ped::Ttree::removeAgent(const Ped::Tagent *a) {
  if(isleaf) {
    size_t removedCount = agents.erase(a);
    return (removedCount > 0);
  }
  else {
    return getChildByPosition(a->getX(), a->getY())->removeAgent(a);
  }
}

int Ped::Ttree::cut() {
  if (isleaf)
    return agents.size();

  int count = 0;
  count += tree1->cut();
  count += tree2->cut();
  count += tree3->cut();
  count += tree4->cut();
  if (count < 5) {
    assert(tree1->isleaf == true);
    assert(tree2->isleaf == true);
    assert(tree3->isleaf == true);
    assert(tree4->isleaf == true);
    agents.insert(tree1->agents.begin(), tree1->agents.end());
    agents.insert(tree2->agents.begin(), tree2->agents.end());
    agents.insert(tree3->agents.begin(), tree3->agents.end());
    agents.insert(tree4->agents.begin(), tree4->agents.end());
    isleaf = true;
    for (set<const Ped::Tagent*>::iterator it = agents.begin(); it != agents.end(); ++it) {
      const Tagent *a = (*it);
      (*treehash)[a] = this;
    }
    delete tree1;
    delete tree2;
    delete tree3;
    delete tree4;
  }
  return count;
}

set<const Ped::Tagent*> Ped::Ttree::getAgents() const {
  if (isleaf)
    return agents;

  set<const Ped::Tagent*> ta;
  set<const Ped::Tagent*> t1 = tree1->getAgents();
  set<const Ped::Tagent*> t2 = tree2->getAgents();
  set<const Ped::Tagent*> t3 = tree3->getAgents();
  set<const Ped::Tagent*> t4 = tree4->getAgents();
  ta.insert(t1.begin(), t1.end());
  ta.insert(t2.begin(), t2.end());
  ta.insert(t3.begin(), t3.end());
  ta.insert(t4.begin(), t4.end());
  return ta;
}

void Ped::Ttree::getAgents(list<const Ped::Tagent*>& outputList) const {
  if(isleaf) {
    for (set<const Ped::Tagent*>::iterator it = agents.begin(); it != agents.end(); ++it) {
      const Ped::Tagent* currentAgent = (*it);
      outputList.push_back(currentAgent);
    }
  }
  else {
    tree1->getAgents(outputList);
    tree2->getAgents(outputList);
    tree3->getAgents(outputList);
    tree4->getAgents(outputList);
  }
}

bool Ped::Ttree::intersects(double px, double py, double pr) const {
  if (((px+pr) >= x) && ((px-pr) <= (x+w)) && ((py+pr) >= y) && ((py-pr) <= (y+h)))
    return true; // x+-r/y+-r is inside
  else
    return false;
}
