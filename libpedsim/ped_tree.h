//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// A quadtree structure. A Quadtree can either be a node that consist
// of 4 child trees or a leaf that contains a set of agents.
//
#ifndef _ped_tree_h_
#define _ped_tree_h_ 1

#ifdef WIN32
#define LIBEXPORT __declspec(dllexport)
#else
#define LIBEXPORT
#endif

#include <set>
#include <list>
#include "ped_model.h"
#include "ped_agent.h"

using namespace std;

namespace Ped {
    class Tagent;
    class Model;

    class LIBEXPORT Ttree {
      friend class Ped::Model;

    public:
      Ttree(Ped::Ttree *root,std::map<const Ped::Tagent*, Ped::Ttree*> *treehash, int depth, int maxDepth, double x, double y, double w, double h);
        virtual ~Ttree();

	// Removes all agents from this tree
        virtual void clear();

        // Adds an agent to the tree. Searches the right node and adds the agent there.
        // If there are too many agents at that node allready, a new child is created.
        virtual void addAgent(const Ped::Tagent *a);

        // Updates the tree structure if an agent moves. Removes the agent and places it again, if outside boundary.
        // If an this happens, this is O(log n), but O(1) otherwise.
        virtual void moveAgent(const Ped::Tagent *a);

	// Removes agent a from tree structure
        virtual bool removeAgent(const Ped::Tagent *a);

	// Returns the set of agents that is stored within this tree
        virtual set<const Ped::Tagent*> getAgents() const;

	// Inserts all agents from this tree into the outputList
        virtual void getAgents(list<const Ped::Tagent*>& outputList) const;

	// Checks if a point x/y is within the space handled by the tree node, or within a given radius r
        // \return  true if the point is within the space
        // \param   px The x co-ordinate of the point
        // \param   py The y co-ordinate of the point
        // \param   pr The radius
        virtual bool intersects(double px, double py, double pr) const;

        double getx() const { return x; };
        double gety() const { return y; };
        double getw() const { return w; };
        double geth() const { return h; };

	// Returns the depth of this tree
        int getdepth() const { return depth; };
	
    protected:
	// A little helper that adds empty child nodes to this node 
        virtual void addChildren();

	// Returns the child that covers the coordinate (x,y)
        Ttree* getChildByPosition(double x, double y);

        // If the total number of agents kept in the children nodes
	// are too low, it moves all the agents into this node, and
	// deletes the child nodes.
	int cut();

	// Keeps track of the leaf that keeps an agent
	std::map<const Ped::Tagent*, Ped::Ttree*> *treehash;

	// Agents in this node (if it is a leaf)
        set<const Ped::Tagent*> agents;	

	// Denotes whether this node is a leaf
        bool isleaf;

	// starting coordinates of the area this node covers
        double x;
        double y;

	// width and height of area covered by this node
        double w;
        double h;
        int depth;
	int maxDepth;

	// Children (if set)
        Ttree *tree1;
        Ttree *tree2;
        Ttree *tree3;
        Ttree *tree4;

	// Tree root
	Ttree *root;
	
    };
}

#endif
