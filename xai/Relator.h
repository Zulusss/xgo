//---------------------------------------------------------------------------

#ifndef RelatorH
#define RelatorH
//---------------------------------------------------------------------------
#include "Evaluator.h"
#include "TNode.h"
#include "Logger.h"

class Relator : public Evaluator {

        protected:

        Relator(SimplyNumbers*, Hashtable* );
        void calculateChilds();
        RelativeBucket childs;
        void updateParents(int addedChilds);
        TNode* getChild(TNode *parent, TMove childMove);
        bool isPerspectiveChild(TMove move);
        bool updateNode(TNode *node, bool updateRating, int addedChilds);


//-------------------------------------
        private:

        //these are used fro parent calculation
//        void updateParents(int childsAdded);
        void updateParents(TNode *node, int removed, int removedFromEnd,
                bool onlyLastRemoved, bool updateRating, int max, int addedChilds);

        bool isPerspectiveChildMode1(TMove move);
        TNode* getParent(TNode *node, TMove move);
        int minRemovedEven;
        int minRemovedOdd;
};

#endif
