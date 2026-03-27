//---------------------------------------------------------------------------

#ifndef ExpanderH
#define ExpanderH
//---------------------------------------------------------------------------

#include "Relator.h"

#define MULTI_EXPAND_TIMES 5

class MovesBucket {
        public:
        int count;
        TMove move[MAX_RELATIVES];
        void add(TMove m);
        bool contains(TMove m);
};

class Expander : public Relator {

public:
        int max_count;

protected:
        Expander(SimplyNumbers *simplyGen, Hashtable *movesHash);
        TMove expand(int start, TNode* cursor);
        void multiExpand(TNode* cursor);
        void fullExpand(TNode* cursor);
        bool isExpected(TNode* curr, TMove i);

private:
        int cnt;
        MovesBucket newChilds;
        MovesBucket otherNewChilds; //second priority
        void findMovesToExpand(int start); //fills in newChilds and otherNewChilds
        void addChildNoDupe(TNode* parent, TMove m);
        void addChild(TNode* parent, TMove m);
};

#endif
