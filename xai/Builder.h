//---------------------------------------------------------------------------
#ifndef BuilderH
#define BuilderH
//---------------------------------------------------------------------------
#include "Expander.h"

#define TRAIN_FROM 900000
#define IF_TRAIN_READY if (history[0].node->totalChilds > TRAIN_FROM)

class Builder : public Expander {
public:
    Builder(SimplyNumbers*, Hashtable*, int gameMode );
    void buildTree();
    void goBack(int c0);

    virtual void trainNetworkOnCurrentPosition(){};
    virtual void trainNetworkOnSingleMove(TMove move, TRating rating){};

private:
    int chooseNodeToExpand();
};

#endif
