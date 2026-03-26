//---------------------------------------------------------------------------
#ifndef BuilderH
#define BuilderH
//---------------------------------------------------------------------------
#include "Expander.h"

class Builder : public Expander {
public:
    Builder(SimplyNumbers*, Hashtable*, int gameMode );
    void buildTree();

    virtual void trainNetworkOnCurrentPosition(){};
    virtual void trainNetworkOnSingleMove(TMove move, TRating rating){};

private:
    int chooseNodeToExpand();
};

#endif
