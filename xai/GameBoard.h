//---------------------------------------------------------------------------

#ifndef GameBoardH
#define GameBoardH
//---------------------------------------------------------------------------

#include "Builder.h"

//can flip board
class GameBoard : public Builder {

public:

        GameBoard(SimplyNumbers *simplyGen, Hashtable *movesHash, int gameMode);
        int getCell(int Col, int Row);

protected:
        bool put(TMove N);//can flip board
        int move();
        int transform(int x, int y);

        bool swapX, swapY, swapW;

};

#endif
 