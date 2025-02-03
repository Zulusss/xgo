//---------------------------------------------------------------------------


#pragma hdrstop

#include "GameBoard.h"
#include <cstring>

//---------------------------------------------------------------------------

#pragma package(smart_init)

GameBoard::GameBoard(SimplyNumbers *simplyGen, Hashtable *movesHash,
                  int gameMode) : Builder(simplyGen, movesHash, gameMode) {

    swapX = swapY = swapW = false;

};

bool GameBoard::put(TMove N) {
                int x = N%fsize - 7;
                int y = N/fsize - 7;
                bool symmW = history[count-1].symmW == 0;
                bool swapped = false;

                if (history[count-1].symmX  == 0) {
                    if (x < 0) {
                        x = -x;
                        if (swapW) {
                                swapY = !swapY;
                        } else {
                                swapX = !swapX;
                        }
                    }
                }
                if ((  history[count-1].symmY  == 0 || history[count].symmXY  == 0) && y < 0) {
                        y = -y;
                        if (swapW) {
                                swapX = !swapX;
                        } else {
                                swapY = !swapY;
                        }
                }

                if ( symmW && !swapped == x < y && x!= y) {
                        int t = x;
                        x = y;
                        y = t;
                        swapW = !swapW;
                }

                return forward((y+7)*15+(x+7));
};


int GameBoard::move() {

  TNode *choosen = 0;

  TNode *node = current()->node;
  int totl = node->totalDirectChilds;
  if (totl == 0) {
        //logger->error("no childs");
        expand();
        //return -32600;
  }


    int choosenMove;
    int rating;
    bool mode1 = gameMode == 1 &&  count == 2;
    for (int i = 0; i < TOTAL_CELLS; ++i) {
        if (mode1 ? kl[i]<=1 && isPerspectiveChildMode1(i) : isPerspectiveChild(i)) {
            TNode* n = getChild(node, i);
            if (n != NULL && (choosen == NULL || rating < n->rating)) {
                choosenMove = i;
                choosen = n;
                rating = choosen->rating;
            }
        }
    }

  if (choosen == NULL) {
         //TODO 
        //logger->error("no move");
        return -32600;
  }

  if (node->rating!=-rating) {
         //TODO
         // logger->error("parents updated");
        updateNode(node,choosen,true,0,0);
        updateParents(0);
  }

  forward(choosenMove, choosen);
  return rating;
};

//-------------------------------------------------------------------------------


//return codes:
// 1  X
// 2  X (last move)
//-1  O
//-2  O (last move)
//0   empty cell
//3+  empty cell with hint
int GameBoard::getCell(int Col, int Row) {


    int N = transform(Col, Row);
    bool found = false;
    int bold = 1;
    int movesCount = building ? count0 : count;
    for(int i = 0; i < movesCount; ++i) {
      if (getMove(i)->move == N) {
          found = true;
          bold = ((i == count - 1) ? 2 : 1);
          break;
      }
    }
    if (found && kl[N]&8) {//draw 'o'
        return -bold;
    }
    if (found && kl[N]&4) {//draw 'x'
        return bold;
    }
//    if (dkl[N]>=1  && isAlllowed(N)) {//draw empty cell with hint
//        return dkl[N] + 2;
//    }
    return 0;//empty cell
};


// x,y - user space coordinates, from 0 to 14
//
//returns index of kl array

int GameBoard::transform(int x, int y) {

    if (swapX) {
        x = 14-x;
    }
    if (swapY) {
        y = 14-y;
    }
    if (swapW) {
        int t = x;
        x = y;
        y = t;
    }
    return y*15+x;
};
