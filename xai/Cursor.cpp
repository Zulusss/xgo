//---------------------------------------------------------------------------


//#pragma hdrstop

#include <iostream>
#include <memory.h>
#include "Cursor.h"


//---------------------------------------------------------------------------

//#pragma package(smart_init)

CursorHistory::CursorHistory() {
        memset((void*)en, 0, sizeof(CursorHistory));
};

#include <cstdio> // Для snprintf


//-------------------------------  CURSOR -----------------------------------

Cursor::Cursor(SimplyNumbers *simplyGen, Hashtable *movesHash) {
        this->simplyGen = simplyGen;
        this->movesHash = movesHash;
        memset(history,0,sizeof(CursorHistory)*TOTAL_CELLS);
        memset(kl,0,sizeof(TMove)*TOTAL_CELLS);
        count = 0;
        count0 = 0;
        building = false;
        cnt = 0;
};

//=============================================================================

//inline
CursorHistory *Cursor::current() {
        return &(history[count-1]);
}

CursorHistory *Cursor::getMove(int i) {
        return &(history[i]);
}

TNode *Cursor::getFirstNode() {
        return history[0].node;
}

TNode *Cursor::lastMove() {
        return current()->node;
}

void Cursor::restart() {
  while(count > 1) {
     back();
  }
  //lastmove = first->node;
}



//=============================================================================
//Increment the Cursor to next move N. Returns true on success, otherwise - false.
bool Cursor::forward(TMove N) {

  TNode *node = current()->node;

  THash hashCodeX = node->hashCodeO;
  THash hashCodeO = node->hashCodeX * simplyGen->getHash(N);

  bool created = false;
  TNode *nextNode = movesHash->getOrCreate(hashCodeX, hashCodeO, node->age + 1, created);
  if (nextNode == NULL) {
    logger->log("Error creating node.");
    return false;
}
  if (created) {

    rate(node, nextNode, N);
    ++node->totalChilds;

    logger->missNode(nextNode);
    logger->log("Warning: not existed node.");
  }

  return forward(N, nextNode);
}

//=============================================================================
bool Cursor::forward(TMove N, TNode* node) {


  int x = N % 15, y = N / 15;

  //begin: forward cursor
  TByte prevVal = kl[N];
  kl[N] = ((count&1)+1)<<2;
  ++count;
  moves [count] = N;

  CursorHistory *curr = current(), *prev = &(history[count <= 1? 0 : count-2]);
  curr->node = node;
  curr->move = N;
  curr->previousKlValue = prevVal;


  //begin: update simmetries history
        int d = 0,i;
        d = x==7 ? 0 : (kl[y*15+(14-x)] == kl[N]) ? -1 : 1;
        curr->symmX = prev->symmX + d;

        d = y==7 ? 0 : (kl[(14-y)*15+x] == kl[N]) ? -1 : 1;
        curr->symmY = prev->symmY + d;

        i = (14-y)*15+(14-x);
        d = N==i? 0 : (kl[i] == kl[N]) ? -1 : 1;
        curr->symmXY = prev->symmXY + d;

        d = x==y ? 0 : (kl[x*15+y] == kl[N]) ? -1 : 1;
        curr->symmW = prev->symmW + d;

        i=(14-x)*15+y;
        d = N==i? 0 : (kl[i] == kl[N]) ? -1 : 1;
        curr->symmXW = prev->symmXW + d;

        i=x*15+14-y;
        d = N==i? 0 : (kl[i] == kl[N]) ? -1 : 1;
        curr->symmYW = prev->symmYW + d;

        i=(14-x)*15+14-y;
        d = N==i? 0 : (kl[i] == kl[N]) ? -1 : 1;
        curr->symmXYW = prev->symmXYW + d;
  //end: update symmetries history

  //begin: update "enablers" history
    curr->enCount = 0;
/*
    if (gameMode == 1 &&  count == 2) {
        for (int x1=4; x1<=10; x1++)
            for (int y1=4; y1<=10; y1++)
                if (x1==4||x1==10||y1==4||y1==10) {
                        TMove move = y1*fsize+x1;
                        if (kl[move] == 0) {
                            if (curr->enCount >= MAX_ENABLERS) {
                                curr->enCount = curr->enCount; //TODO
                            } else {
                                kl[move] = 1;
                                curr->en[curr->enCount++] = move;
                            }
                        }
                }
    } else if (gameMode == 1 &&  count == 3) {
        enable(curr, x, y, 1);
        enable(curr, 7, 7, 1);
        enable(curr, history[1].move%15, history[1].move/15, 1);
    } else */


    int max = count > 3 ? 2 : 1;
    enable(curr, x, y, max);

  //end: update "enablers" history

  if (current()->node == NULL) {
        return false;
  }

  return true;
};

//=============================================================================

void Cursor::enable(CursorHistory *curr, int x, int y, int maxDistance) {
        for (int d=1; d<=maxDistance; d++)
            for (int i=-1; i<=1; i++)
              for (int j=-1; j<=1; j++)
                if ((!i)&&(!j)) continue;
                else {
                    int x1 = x+i*d, y1 = y+j*d;
                    TMove move = y1*fsize+x1;
                    if (x1>=0 && y1>=0 && x1<fsize && y1<fsize && kl[move] == 0 ) {
                          if (curr->enCount >= MAX_ENABLERS) {
                              //curr->enCount = curr->enCount; //TODO
                          } else {
                              kl[move] = 1;
                              curr->en[curr->enCount++] = move;
                        }
                    }
                }

}

//set cursor to one move back
bool Cursor::back() {
  if (count == 0) {
        return false;
  }
  CursorHistory *curr = current();
  kl[curr->move] = curr->previousKlValue;
  for(int t=0; t<curr->enCount; t++)
        kl[curr->en[t]] = 0;
  --count;
  if (current()->node == NULL) {
        return false;
  }
  return true;
}

//=============================================================================

bool Cursor::isAlllowed(TMove N) {
    int x = N%fsize - 7;
    int y = N/fsize - 7;
    if ((  history[count-1].symmX  == 0) && x < 0) {
      return false;
    }
    if ((  history[count-1].symmY  == 0 || history[count-1].symmXY  == 0) && y < 0) {
      return false;                }
    if ((  history[count-1].symmW  == 0) && x < y) {
      return false;
    }
    if ((  history[count-1].symmXYW  == 0) && x < -y) {
      return false;
    }
    return allow(N);
};


//------------------------------------------------------------------------------

//bool Cursor::unique(TMove move) {
//        if (history[count-1].symmX && history[count-1].symmY && history[count-1].symmXY && history[count-1].symmW
//                //&& history[count-1].symmXW && history[count-1].symmYW && history[count-1].symmXYW
//                ) {
//                return childCount;
//        }

//                int x = move%fsize - 7;
//                int y = move/fsize - 7;

//                return (history[count-1].symmX  != 0 || x >= 0)
//                              && (history[count-1].symmY  != 0 || y >= 0)
//                              //&& (history[count-1].symmXY != 0 || x + y >= 0)
//                              && (history[count-1].symmW != 0 || x >= y)
//                              //&& (history[count-1].symmXW  != 0 || ((x>=0 || y<=0) && (x<=0 || y>=)0))
//                              //&& (history[count-1].symmYW != 0 || ((x>=0 || y<=0) && (x<=0 || y>=0)))
//                              && (history[count-1].symmXYW != 0 || x + y  >= 0)
//                              && (allow(move));
//}

//------------------------------------------------------------------------------

//initial implementation for supporting of prohibited moves for
//balanced games, currently supported 1st player
//to make 2nd move in central 5x5 square
inline bool Cursor::allow(int move) {
  if (gameMode == 1 && this->lastMove()->age == 1) {
        int x = move % 15 - 7, y = move / 15 - 7;
        return ! (x < 3 && x > -3 && y < 3 && y > -3);
  }
  return true;
}


//------------------------------------------------------------------------------

int Cursor::getMovesCount() {
  return count0 > 0 && count0<=count ? count0 : count;
}


//------------------------------------------------------------------------------

void Cursor::printHistory(const char* pref, TNode *node) {
    char buf[200];
    node->printPosition(buf, 200);
    std::cout << pref << buf << " \n";
    std::cout << "nChilds=" << node->totalChilds
        << " Direct " << (int)(node->totalDirectChilds)
        << " Rage Def " << node->isRageDef()
        << " Rage Attk " << node->isRageAttack()
        << " Rating " << node->rating
        << " Attacks count (self)  " << (int)node->o4 << "+" << (int)node->o3 << "+" << (int)node->o2
        << " Attacks count (op)  " << (int)node->x4 << "+" << (int)node->x3 << "+" << (int)node->x2
        << "\n";
    printHistory();
}

void Cursor::printHistory() {
    // 1. Собираем ходы X (Крестики) - теперь это 2-й, 4-й ходы партии (индексы 0, 2, 4...)
    std::cout << "X: ";
    bool firstX = true;
    for (int i = 0; i < count; i += 2) {
        if (!firstX) std::cout << ", ";
        std::cout << (int)history[i].move;
        firstX = false;
    }

    // 2. Собираем ходы O (Нолики) - теперь это 3-й, 5-й ходы партии (индексы 1, 3, 5...)
    std::cout << " | O: ";
    bool firstO = true;
    for (int i = 1; i < count; i += 2) {
        if (!firstO) std::cout << ", ";
        std::cout << (int)history[i].move;
        firstO = false;
    }
    std::cout << std::endl;
}
