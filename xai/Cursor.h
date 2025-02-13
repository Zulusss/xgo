//---------------------------------------------------------------------------

#ifndef cursorH
#define cursorH
#include "TNode.h"
#include "SimplyNumbers.h"
#include "Hashtable.h"

#define fsize 15  //field size
#define TOTAL_CELLS 225 //fsize*fsize, cells total

#define MAX_ENABLERS 32
#define MAX_RELATIVES 224 // == (fsize*fsize - 1)
//---------------------------------------------------------------------------
class RelativeBucket {
        public:
        int count;
        TNode* node[MAX_RELATIVES];
        TMove move[MAX_RELATIVES];
};

struct CursorHistory {
    CursorHistory();
    TMove en[MAX_ENABLERS],//"Enabled" moves buffer
          move;
    int enCount;//count of actually "Enabled" moves
    int symmX, symmY, symmXY, symmW, symmXW, symmYW, symmXYW;
    TNode *node;
    RelativeBucket parents;
    int previousKlValue;
//  bool removed;
};


class Cursor {

public:

  Logger *logger;
  int getMovesCount();

protected:

  int gameMode;//0 = Go - Moku, 1 = 5-in-a-row, 2 = Renjue
  int count;  //count of moves made (may include imaginary moves of AI)
  int count0; //count of moves made (only by players)
  bool building; //indicates that some moves made by AI while executing buildTree() function, which will be taken back

  //inline
  CursorHistory *current();
  TNode *getFirstNode();
  TNode *lastMove();
  CursorHistory *getMove(int i);
  void restart();

  Hashtable *movesHash;
  CursorHistory history[TOTAL_CELLS];


  SimplyNumbers *simplyGen;
  TMove moves[fsize*fsize]; //ordered history of moves

  TMove kl[fsize*fsize]; // moves on field
  bool isAlllowed(TMove N);

private:
        inline bool allow(int move);
        void enable(CursorHistory *curr, int x, int y, int maxDistance);
protected:

        //bool unique(TMove move);

        Cursor(SimplyNumbers *simplyGen, Hashtable *movesHash);

        bool forward(TMove N);
        bool forward(TMove N, TNode* node);
        bool back();

};


#endif
