//---------------------------------------------------------------------------

#ifndef GrowerH
#define GrowerH
//---------------------------------------------------------------------------

#include "Neuro.h"

class Grower : public Neuro {

public:
        Grower(SimplyNumbers *simplyGen, Hashtable *movesHash, int gameMode);
        void grow();
        void gridClick(int Col, int Row);
        int  getRResult();
        void restartClick();
        void takeBackClick();
        void moveClick();
        void moveNeuroClick();

        char* getMsg1();
        char* getMsg2();
        char* getMsg3();
        char* getMsg4();
        char* getMsg5();
        char* getMsg6();
        char* getMsg7();
        char* getMsg8();
        char* getMsg9();
        char* getMsgStatus();

        volatile bool doTrain = false;
        volatile bool onlySelfPlay = false;
        void SwitchPlayMode(char* mode);

protected:
        volatile int userMoveRequested;
        int resultRecieved;

private:
        int neuroWinsXCount = 0;
        int neuroWinsOCount = 0;
        int regularWinsXCount = 0;
        int regularWinsOCount = 0;
        int drawsCount = 0;
        int nnXCount = 0;
        int nnOCount = 0;
        int nnDCount = 0;

        bool restartRequested;
        bool takeBackRequested;
        bool exitRequested;
        bool moveRequested;
        bool moveNeuroRequested;

        int movesCount;

        unsigned char dkl[fsize*fsize];

        char msg1[200];
        char msg2[200];
        char msg3[200];
        char msg4[200];
        char msg5[200];
        char msg6[200];
        char msg7[200];
        char msg8[200];
        char msg9[200];
        short int xRating;

        char msgStatus[200];

};

Grower* getXBoard(int gameMode);

#endif
 