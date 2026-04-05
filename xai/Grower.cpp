//---------------------------------------------------------------------------
#include <iostream>

#pragma hdrstop

#include "Grower.h"
#include <cstring>
#include <thread>
#include <chrono>

//---------------------------------------------------------------------------

#pragma package(smart_init)

Grower::Grower(SimplyNumbers *simplyGen, Hashtable *movesHash,
                  int gameMode) :  Neuro(simplyGen, movesHash, gameMode) {

    bool isCreated;
    TNode *node = movesHash->getOrCreate(1, 1, 0, isCreated);

    userMoveRequested = 255;
    forward(112, node);
    multiExpand(node);
};



#define MAX_CHILDS_WIZARD  40000000

#define MAX_CHILD_PER_MOVE_ZONE0  30000000 //30m
#define MAX_CHILD_PER_MOVE_ZONE1  2000000

#define MAX_CHILD_DECREASE_FACTOR 6
#define MAX_CHILD_DECREASE_SINCE 30000000
#define MAX_CHILD_DECREASE_TILL 90000000

#define ZONE01_RATING 6100
#define ZONE12_RATING 10000


void Grower::grow() {
  static int count;

  int wizardMode = 1000;
  short int flowRating = 0;

//   SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL)) 

  int childs0 = 0;
  int playMode = 1;//0= Human vs Human, 1= Comp vs Human, 2= Comp vs Comp, 3= Debuts calc

  bool neuroPlays = true;
  bool neuroPlaysO = true;
   
//  unsigned long beginTime = GetTickCount();
  while (!exitRequested) {

        //********* STEP 1   initialize ***************

        if (playMode < 2) { //Human vs Human or Human vs Comp
              wizardMode = 0;
        } else if (wizardMode == 0 && playMode >= 2) { // Comp vs Comp or Debuts calculation
            wizardMode = 1000;
        }

        bool changed = true;

        TNode *firstNode = getFirstNode();

        TRating currRating = lastMove()->rating;
        TRating currXRating = count%2
                        ? currRating
                        : -currRating;

        int delta = currXRating - firstNode->rating;
        if (delta < 0) {
                delta = -delta;
        }
        bool medRating = (delta < 500);
        unsigned int totalChilds = firstNode->totalChilds;

        bool mediumicPlay = medRating;
/*              && (count <  4
                || (count <  5 && totalChilds >  500000)
                || (count <  6 && totalChilds >  600000)
                || (count <  7 && totalChilds >  700000)
                || (count <  8 && totalChilds >  800000)
                || (count <  9 && totalChilds > 1200000)
                || (count < 11 && totalChilds > 1400000)
                || (count < 13 && totalChilds > 2000000)
                || (count < 15 && totalChilds > 3000000)
                || (count < 17 && totalChilds > 4000000)
                || (count < 18 && totalChilds > 7000000)
                || (count < 19 && totalChilds >18000000)
                || (count < 20 && totalChilds >20000000)
                || (count < 21 && totalChilds >23000000)
                );*/

        int childPerMove = 13000;//CSpinEditChilds->Value*1000;
        if (wizardMode && childs0 && childs0 >= childPerMove && childs0 < childPerMove*2) {
                childPerMove = childs0 + 50000;
        }

        double mcdf = totalChilds > MAX_CHILD_DECREASE_SINCE
                ? totalChilds > MAX_CHILD_DECREASE_TILL
                        ? MAX_CHILD_DECREASE_FACTOR
                        : MAX_CHILD_DECREASE_FACTOR
                                * ((totalChilds - MAX_CHILD_DECREASE_SINCE)/1000)
                                / (double)((MAX_CHILD_DECREASE_TILL - MAX_CHILD_DECREASE_SINCE)/1000)
                : 1;
        if (mcdf < 1) mcdf=1;

        int maxChilds = currRating < ZONE01_RATING && currRating > -ZONE01_RATING
                ? MAX_CHILD_PER_MOVE_ZONE0 / mcdf
                : currRating < ZONE12_RATING && currRating > -ZONE12_RATING
                        ? MAX_CHILD_PER_MOVE_ZONE1 / mcdf
                        :childPerMove + 1;
        if (maxChilds < childPerMove) {
                maxChilds = childPerMove;
        }

        //========= !!! NEURO vs Comp AUTOPLAY CODE BEGINS ==========
        IF_TRAIN_READY {
            float lastRating = lastMove()->rating;
            bool isWin = std::abs(lastRating) > 9000;
            static bool neuroWithNeuro = false, nwnActivated = false;

            // Условие перезапуска: победный рейтинг или лимит ходов
            if (!restartRequested && (this->count > 21 || (isWin && this->count > 1))) {
                restartRequested = true;

                if (!neuroWithNeuro && drawsCount+neuroWinsCount > 15) {
                    neuroWithNeuro = true;
                    if (!nwnActivated) {
                        nwnActivated = true;
                        std::cout << "    Neuro vs Neuro mode activated!" << std::endl;
                    }
                }
                int totalPlayed = drawsCount+neuroWinsCount+regularWinsCount+nnXCount+nnOCount+nnDCount;
                if (neuroWithNeuro && !neuroPlaysO && totalPlayed%4>1) {
                    neuroWithNeuro = false;
                }

                bool xWon = this->count % 2 == lastRating > 0;

                bool log = totalPlayed % 9 == 0;

                if (!isWin) {
                    if (log) std::cout << "[RESTART] Reason: DRAW (";
                    drawsCount++;
                } else {

                    // Определяем, была ли нейросеть тем, кто сделал последний ход
                    bool neuroMovedLast = (neuroPlaysO == (this->count % 2 == 0));

                    // Нейросеть победила, если:
                    // 1. Она ходила последней и рейтинг положительный (она создала победную ситуацию)
                    // 2. Она НЕ ходила последней и рейтинг отрицательный (соперник подставился)
                    bool neuroWon = (neuroMovedLast == (lastRating > 0));

                    if (neuroWithNeuro) {

                        if (!isWin) {
                        if (log) std::cout << "[RESTART] Reason: N-N DRAW (";
                            nnDCount++;
                        }
                        else if (xWon) {
                            trackerNNX->addLoss(this->count);
                            if (log) std::cout << "[RESTART] Reason: NX WON (";
                            nnXCount++;
                        } else {
                            trackerNNO->addLoss(this->count);
                            if (log) std::cout << "[RESTART] Reason: NO WON (";
                            nnOCount++;
                        }
                    }
                    else if (neuroWon) {
                        neuroWinsCount++;
                        if (log) std::cout << "[RESTART] Reason: Neuro WON (";

                        if (neuroPlaysO) {
                            trackerNO->addLoss(this->count);
                        } else {
                            trackerNX->addLoss(this->count);
                        }
                    } else {
                        regularWinsCount++;
                        if (log) std::cout << "[RESTART] Reason: Legacy WON (";

                        if (neuroPlaysO) {
                            trackerLX->addLoss(this->count);
                        } else {
                            trackerLO->addLoss(this->count);
                        }
                    }
                }

                if (log) {
                    std::cout << (xWon ? " X " : " O ")
                          << " Count: " << this->count
                          << " Rating: "
                          << lastRating << ")";
                    if (neuroWithNeuro) {
                        std::cout
                              << " CURRENT SCORE -> NX: " << nnXCount
                              << " | NO: " << nnOCount
                              << " | N-Draws: " << nnDCount
                              << " Avg.Age, NNX/NNO: "
                              << trackerNNX->toString() << " / "
                              << trackerNNO->toString() << std::endl;
                    } else {
                        std::cout
                              << " CURRENT SCORE -> Neuro: " << neuroWinsCount
                              << " | Legacy: " << regularWinsCount
                              << " | Draws: " << drawsCount
                              << " Avg.Age, NX/NO/LX/LO: "
                              << trackerNX->toString() << " / "
                              << trackerNO->toString() << " / "
                              << trackerLX->toString() << " / "
                              << trackerLO->toString() << std::endl;
                    }
                 }

                trainFromGame( lastRating > 0);

                // Смена ролей и сброс
                if (neuroPlays) neuroPlaysO = !neuroPlaysO;
                //std::cout << "Requested restart. Neuro next: " << (neuroPlaysO ? "O" : "X") << std::endl;
            }
            else if (lastMove()->totalChilds > childPerMove) {
                // Выбор следующего игрока:
                // Если count=0 (ход X) и neuroPlaysO=false — ходит нейросеть
                if (neuroPlays && (neuroWithNeuro || neuroPlaysO == (this->count % 2 != 0))) {
                    moveNeuroRequested = true;
                    //std::cout << "Requested neuro move " << (this->count%2?"(O)":"(X)") << std::endl;

                } else {
                    moveRequested = true;
                    //std::cout << "Requested legacy move " << (this->count%2?"(O)":"(X)") << std::endl;
                }
            }
        }
        //================= !!! NEURO AUTOPLAY CODE ENDS =======================


        //********* STEP 2   process requested actions ***************
        if (restartRequested) {
            restartRequested = false;
            //goBack(1);
            restart();
            //std::cout << "restarted" << std::endl;
        } else if (userMoveRequested != 255) {
                int i;
                bool res = put(userMoveRequested);
                if (res) {
                        int totl = lastMove()->totalChilds;
                        resultRecieved = lastMove()->rating;
                        if (res < 32600) {
                            //moveRequested = true;
                        }
                }
                userMoveRequested = 255;

        } else if (moveNeuroRequested || moveRequested &&
                (wizardMode || lastMove()->totalChilds >= childPerMove
                        || lastMove()->rating < -20000
                        || lastMove()->rating > 20000
                        || lastMove()->totalDirectChilds == 1)) {

          resultRecieved = moveNeuroRequested ? moveNeuro() : move();
          moveNeuroRequested = false;
          moveRequested = false;
          flowRating = currRating;
          childs0 = lastMove()->totalChilds;
          currRating = lastMove()->rating;
        } else if (takeBackRequested) {
            takeBackRequested = false;

            if (wizardMode>0) {
                if (totalChilds > MAX_CHILDS_WIZARD) {
                        wizardMode = 0;
                } else {
                        --wizardMode;
                }
                if (wizardMode ==0) {
//                // drop to human vs comp from Comp vs comp modes if "takeback" pressed
                        if (playMode == 2) { //comp vs comp
                                restartRequested = true;
                                playMode = 1; //human vs comp
                        }
                }
            }

            if (this->count > 1) {
                back();
            }

            currRating = lastMove()->rating;
        }

        //********* STEP 3   autoplay stuff ***************

        if (playMode == 3 && wizardMode > 0) {//Show debuts
                if (!medRating) {
                        if (count%5 == 0 || count%7 == 0) {
                                restartRequested = true;
                        } else {
                                takeBackRequested = true;
                        }
                        ++count;
                        continue;
                }
        }

        unsigned int lastCount = lastMove()->totalChilds;
        if (wizardMode >= 0 && (playMode > 1)
                && (lastCount >= (mediumicPlay
                                        ? childPerMove
                                        : 20000))) {
                moveRequested = true;
                continue;
        }

        //********* STEP 4   tree grow ***************

        if (lastCount <  maxChilds
                        && currRating < 32300
                        && currRating > -32300) {
          for (int i=0; i<10; ++i) {
            buildTree();
          }
          changed = false;
          ++count;
        } else {
          
          std::this_thread::sleep_for(std::chrono::nanoseconds(500000000));
          if (wizardMode) {

              if (mediumicPlay || playMode == 2) { //Comp vs Comp

                    moveRequested = true;
              } else
              {
                    takeBackRequested = true;
              }
              continue;
          }
        }

        //********* STEP 5   stat outputs ***************

        if (!(count%4)) {
              int min = 1000000000;
              int max = 0;
              int decr;
              int i=0;

              TNode *node = lastMove();

              //begin hints calculation
              calculateChilds();
              for(i=0; i < childs.count; ++i) {
                int r = childs.node[i]->rating;
                if (r<min) min = r;
                if (r>max) max = r;
              }

              memset(dkl, 0, fsize*fsize);
              for(i=0; i<childs.count; ++i) {

                int r = childs.node[i]->rating;
                decr = min-1;
                dkl[childs.move[i]] = 30+(r-decr)*225/(max-decr);
              }
              //end hints calculation

              movesCount = count;
              int i1 = firstNode->totalChilds;
              int i2 = node->totalChilds;

              sprintf(msg1, "Childs count: %d%c / %d%c (%d)",
                            i1 / (i1 > 2000000 ? 1000000 : i1 > 2000 ? 1000 : 1), (i1 > 2000000 ? 'M' : i1 > 2000 ? 'K' : ' '),
                            i2 / (i2 > 2000000 ? 1000000 : i2 > 2000 ? 1000 : 1), (i2 > 2000000 ? 'M' : i2 > 2000 ? 'K' : ' '),
                            node->totalDirectChilds);

              sprintf(msg2, "Rating: %d / %d",
                        getFirstNode()->rating,
                        node->rating);

              sprintf(msg3, "Max path length: %d",
                        max_count);

              sprintf(msgStatus, wizardMode
                        ? "Please either switch playing mode or just wait.."
                        : restartRequested || moveRequested || takeBackRequested || userMoveRequested != 255
                              ? "Please wait.."
                              : "Please make your move. Or, you can give it to me, by pressing Move button.");

              logger->printLastError(msg4);
//              sprintf(msg4, movesHash->miss3 > 0 || movesHash->miss4 > 0
//                        ? "Hash collisions %d / %d" : "\0\0", movesHash->miss3, movesHash->miss4);

              double updateFreq = (TNode::updatesCount || TNode::skippedCount)
                        ? 100 * TNode::updatesCount / (double)(TNode::updatesCount + TNode::skippedCount)
                        : 0;
              sprintf(msg5, "Dev: %.3f%% : %d [%d / %d] [%d / %d]",
                        updateFreq,
                        (int)TNode::maxUpdated,
                        (int)TNode::updatesCount,
                        (int)TNode::skippedCount,
                        (int)TNode::avgDiff,
                        (int)TNode::avgSquareDiff
                        );

              logger->printMissStats(msg6);
              node->printPosition(msg7, 200);
              node->printScores(msg8, 200, this->count, neuroPlaysO);

              sprintf(msg9, "Trained [field: %d, single: %d, f.skip: %d]", trainedFieldCount, trainedSingleCount, skipTrainFieldCount);
              //current()->printAttacks(msg9, 200);

/* TODO
              if (xo != NULL) {
                unsigned long time = GetTickCount() - beginTime;
                sprintf(msg5, "RTime: %.3f%%",
                        100 * rateTime / (float)time,
                        (int)TNode::avgDiff,
                        (int)TNode::avgSquareDiff);
              }
*/

              xRating = count%2
                ? lastMove()->rating
                : -lastMove()->rating;

        }
  }
}


void Grower::gridClick(int Col, int Row) {

  userMoveRequested = transform(Row, Col);
}

int Grower::getRResult(){

  return resultRecieved;
};

void Grower::moveClick() {

    moveRequested = true;
}

void Grower::moveNeuroClick() {

    moveNeuroRequested = true;
}

void Grower::restartClick() {

    restartRequested = true;
}

void Grower::takeBackClick() {

    takeBackRequested = true;
}

//------------------------
Grower* xo = NULL;

Grower* getXBoard(int gameMode) {

    if (xo == NULL) {

        Logger *logger = new Logger();
        SimplyNumbers *sn = new SimplyNumbers();
        Hashtable *ht = new Hashtable(logger);
        std::cout << "Start!\n";
        xo = new Grower(sn, ht, gameMode);
        xo->logger = logger;
    }
    return xo;
};

//------------------------
char* Grower::getMsg1() {
  return msg1;
};

char* Grower::getMsg2() {
  return msg2;
};

char* Grower::getMsg3() {
  return msg3;
};

char* Grower::getMsg4() {
  return msg4;
};

char* Grower::getMsg5() {
  return msg5;
};

char* Grower::getMsg6() {
  return msg6;
};

char* Grower::getMsg7() {
  return msg7;
};

char* Grower::getMsg8() {
  return msg8;
};

char* Grower::getMsg9() {
  return msg9;
};

char* Grower::getMsgStatus() {
  return msgStatus;
};
