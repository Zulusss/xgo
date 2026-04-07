//---------------------------------------------------------------------------
#include <iostream>
#include <fstream>

#pragma hdrstop

#include "Builder.h"

//---------------------------------------------------------------------------

#pragma package(smart_init)


Builder::Builder(SimplyNumbers* s, Hashtable* h, int gameMode)
        : Expander(s, h){

    this->gameMode = gameMode;
};

//==================================================================

void Builder::buildTree() {
  //assert cursor==lastmove;

  count0 = count;
  building = true;
  CursorHistory *cur = current();

  int i;

  while (cur->node->totalDirectChilds > 0 && count < 224) {

    fullExpand(cur->node);

    i = chooseNodeToExpand();

    if (i == -1) {
        std::cout << "builder fallback. count0=" << count0
                << " count=" << count
                << " move=" << (int)current()->move
                << " totalDirectChilds=" << (int)current()->node->totalDirectChilds
                << " totalChilds=" << (int)current()->node->totalChilds
                << std::endl;
        //printHistory("builder fallback.", cur->node);
        cur->node->totalChilds += 10;
        back();
        cur = current();
        continue;
    }
    forward(childs.move[i], childs.node[i]);
    cur = current();
  }

  multiExpand(cur->node);

  int added = cur->node->totalChilds;

  goBack(count0);

  building = false;

  /*
  //Auto cleaning (not works)
  int crit = first->node->rating*2;
  if (crit < 0) {
        crit = -crit;
  }
  if (crit > 3300 && (expanded->rating < -crit || expanded->rating > crit)) {
        while (expanded->firstParent->rating < -crit || expanded->firstParent->rating > crit) {
                expanded = expanded->firstParent;
        }
        expanded->cleanChilds(crit, 0);
  }
  */
  //assert cursor==lastmove;
};


void Builder::goBack(int count0) {
  TMove critical;
  TRating cr;
  while(count>count0) {
    IF_READY_FOR_TRAIN { // Train neural network if we are prepared enough
        TNode *n = current()->node;
        critical = std::abs(n->rating) > 8000 ? current()->move : 0;
        back();
        n = current()->node;

        // Обучаем, если позиция достаточно изучена
        if (count > 0 && (n->totalChilds >= 40000
             || n->totalDirectChilds == 1
             || n->totalDirectChilds <= 4 && n->totalChilds >= 5000
            )) {
            //std::cout << "[AI] added = " << added << std::endl;

            static int skip = 0;
            if (++skip % (3+count*8) == 0) trainNetworkOnCurrentPosition();
        }
    }
    else
    {
        back();
    }
  }
}

//==================================================================

//int Builder::chooseNodeToExpand() {
//
//  float f0 = -99999;
//  int choosen = -1;
//  calculateChilds();
//
//  for (int i = 0; i < childs.count; ++i) {
//        TNode *node = childs.node[i];
////            if (node->rating <= - 32300 || node->rating >= 32300)
////                continue;
//
//        int ttc = node->ratingToTotalChilds();
//        int ret = (node->totalChilds > 60000) ? ttc*ttc : ttc;
//        float f = ret/ (float)(50+node->totalChilds);
//        if (f > f0 || f0 == -99999) {
//          choosen = i;
//          f0 = f;
//        }
//  }
//  return choosen;
//};


int Builder::chooseNodeToExpand() {
    calculateChilds();
    if (childs.count == 0) {
        std::cout << "Warning: child nodes not found!" << std::endl;
        return -1;
    }

    // 1. Calculate total children across all branches
    int totalAllChilds = 0; // = current()->node->totalChilds;
    for (int i = 0; i < childs.count; ++i) {
        totalAllChilds += childs.node[i]->totalChilds;
    }

    float maxUrgency = -1e9f;
    int chosen = -1;

    // 2. For each node, determine its rank by rating
    // Higher rating = lower rank (0 is highest)
    for (int i = 0; i < childs.count; ++i) {
        TNode *node = childs.node[i];

        int rank = 0;
        for (int j = 0; j < childs.count; ++j) {
            if (childs.node[j]->rating > node->rating) {
                rank++;
            }
        }

        // 3. Calculate target percentage: 50% for rank 0, 25% for rank 1, etc.
        // Target = totalAllChilds * (0.5 ^ (rank + 1))
        float targetPercent = 1.0f / (float)(1 << (rank + 1));
        float targetChilds = (float)totalAllChilds * targetPercent;

        // 4. Calculate urgency: how many nodes are we missing compared to the target?
        // We add a small constant (50) to the denominator to prevent over-expansion
        // of new nodes with very small totalChilds.
        float urgency = (targetChilds - node->totalChilds) / (float)(50 + node->totalChilds);

        if (urgency > maxUrgency || chosen == -1) {
            maxUrgency = urgency;
            chosen = i;
        }
    }

    return chosen;
};

