#include <iostream>

#include "TNode.h"

// Define and optionally initialize the static members
volatile double TNode::avgDiff = 0.0;
volatile double TNode::avgSquareDiff = 0.0;
volatile long long TNode::updatesCount = 0;
volatile long long TNode::skippedCount = 0;
volatile long long TNode::maxUpdated = 0;
volatile TNode* TNode::first = NULL;

TNode::TNode() {
    //std::cout << "TNode Start!\n";
    totalChilds = 0;
    rating = 0;
    flags = 0;
    totalDirectChilds = 0;
    hashCodeX = 1;
    hashCodeO = 1;
    x2 = x3 = x4 = o2 = o3 = o4 = 0;
    next = 0;
    ownAttacks = 0;
#ifdef STORE_ATTACKS
    attacks[0].l = attacks[0].r = 0;
#endif
    if (first == NULL) first = this;
};


#include <cstdio>
#include <string>

void TNode::printPosition(char* buffer, size_t size) {
    // 1. Печатаем хеши
    int offset = snprintf(buffer, size, "Hash %llu / %llu",
                          (unsigned long long)hashCodeX,
                          (unsigned long long)hashCodeO);

    if (isFixedRating()) {
        offset += snprintf(buffer + offset, size - offset, " Fixed ");
    }

#ifdef STORE_ATTACKS
    // Проверяем, есть ли вообще атаки (хотя бы первая пара не нулевая)
    if (attacks[0].l != 0 || attacks[0].r != 0) {
        offset += snprintf(buffer + offset, size - offset, " | Atks: ");
    }

    // 2. Проходим по массиву атак
    for (int i = 0; i < MAX_ATTACK_2; ++i) {
        if (attacks[i].l == 0 && attacks[i].r == 0) break; // Конец списка
        if (offset >= size - 10) break; // Запас для предотвращения обрезания

        bool isOwn = (i < ownAttacks); // Первые ownAttacks элементов — наши

        // Формат вывода: [L-R] для своих, (L-R) для чужих
        // Если L == R, выводим одно число для экономии места
        char open = isOwn ? '[' : '(';
        char close = isOwn ? ']' : ')';

        int written;
        if (attacks[i].l == attacks[i].r) {
            written = snprintf(buffer + offset, size - offset, "%c%hhu%c ",
                               open, (unsigned char)attacks[i].l, close);
        } else {
            written = snprintf(buffer + offset, size - offset, "%c%hhu-%hhu%c ",
                               open, (unsigned char)attacks[i].l, (unsigned char)attacks[i].r, close);
        }


        offset += written;
    }
#endif
}

// Печатает счётчики двоек, троек и четвёрок для каждого игрока
void TNode::printScores(char* buffer, size_t size, int count, bool neuroPlaysO) {
    snprintf(buffer, size, "%s %d: %d+%d+%d / %d+%d+%d Next move: %c",
            neuroPlaysO ? "NO" : "NX",
            count,
             (int)o4, (int)o3, (int)o2,
             (int)x4, (int)x3, (int)x2,
             count%2?'O':'X'
             );
}

/*
//==================================================================
int TNode::removeChild(TNode *child) {
        int found = 0;
        for(int i=0; i<totalDirectChilds; ++i) {
                if (childs[i].node == child) {
                        ++found;
                } else if (found > 0) {
                        childs[i-found].node = childs[i].node;
                        childs[i-found].move = childs[i].move;
                }
        }
        if (totalDirectChilds > found) {
                totalDirectChilds -= found;
        } else {
                totalDirectChilds = 0;
                delete[] childs;
        }
        return 0;
}
//==================================================================
void TNode::remove() {
        firstParent->removeChild(this);
        TNodeLink2** next = &nextParent;
        while((*next) != NULL) {
                (*next)->node->removeChild(this);
                next = &((*next)->next);
        }
}

//==================================================================

int TNode::cleanChilds(int critRating, int recursion) {
  int cleaned = 0;
  for(int i=0; i < totalDirectChilds; ++i) {
        int c = childs[i].node->cleanChilds(critRating, recursion+1);
        if (c > 0 && childs[i].node->totalDirectChilds == 0) {
                delete childs[i].node;
        }
        cleaned += c;
  }
  if (recursion > 0 && totalDirectChilds == 0) {
        if (rating > -critRating && rating < critRating) {
                return 0;
        }
        remove();
        ++cleaned;
  }
  return cleaned;
}

//==================================================================

int TNode::checkChilds(int recursion) {
  if (!totalDirectChilds) {
        return 0;
  }
  int violationsCount = 0;
  int maxR = -33000;
  for(int i=0; i < totalDirectChilds; ++i) {
        if (childs[i].node->rating > maxR) {
                maxR = childs[i].node->rating;
        }
  }
  maxR = -maxR;

  float k = maxR == rating
        ? 1
        : rating/ (float)(maxR ? maxR : 1);
  if (k < 0.9 || k > 1.1) {
        ++violationsCount;
  }

  for(int i=0; i < totalDirectChilds; ++i) {
        violationsCount += childs[i].node->checkChilds(recursion+1);
  }
  return violationsCount;
}
  */

  
//==================================================================

int TNode::ratingToTotalChilds() {

//  if (age == 1 && rating < -firstNode->rating*1.01 && hashCodeX == 1 && hashCodeO <=3
//        && firstNode->rating > 0 && this != firstNode) {
//    rating = -firstNode->rating*1.01;
//  }

  if (rating < -20000) {
    return 1;
  }
  if (rating < -10000) {
    return 2 + (rating+20000)/1000;//2..12
  }

  if (rating < -4500) {
    return 13 + (rating+10000)/200;//13..40
  }

  if (rating < -2000) {
    return 41 + (rating+4500)/10;//41..291
  }

  if (rating < 20000) {
    return 292 + (rating+2000);//292..22292
  }

  return 22292;
}




//==================================================================
#define MANY_CHILDS 3000
void TNode::update(short int newRating, unsigned int addedChilds) {
    totalChilds += addedChilds;

    if (rating != newRating) {

        if (rating == 0) {
            rating = newRating;
//        } else if ((rating <= - 16000 && newRating > rating) ||
//                (rating >= 16000 && newRating < rating)) {
                //suppress decreasing of rating
//                if (totalChilds > 150000) {
//                        ++TNode::skippedCount;
//                }
        } else {
                if (this==first){//
                    double diff = (newRating - rating);
                    double sqDiff = diff > 0 ? diff : -diff;
                    avgDiff = (avgDiff*updatesCount + diff)/(updatesCount+1);
                    avgSquareDiff = (avgSquareDiff*updatesCount + sqDiff)/(updatesCount+1);
                    ++updatesCount;
                    maxUpdated = totalChilds;
                }

//                if (newRating > 16384) {
//                        rating = newRating - 1;
//                } else if (newRating < -16384) {
//                        rating = newRating + 1;
////                } else if (totalChilds < 50000) {
//                } else {
                        if (!isFixedRating()) {
                                rating = newRating;
                        }
//                } else if (totalChilds < 150000) {
//                        rating = (short int)floor(0.5 + 0.99*newRating + 0.01*rating);
//                } else {
//                        rating = (short int)floor(0.5 + 0.98*newRating + 0.02*rating);
//                }
        }
    } else {
        if (this==first){//totalChilds > MANY_CHILDS) {
                ++skippedCount;
        }
    }
};


//==================================================================

void TNode::setFixedRating(bool set) {
    if (set) flags |= FLAG_FIXED_RATING;
    else flags &= ~FLAG_FIXED_RATING;
};
bool TNode::isFixedRating() {
    return (flags & FLAG_FIXED_RATING) != 0;
};

// --- Управление флагом Атаки ---
void TNode::setRageAttack(bool set) {
    if (set) flags |= FLAG_RAGE_ATTACK;
    else flags &= ~FLAG_RAGE_ATTACK;
}
bool TNode::isRageAttack() {
    return (flags & FLAG_RAGE_ATTACK) != 0;
}

// --- Управление флагом Защиты ---
void TNode::setRageDef(bool set) {
    if (set) flags |= FLAG_RAGE_DEF;
    else flags &= ~FLAG_RAGE_DEF;
}
bool TNode::isRageDef() {
    return (flags & FLAG_RAGE_DEF) != 0;
}

// Общий метод (если нужно проверить, является ли узел "форсированным" в принципе)
bool TNode::isRageAny() {
    return (flags & (FLAG_RAGE_ATTACK | FLAG_RAGE_DEF)) != 0;
}
