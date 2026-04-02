//---------------------------------------------------------------------------
#include <iostream>


#pragma hdrstop

#include "Expander.h"

void MovesBucket::add(TMove m) {
    if (count < MAX_RELATIVES) {
        move[count++] = m;
    }
}

bool MovesBucket::contains(TMove m) {
    for (int i = 0; i < count; ++i) {
        if (move[i] == m) return true;
    }
    return false;
}
//---------------------------------------------------------------------------

#pragma package(smart_init)

Expander::Expander(SimplyNumbers *simplyGen, Hashtable *movesHash)
        : Relator (simplyGen, movesHash) {
  max_count = 0;
  cnt = 0;
}

//==================================================================
void Expander::fullExpand(TNode* cursor) {
  // 1. Проверка условий выхода (Rage logic)
  if (cursor->isRageAttack() && cursor->rating < -1000) {
  }
  else {
    if (!cursor->isRageAny() || cursor->totalChilds < cursor->totalDirectChilds * 60000) return;
  }

  cursor->setRageAttack(false);
  cursor->setRageDef(false);

  // 2. Поиск ходов (заполняет newChilds и otherNewChilds)
  findMovesToExpand(1);

  // 3. Выбор источника ходов
  MovesBucket* targetBucket = &newChilds;
  if (newChilds.count == 0) {
    if (otherNewChilds.count == 0) return; // Оба пусты — выходим
    targetBucket = &otherNewChilds;
  }

  TRating max_rating = -32600;
  int created = 0;

  // 4. Основной цикл создания узлов
  for(int i = 0; i < targetBucket->count; ++i) {
    TMove move = targetBucket->move[i];

    // ВАЖНО: В логе Hash A/B меняются местами каждый ход
    THash newHashX = cursor->hashCodeO;
    THash newHashO = cursor->hashCodeX * simplyGen->getHash(move);

    bool isCreated;
    TNode *node = movesHash->getOrCreate(newHashX, newHashO, cursor->age + 1, isCreated);

    if (isCreated) {
      ++created;
      rate(cursor, node, move);
      // Увеличиваем счетчики родителя
      ++cursor->totalChilds;
      ++cursor->totalDirectChilds;
    }

    if (node->rating > max_rating) max_rating = node->rating;
  }

  // 5. Обновление рейтинга текущего узла (Minimax)
  short int oldRating = cursor->rating;
  if (!cursor->isFixedRating()) {
    // Инвертируем рейтинг дочернего узла для текущего игрока
    cursor->rating = (TRating)-max_rating;
  }

  // Если ничего не создано и рейтинг не изменился — выходим
  if (created == 0 && oldRating == cursor->rating) return;

  // 6. Проброс изменений вниз по дереву
  updateParents(created);
}

void Expander::multiExpand(TNode* startCursor) {

    int forwardCount = 0;
    int totalAdded = 0;

    int addedPerStep[MULTI_EXPAND_TIMES]; // 🔥 добавили

    TNode* cursor = startCursor;

    // ====== ▶️ ФАЗА ВПЕРЁД ======
    for (int i = 0; i < MULTI_EXPAND_TIMES; ++i) {
        TMove move = expand(0, cursor);
        addedPerStep[forwardCount] = lastCreated;
        totalAdded += lastCreated;
        if (move == 255 || !forward(move)) {
            break;
        }
        cursor = current()->node;
        ++forwardCount;
        if (std::abs(cursor->rating) > 8000)
            break;
    }

    // ====== ◀️ ФАЗА НАЗАД ======
    for (int i = 0; i < forwardCount; ++i) {
        CursorHistory* childH = current();
        TNode* child = childH->node;
        back();
        CursorHistory* parentH = current();
        TNode* parent = parentH->node;
        int stepIndex = forwardCount - 1 - i;
        int added = addedPerStep[stepIndex];
        updateNode(parent, true, added);
    }

    // ====== финальный DAG-проброс ======
    if (totalAdded > 0) {
        updateParents(totalAdded);
    }
}

//adds childs to cursor node
TMove Expander::expand(int startPass, TNode* cursor) {

  if (cursor->rating > 32500 || cursor->rating < -32500) {
      cursor->totalChilds += 100;
      updateParents(100);
      logger->missExpand(cursor);
      return 255;
  }
//  if (logger != NULL) {
//        logger->expand(count);
//  }
  findMovesToExpand(startPass);

  // 3. Выбор источника ходов
  MovesBucket* targetBucket = &newChilds;
  if (newChilds.count == 0) {
    if (otherNewChilds.count == 0) {
        logger->missMoves(cursor);
        return 255; // Оба пусты — выходим
    }
    targetBucket = &otherNewChilds;
  }

  TRating max_rating = -32600;
  TMove chosen = 0;

  int created = 0;
  cursor->totalChilds = 0;
  for(int i=0; i<targetBucket->count; ++i) {

    TMove move = targetBucket->move[i];

    THash newHashX = cursor->hashCodeO;
    THash newHashO = cursor->hashCodeX * simplyGen->getHash(move);

    bool isCreated;
    TNode *node = movesHash->getOrCreate(newHashX, newHashO, cursor->age + 1, isCreated);

    if (isCreated) {
        ++created;
        rate(cursor, node, move);
        ++cursor->totalChilds;

    } else {
        if (startPass == 0) {
            cursor->totalChilds += (node->totalChilds+1);
        }
    }

    if (node->rating > max_rating) {
        max_rating = node->rating;
        chosen = move;
    }
  }

  short int oldRating = cursor->rating;
  if (!cursor->isFixedRating()) {
        cursor->rating = (TRating)-max_rating; //(TRating)(0.4*(double)oldRating-0.6*(double)max_rating);
  }

  cursor->totalDirectChilds = targetBucket->count;

  lastCreated = created;
// TODO:
//  updatedParentsCounter = 0;
//  updateParents(cursor, cursor->totalChilds, oldRating);
//  current()->

  if (count > max_count) max_count = count;

  return chosen;
};

//----------------------------------------------------------------------------

void Expander::findMovesToExpand(int startPass) {
    newChilds.count = 0;
    otherNewChilds.count = 0;
    bool mode1 = (gameMode == 1 && count == 2);
    CursorHistory *h = current();
    TNode* curr = h->node;

#ifdef STORE_ATTACKS
    if (startPass == 0 && curr->attacks[MAX_ATTACK_2-1].r == 0) {//use attacks only if they are not overflown

        bool forceAttack = false, forceDefense = false;

        if (curr->x4 > 0) {
            forceAttack = true;//build 5
        } else if (curr->o4 > 0) {
            forceDefense = true;//close 4
        } else if (curr->o3 > 0) {
            forceDefense = true;//close 3
            if (curr->x3 > 0) {
                forceAttack = true;//try build 4
            }
        } else {
            if (curr->x3 > 0 || curr->x2 > 0) {
                forceAttack = true;//build 3 or 4
            }
            if (curr->o2 > 0) {
                forceDefense = true;//close 2
            }
        }

        if (forceAttack || forceDefense) {
            int startIdx = forceDefense ? 0 : curr->ownAttacks;
            int endIdx = forceAttack ?  MAX_ATTACK_2 : curr->ownAttacks;

            int i;
            for (i = startIdx; i < endIdx; ++i) {
                TAttack &atk = curr->attacks[i];
                if (atk.l == 0 && atk.r == 0) break;

                // 1. Вычисляем вектор направления между l и r
                int x1 = atk.l % fsize, y1 = atk.l / fsize;
                int x2 = atk.r % fsize, y2 = atk.r / fsize;

                int dx = (x2 > x1) ? 1 : (x2 < x1 ? -1 : 0);
                int dy = (y2 > y1) ? 1 : (y2 < y1 ? -1 : 0);

                // 2. Проходим от l до r включительно
                int curX = x1, curY = y1;
                while (true) {
                    TMove m = (TMove)(curX + curY * fsize);

                    // Проверка: клетка должна быть пустой (kl[m] & 3 == 0) и разрешенной
                    if ((kl[m] & 12) == 0) {
                        if ((kl[m] == 1 && mode1 ? isPerspectiveChildMode1(m) : isPerspectiveChild(m)) && isAlllowed(m)) {
                            addChildNoDupe(curr, m);

                        }
                    }

                    if (curX == x2 && curY == y2) break;//завершили интервал от l до r
                    curX += dx; curY += dy;
                    if (curX < 0 || curX >= fsize || curY < 0 || curY >= fsize) break; // На всякий случай
                }
            }

//            if (curr->hashCodeX==32349 && curr->hashCodeO==20713) {
//                printHistory("New Childs ", curr);
//                std::cout << "\n newChilds.count " << newChilds.count << " moves "
//                    << (int)newChilds.move[0] << ","
//                    << (int)newChilds.move[1] << ","
//                    << (int)newChilds.move[2] << ","
//                    << (int)newChilds.move[3]
//                    << "\n";
//            }

            if (newChilds.count > 0) {
                if (forceAttack) curr->setRageAttack(true);
                if (forceDefense) curr->setRageDef(true);
                return;
            }

            // Логирование промахов...   printed as   miss5Count, miss4oCount, miss4Count, miss3oCount, miss3Count,
            if (curr->x4 > 0)  {
                logger->miss5();
                if (cnt < 10) {
                    ++cnt;
                    char msg7[200];
                    curr->printPosition(msg7, 200);
                    std::cout << "Miss5: " << msg7 << " from: " << startIdx << " to: " << i << " \n";
                    printHistory();
                }
            }
            else if (curr->o4 > 0) logger->miss4o();
            else if (curr->x3 > 0) logger->miss4();
            else if (curr->o3 > 0) logger->miss3o();
            else if (curr->x2 > 0 && (curr->totalDirectChilds == 0 || curr->rating > 2400)) logger->miss3();
        }
    }
#endif // -----  end of #ifdef STORE_ATTACKS

    // 2. Общий сбор ходов (выполняется если атак нет)
    for (TMove i = 0; i < TOTAL_CELLS; ++i) {
        if ((kl[i] == 1 && mode1 ? isPerspectiveChildMode1(i) : isPerspectiveChild(i)) && isAlllowed(i)) {
            addChild(curr, i);
        }
    }
}

void Expander::addChildNoDupe(TNode* parent, TMove m) {
    if (isExpected(parent, m)) {
        if (!newChilds.contains(m)) {
            newChilds.add(m);
        }
    } else {
        if (!otherNewChilds.contains(m)) {
            otherNewChilds.add(m);
        }
    }
}

void Expander::addChild(TNode* parent, TMove m) {
    if (isExpected(parent, m)) {
        newChilds.add(m);
    } else {
        otherNewChilds.add(m);
    }
}

bool Expander::isExpected(TNode* curr, TMove i) {
    int t = 15;
    if (curr->x4 > 0) {
        if (scanlines(0, t, i) <= 0) {
            //filter out nodes which not allows to build 5
            return false;
        }
    } else if (curr->o4 > 0) {
        if (scanlines(1, t, i) <= 0) {
            //filter out nodes which not allows to close 4
            return false;
        }
    } else if (curr->x3 > 0) {
        if (scanlines(2, t, i) <= 0) {
            //filter out nodes which not allows to build opened 4
            return false;
        }
    } else if (curr->o3 > 0) {
        if (scanlines(3, t, i) <= 0 && scanlines(4, t, i) <= 0) {
            //filter out nodes which allows neither close 3 nor build closed  4
            return false;
        }
    }
    else if (curr->x2 > 0 && curr->rating < 2400) {
        if (scanlines(4, t, i) <= 0 && scanlines(5, t, i) <= 0) {
            //filter out nodes which not allows to build 3 or 4,
            //only if parent rating is not too high,
            //otherwise will need to search for defence nodes
            return false;
        }
    }
    return true;
}