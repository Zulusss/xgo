//---------------------------------------------------------------------------


#pragma hdrstop

#include "Logger.h"
#include <iostream>

//---------------------------------------------------------------------------

#pragma package(smart_init)

Logger::Logger() {
        hitCount = 0;
        missAgeCount = 0;
        missHashCount = 0;
        missIndexCount = 0;
        missExpandCount = 0;
        missNodeCount = 0;

        miss5Count = 0;
        miss4Count = 0;
        miss3Count = 0;
        miss4oCount = 0;
        miss3oCount = 0;

        lastError = NULL;
        expandEven = 0;
        expandOdd = 0;

        parents1Culled1 = 0;
        parents1Culled2 = 0;
        parents2Culled1 = 0;
        parents2Culled2 = 0;
        parents3Culled1 = 0;
        parents3Culled2 = 0;
        parents4Culled1 = 0;
        parents4Culled2 = 0;
        parents5Culled1 = 0;
        parents5Culled2 = 0;

        persister = new Persister();
}

void Logger::hit() {
        ++hitCount;
}

void Logger::missAge() {
        ++missAgeCount;
}

void Logger::missHash() {
        ++missHashCount;
}

void Logger::missIndex() {
        ++missIndexCount;
}

void Logger::miss5() {
        ++miss5Count;
}

void Logger::miss4() {
        ++miss4Count;
}

void Logger::miss3() {
        ++miss3Count;
}

void Logger::miss4o() {
        ++miss4oCount;
}

void Logger::miss3o() {
        ++miss3oCount;
}

void Logger::missExpand(TNode *node) {
        ++missExpandCount;
};
void Logger::missNode(TNode *node) {
        ++missNodeCount;
};
void Logger::missMoves(TNode *node) {
        ++missMovesCount;
};

void Logger::error(const char* message) {
        this->lastError = message;
        log(message);
}

void Logger::log(const char* message) {
        std::cout << message << std::endl;
}

void Logger::log(const char* message, const int i) {
        std::cout << message << i << std::endl;
}

void Logger::expand(int count) {
        if (count%2) {
                ++expandOdd;
        } else {
                ++expandEven;
        }
}

void Logger::printLastError(char *buffer) {

        //sprintf(buffer, "%d ", (expandEven*100 / (1+expandOdd + expandEven)));

        if (lastError == NULL) {
            sprintf(buffer, "Cull %d/%d/%d/%d/%d - %d/%d/%d/%d/%d",
                parents1Culled1, parents2Culled1, parents3Culled1, parents4Culled1, parents5Culled1,
                parents1Culled2, parents2Culled2, parents3Culled2, parents4Culled2, parents5Culled2);
        } else {

            sprintf(buffer,
                lastError != NULL
                        ? lastError : missAgeCount > 0
                        ? "Hash collisions %d"
                        : " ", missAgeCount);
        }
}

void Logger::printMissStats(char *buffer) {
        sprintf(buffer, "Miss %d [%d,%d,%d,%d,%d] %d / %d / %d", missExpandCount,
            miss5Count, miss4oCount, miss4Count, miss3oCount, miss3Count,
            missMovesCount, missNodeCount, missAgeCount);
}

void Logger::cull(TRating ratingOld, TRating max_rating, TNode *node) {

    if (ratingOld > -CULL_RATING2 && max_rating >= CULL_RATING2) {

        if (node->totalChilds >= BIG_PARENT5) {
           ++parents5Culled2;
        } else if (node->totalChilds >= BIG_PARENT4) {
           ++parents4Culled2;
        } else if (node->totalChilds >= BIG_PARENT3) {
           ++parents3Culled2;
        } else if (node->totalChilds >= BIG_PARENT2) {
           ++parents2Culled2;
        } else  {
           ++parents1Culled2;
        }
        node->setFixedRating(true);
        persister->save(node);
    } else if (ratingOld > -CULL_RATING1 && max_rating >= CULL_RATING1) {

        if (node->totalChilds >= BIG_PARENT5) {
           ++parents5Culled1;
        } else if (node->totalChilds >= BIG_PARENT4) {
           ++parents4Culled1;
        } else if (node->totalChilds >= BIG_PARENT3) {
           ++parents3Culled1;
        } else if (node->totalChilds >= BIG_PARENT2) {
           ++parents2Culled1;
        } else  {
           ++parents1Culled1;
        }
        node->setFixedRating(true);
    }

}