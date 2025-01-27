//---------------------------------------------------------------------------

#ifndef EvaluatorH
#define EvaluatorH

#include "SimplyNumbers.h"
#include "Hashtable.h"
#include "Cursor.h"
//---------------------------------------------------------------------------

class Evaluator : public Cursor {

        private:
        inline bool comp(int x, int y, unsigned char c);
        bool  scanlines(int BlNo, int &lines, int N);
        
        public:

        Evaluator(SimplyNumbers* sn, Hashtable* ht);
        void rate(TNode *src, TNode *destNode, TMove move); //fills {totalRating,x3,x4,o3,o4} of dest;
};
#endif
