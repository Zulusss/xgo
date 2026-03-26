//---------------------------------------------------------------------------
#ifndef NeuroH
#define NeuroH
//---------------------------------------------------------------------------
#include "GameBoard.h"
#include <torch/torch.h>

#include <memory> // для shared_ptr

// 1. Предварительное объявление (чтобы не тянуть тяжелый torch.h в хедер)
struct GomokuNet;
namespace at { class Tensor; }
namespace torch { namespace optim { class Adam; } } // для оптимизатора

class Neuro : public GameBoard {
public:
    Neuro(SimplyNumbers *simplyGen, Hashtable *movesHash, int gameMode);
    void trainNetworkOnCurrentPosition();
    void trainNetworkOnSingleMove(TMove move, TRating rating);
    int moveNeuro();

protected:
    // 2. Указатели на нейросеть и оптимизатор
    std::shared_ptr<GomokuNet> model;
    std::unique_ptr<torch::optim::Adam> optimizer;
    torch::Tensor getTensorFromField();
    TMove predictBestMove();

private:
    TRating getNNRating(TMove move);

};

#endif
