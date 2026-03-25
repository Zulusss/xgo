//---------------------------------------------------------------------------
#ifndef BuilderH
#define BuilderH
//---------------------------------------------------------------------------
#include <torch/torch.h>
#include "Expander.h"

#include <memory> // для shared_ptr

// 1. Предварительное объявление (чтобы не тянуть тяжелый torch.h в хедер)
struct GomokuNet;
namespace at { class Tensor; }
namespace torch { namespace optim { class Adam; } } // для оптимизатора

class Builder : public Expander {
public:
    Builder(SimplyNumbers*, Hashtable*, int gameMode );
    void buildTree();
    void trainNetworkOnCurrentPosition();

protected:
    // 2. Указатели на нейросеть и оптимизатор
    std::shared_ptr<GomokuNet> model;
    std::unique_ptr<torch::optim::Adam> optimizer;
    torch::Tensor getTensorFromField();

private:
    int chooseNodeToExpand();
};

#endif
