//---------------------------------------------------------------------------
#ifndef NeuroH
#define NeuroH
//---------------------------------------------------------------------------
#include "GameBoard.h"
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <torch/torch.h>

#include <memory> // для shared_ptr

// 1. Предварительное объявление (чтобы не тянуть тяжелый torch.h в хедер)
struct GomokuNet;
namespace at { class Tensor; }
namespace torch { namespace optim { class Adam; } } // для оптимизатора

class LossTracker {

private:
    float movingAverage = 0.0f;
    int count = 0; // Счётчик вызовов
    const int threshold;

public:
    explicit LossTracker(int t = 10) : threshold(t > 0 ? t : 1) {}

    void addLoss(float loss) {
        count++;

        // Определяем вес нового значения (альфа)
        // Если count <= 10, это будет 1/1, 1/2 ... 1/10
        // Если count > 10, фиксируем на 1/10 (0.1f)
        float alpha = (count <= threshold) ? (1.0f / count) : (1.0f / threshold);

        if (count == 1) {
            movingAverage = loss;
        } else {
            // Формула скользящего среднего
            movingAverage = (loss * alpha) + (movingAverage * (1.0f - alpha));
        }
    }

std::string toString() const {
    std::stringstream ss;

    if (movingAverage == 0.0f) {
        ss << "0";
    } else {
        // Находим позицию первой значащей цифры после запятой
        // Если avg = 0.00123, то -log10(0.00123) ≈ 2.9, ceil ≈ 3.
        // Прибавляем 1, чтобы получить 2 значащие цифры (3+1=4 знака после запятой).
        int firstDigitPos = static_cast<int>(std::ceil(-std::log10(std::abs(movingAverage))));
        if (firstDigitPos < 0) firstDigitPos = 0; // Если число > 1
        int precision = firstDigitPos + 1; // 1 доп. знак после первого значащего
        ss << std::fixed << std::setprecision(precision) << movingAverage;
    }
    return ss.str();
}

std::string toLossString() const {
    std::stringstream ss;

    if (movingAverage == 0.0f) {
        ss << "0, loss_cnt=" << count;
    } else {
        // Находим позицию первой значащей цифры после запятой
        // Если avg = 0.00123, то -log10(0.00123) ≈ 2.9, ceil ≈ 3.
        // Прибавляем 1, чтобы получить 2 значащие цифры (3+1=4 знака после запятой).
        int firstDigitPos = static_cast<int>(std::ceil(-std::log10(std::abs(movingAverage))));
        if (firstDigitPos < 0) firstDigitPos = 0; // Если число > 1

        int precision = firstDigitPos + 1; // 1 доп. знак после первого значащего

        ss << std::fixed << std::setprecision(precision) << movingAverage
           << ", loss_cnt=" << count;
    }
    return ss.str();
}

};

struct TrainSample {
    torch::Tensor state;
    int move;
    float result;// победа/поражение (0/1)
    bool isXTurn;
    TNode* node;
};

class Neuro : public GameBoard {
public:
    Neuro(SimplyNumbers *simplyGen, Hashtable *movesHash, int gameMode);
//    void trainNetworkOnCurrentPosition();
//    void trainNetworkOnSingleMove(TMove move, TRating rating);
    int moveNeuro();

protected:

    int trainedFieldCount;
    int trainedSingleCount;
    int skipTrainFieldCount;

    LossTracker *trackerNX, *trackerNO, *trackerLX, *trackerLO;

    void trainFromGame(bool lastPlayerWon);
    void trainSample(const TrainSample& s);

private:
    LossTracker *lossTracker;

    //Указатели на нейросеть и её оптимизатор
    std::shared_ptr<GomokuNet> model;
    std::unique_ptr<torch::optim::Adam> optimizer;
    torch::Tensor getTensorFromField();

    void save(float loss);
    void addSample();
    void updateSample(TMove move);
    TMove predictBestMove();
    TRating getNNRating(TMove move);
    inline float decodeRating(int rating);

};

#endif
