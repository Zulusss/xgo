//---------------------------------------------------------------------------
#include <iostream>
#include <fstream>

#pragma hdrstop

#include "Builder.h"

//---------------------------------------------------------------------------

#pragma package(smart_init)


// Архитектура сети
struct GomokuNet : torch::nn::Module {
    GomokuNet() {
        // Слой 1: Вход 1 канал (ваше поле), выход 32 фильтра. Окно 3x3.
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).padding(1)));
        // Слой 2: Ищем более сложные паттерны (вилки)
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
        // Слой 3: Финальный сбор признаков
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
        // Выход: 225 нейронов (по одному на каждую клетку поля 15x15)
        fc = register_module("fc", torch::nn::Linear(64 * 15 * 15, 225));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = torch::relu(conv3->forward(x));
        x = x.view({x.size(0), -1});

        // Используем tanh, чтобы выходы сети лежали в диапазоне [-1, 1]
        return torch::tanh(fc->forward(x));
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Linear fc{nullptr};
};

Builder::Builder(SimplyNumbers* s, Hashtable* h, int gameMode)
        : Expander(s, h){

    // 1. Отключаем использование NNPACK для всех операций
    at::set_num_threads(1); // Для 15x15 одного потока за глаза, это уберет лишние проверки CPU

    // 2. Глобальный флаг отключения оптимизаций, вызывающих NNPACK
    // (актуально для сверточных сетей на CPU)
    torch::set_num_threads(1);

    this->gameMode = gameMode;

    model = std::make_shared<GomokuNet>();
    optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(1e-3));

    // Создаем тензор 3x3 (как маленькое поле 5-в-ряд)
    //torch::Tensor tensor = torch::eye(3);
    //std::cout << "LibTorch говорит привет! Матрица 3x3:\n" << tensor << std::endl;

    // Пытаемся загрузить существующую модель
    std::ifstream file("gomoku_model.pt");
    if (file.good()) {
        try {
            torch::load(model, "gomoku_model.pt");
            std::cout << "[AI] Загружена обученная модель из файла." << std::endl;
        } catch (...) {
            std::cout << "[AI] Файл модели поврежден, начинаем обучение с нуля." << std::endl;
        }
    } else {
        std::cout << "[AI] Файл модели не найден, создана новая сеть." << std::endl;
    }

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
        printHistory("builder fallback.", cur->node);
        cur->node->totalChilds += 10;
        back();
        cur = current();
        continue;
    }
    forward(childs.move[i], childs.node[i]);
    cur = current();
  }

  expand(0, cur->node);

  while(count>count0) back();

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
    if (childs.count == 0) return -1;

    // Обучаем, если позиция достаточно изучена
    if (current()->node->totalChilds > 10000) {
        trainNetworkOnCurrentPosition();

        // Статический счетчик для сохранения
        static int saveCounter = 0;
        saveCounter++;

         //std::cout << saveCounter  << std::endl;
        // Сохраняем каждые 500 шагов обучения
        if (saveCounter % 500 == 0) {
            try {
                torch::save(model, "gomoku_model.pt");
                std::cout << "[AI] Модель успешно сохранена." << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[AI] Ошибка сохранения: " << e.what() << std::endl;
            }
        }
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
}

//==================================================================

// Реализация метода подготовки данных (теперь он видит тип torch::Tensor)
torch::Tensor Builder::getTensorFromField() {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto t = torch::zeros({1, 1, 15, 15}, options);
    float* data = t.data_ptr<float>();

    for (int i = 0; i < 225; ++i) {
        if (kl[i] == 1) data[i] = 1.0f;
        else if (kl[i] == 2) data[i] = -1.0f;
        else data[i] = 0.0f;
    }
    return t;
}

void Builder::trainNetworkOnCurrentPosition() {
    torch::Tensor input = getTensorFromField();

    // Целевые значения рейтингов для всех клеток (по умолчанию 0.0)
    auto targetRatings = torch::zeros({1, 225});

    // Маска: обучаем только те клетки, которые реально обсчитал алгоритм
    auto mask = torch::zeros({1, 225}, torch::kBool);

    for (int i = 0; i < childs.count; ++i) {
        TNode *node = childs.node[i];
        TMove move = childs.move[i];

        // Нормализация рейтинга [-32k, +32k] -> [-1.0, 1.0]
        float normRating = (float)node->rating / 32768.0f;

        targetRatings[0][(int)move] = normRating;
        mask[0][(int)move] = true;
    }

    optimizer->zero_grad();
    auto output = model->forward(input);

    // Считаем ошибку (MSE) только для тех ходов, которые были в childs
    auto loss = torch::mse_loss(output.index({mask}), targetRatings.index({mask}));

    loss.backward();
    optimizer->step();

    // Периодическое сохранение
    static int iter = 0;
    if (++iter % 1000 == 0) {
        torch::save(model, "gomoku_model.pt");
    }
}

int Builder::moveNeuro() {
    TMove move = predictBestMove();
    TNode* n = getChild(current()->node, move);
    forward(move, n);
    return n->rating;
}

TMove Builder::predictBestMove() {
    // 1. Включаем режим оценки (отключает градиенты и дропаут для скорости)
    torch::NoGradGuard no_grad;
    model->eval();

    // 2. Превращаем текущее поле kl в тензор
    torch::Tensor input = getTensorFromField();

    // 3. Прогоняем через сеть
    torch::Tensor output = model->forward(input);

    // 4. Находим индекс самого вероятного хода
    // argmax вернет индекс нейрона с максимальным значением
    int64_t bestMoveIdx = output.argmax(1).item<int64_t>();

    // Возвращаем режим обучения обратно
    model->train();

    return (TMove)bestMoveIdx;
}

TRating Builder::getNNRating(TMove move) {
    torch::NoGradGuard no_grad;
    model->eval();

    auto output = model->forward(getTensorFromField());
    float normVal = output[0][(int)move].item<float>();

    model->train();
    // Возвращаем в ваш масштаб
    return (TRating)(normVal * 32768.0f);
}
