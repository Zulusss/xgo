//---------------------------------------------------------------------------
#include <iostream>
#include <fstream>

#pragma hdrstop

#include "Neuro.h"

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

Neuro::Neuro(SimplyNumbers* s, Hashtable* h, int gameMode)
        : GameBoard(s, h, gameMode){

    trainedFieldCount = 0;
    trainedSingleCount = 0;
    skipTrainFieldCount = 0;

    // 1. Отключаем использование NNPACK для всех операций
    at::set_num_threads(1); // Для 15x15 одного потока за глаза, это уберет лишние проверки CPU

    // 2. Глобальный флаг отключения оптимизаций, вызывающих NNPACK
    // (актуально для сверточных сетей на CPU)
    torch::set_num_threads(1);

    model = std::make_shared<GomokuNet>();
    optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(1e-3));

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

// Реализация метода подготовки данных (теперь он видит тип torch::Tensor)
torch::Tensor Neuro::getTensorFromField() {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto t = torch::zeros({1, 1, 15, 15}, options);
    float* data = t.data_ptr<float>();
    TMove cm = current()->move;

    for (int i = 0; i < 225; ++i) {
        if (kl[i] == 1) data[i] = 1.0f;
        else if (kl[i] == 2) data[i] = -1.0f;
        else data[i] = 0.0f;
    }
    return t;
};

void Neuro::trainNetworkOnCurrentPosition() {
    torch::Tensor input = getTensorFromField();
    auto targetRatings = torch::zeros({1, 225});
    auto mask = torch::full({1, 225}, true, torch::kBool); // Обучаем все 225 клеток

    static TNode *prev = NULL;
    TNode *par = current()->node;
    if (prev == par) {
        ++skipTrainFieldCount;
        return;
    }
    prev = par;

    // 1. Проходим по всем клеткам поля
    for (int i = 0; i < 225; ++i) {
        if (kl[i] > 1) {
            // Клетка занята: жесткий штраф
            targetRatings[0][i] = -1.0f;
        }
        else {
            // Клетка пуста: проверяем, есть ли она в нашей базе знаний (TNode)
            TNode* node = getChild(par, i);

            if (node) {
                // Алгоритм уже знает эту позицию! Берем реальный рейтинг
                targetRatings[0][i] = (float)(node->rating / 32768.0f);
            } else {
                // Позиция не исследована
                targetRatings[0][i] = -0.5f;
            }
        }
    }

    // 2. Стандартный цикл обучения LibTorch
    optimizer->zero_grad();
    auto output = model->forward(input);
    auto loss = torch::mse_loss(output, targetRatings);
    loss.backward();
    optimizer->step();

    // лог
    static int iter = 0;
    if (++iter % 50 == 0) {
        TNode *n = current()->node;
        std::cout << "[AI] Полевое обучение: Ходов " << (int)count
          << " ХэшХ " << n->hashCodeX
          << " ХэшO " << n->hashCodeO
          << std::endl;
    }
    // 5. Периодическое сохранение
    save(loss);

    ++trainedFieldCount;
}

void Neuro::save(torch::Tensor loss) {

    static int iter = 0;
    if (++iter % 500 == 0) {
        try {
            torch::save(model, "gomoku_model.pt");
            TNode *n = current()->node;
            if (++iter % 2000 == 0) {
                std::cout << "[AI] Модель сохранена. Ходов " << count
                  << " ХэшХ " << n->hashCodeX
                  << " ХэшO " << n->hashCodeO
                  << " Rating " << n->rating
                  //<< " Ход: " << (int)lastMove
                  //<< " | Рейтинг: " << normRating
                  << " | Loss: " << loss.item<float>() << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[AI] Ошибка сохранения: " << e.what() << std::endl;
        }
    }
};

void Neuro::trainNetworkOnSingleMove(TMove move, TRating rating) {

    static int iter = 0;
    if (++iter % 5 != 0) return;

    torch::Tensor input = getTensorFromField();

    // 1. Получаем текущее предсказание сети (чтобы не менять остальные 224 клетки)
    model->eval(); // Временно выключаем дропаут
    torch::Tensor currentOutput;
    {
        torch::NoGradGuard no_grad;
        currentOutput = model->forward(input).clone();
    }
    model->train();

    // 2. Создаем Target: оставляем всё как есть, кроме целевого хода
    auto target = currentOutput;
    float normRating = (float)rating / 32768.0f;
    target[0][(int)move] = normRating;

    // 3. Шаг обучения
    optimizer->zero_grad();
    auto output = model->forward(input);

    // Считаем ошибку MSE
    auto loss = torch::mse_loss(output, target);
    loss.backward();
    optimizer->step();

    // лог
    ++trainedSingleCount;
    if (trainedSingleCount % 2000 == 0) {
        std::cout << "[AI] Точечное обучение: Ходов " << (int)count
                  << " | Новый рейтинг: " << rating
                  << " | Loss: " << loss.item<float>() << std::endl;
    }

    save(loss);

}


int Neuro::moveNeuro() {
    TMove move = predictBestMove();
    TNode* node = current()->node;
    do {
        if (kl[move] > 1) {
            std::cout << "cell is already occupied " << move << std::endl;
            return 0;
        }

        TNode* child = getChild(node, move);

        if (child != NULL && child->rating < 6200) {
            trainNetworkOnSingleMove(move, child->rating);
            TMove move1 = predictBestMove();
            if (move1 != move) {
                std::cout << "predicted cell changed " << move << " -> " << move1 << std::endl;
                move = move1;
                continue;
            }
        }
        break;
    } while(true);

    put(move);
    int rat = node->rating;
    //std::cout << "[AI] best move " << (int)move << " node rating = " << rat << std::endl;
    return rat;
};

TMove Neuro::predictBestMove() {
    torch::NoGradGuard no_grad;
    model->eval();

    torch::Tensor input = getTensorFromField();
    // Убираем размерность батча, делаем плоский вектор 225
    torch::Tensor output = model->forward(input).view({-1});

    // 1. Применяем маску занятых клеток
    for (int i = 0; i < 225; ++i) {
        if (kl[i] > 1) {
            output[i] = -2.0f; // Занято -> худший возможный рейтинг
        }
    }

    // 2. Выбираем лучший из ОСТАВШИХСЯ
    int64_t bestMoveIdx = output.argmax(0).item<int64_t>();

    model->train();
    return (TMove)bestMoveIdx;
};

TRating Neuro::getNNRating(TMove move) {
    torch::NoGradGuard no_grad;
    model->eval();

    auto output = model->forward(getTensorFromField());
    float normVal = output[0][(int)move].item<float>();

    model->train();
    // Возвращаем в ваш масштаб
    return (TRating)(normVal * 32768.0f);
};
