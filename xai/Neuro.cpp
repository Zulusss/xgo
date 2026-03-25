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
        if (i == current()->move) data[i] = 0.0f;
        else if (kl[i] == 1) data[i] = 1.0f;
        else if (kl[i] == 2) data[i] = -1.0f;
        else data[i] = 0.0f;
    }
    return t;
};

void Neuro::trainNetworkOnCurrentPosition() {
    // 1. Получаем данные о текущем сделанном ходе
    // current() возвращает CursorHistory, где лежит ход и ссылка на узел
    TMove lastMove = current()->move;
    TRating lastRating = current()->node->rating;

    // 2. Подготовка входных данных (состояние поля kl)
    // Важно: в kl уже должен быть результат хода, либо вызовите это ДО хода
    torch::Tensor input = getTensorFromField();

    // 3. Создаем целевой тензор (225 выходов)
    auto targetRatings = torch::zeros({1, 225});
    auto mask = torch::zeros({1, 225}, torch::kBool);

    // Нормализация рейтинга [-32768, 32767] -> [-1.0, 1.0]
    float normRating = (float)((lastRating+32768) / 65536.0f);

    // Указываем нейросети: для этой клетки (lastMove) правильный рейтинг - normRating
    targetRatings[0][(int)lastMove] = normRating;
    mask[0][(int)lastMove] = true;

    // 4. Проход обучения
    optimizer->zero_grad();
    auto output = model->forward(input); // выход нейросети (225 значений tanh)

    // Считаем ошибку только для ОДНОЙ клетки, про которую мы точно знаем рейтинг
    auto loss = torch::mse_loss(output.index({mask}), targetRatings.index({mask}));

    loss.backward();
    optimizer->step();

    // 5. Периодическое сохранение
    static int iter = 0;
    if (++iter % 30 == 0) {
        try {
            torch::save(model, "gomoku_model.pt");
            std::cout << "[AI] Модель сохранена. ХэшХ " << current()->node->hashCodeX
                      << " ХэшO " << current()->node->hashCodeO
                      << " Ход: " << (int)lastMove
                      << " | Рейтинг: " << normRating
                      << " | Loss: " << loss.item<float>() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[AI] Ошибка сохранения: " << e.what() << std::endl;
        }
    }
};


int Neuro::moveNeuro() {
    TMove move = predictBestMove();
    std::cout << "[AI] best move = " << (int)move << std::endl;
    put(move);
    //TNode* n = getChild(current()->node, move);
    //if (n == NULL) return 0;
    int rat = current()->node->rating;
    std::cout << "[AI] best move node rating = " << rat << std::endl;
    //forward(move, n);
    return rat;
};

TMove Neuro::predictBestMove() {
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
