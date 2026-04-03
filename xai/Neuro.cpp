//---------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <thread>

#pragma hdrstop

#include "Neuro.h"

//---------------------------------------------------------------------------

#pragma package(smart_init)


// ==========================================================
// Новая архитектура сети (БЕЗ огромного Linear слоя)
// ==========================================================
struct GomokuNet : torch::nn::Module {

    GomokuNet() {
        // Вход: 3 канала (мои, чужие, чей ход)

        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));

        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));

        conv3 = register_module("conv3",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(128));

        // 🔥 ВАЖНО: заменяем Linear на 1x1 свертку
        // Это даёт 1 канал (карта оценок)
        conv_out = register_module("conv_out",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 1, 1)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::tanh(bn1->forward(conv1->forward(x)));
        x = torch::tanh(bn2->forward(conv2->forward(x)));
        x = torch::tanh(bn3->forward(conv3->forward(x)));

        x = conv_out->forward(x); // [B,1,15,15]

        // разворачиваем в [B,225]
        x = x.view({x.size(0), -1});
        return x;
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv_out{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
};


// ==========================================================
// Конструктор
// ==========================================================
Neuro::Neuro(SimplyNumbers* s, Hashtable* h, int gameMode)
        : GameBoard(s, h, gameMode){

#ifdef __linux__
    setenv("NNPACK_MODE", "0", 1);
#endif

    lossTracker = new LossTracker(3000);

    trackerNX = new LossTracker(30);
    trackerNO = new LossTracker(30);
    trackerLX = new LossTracker(30);
    trackerLO = new LossTracker(30);

    trainedFieldCount = 0;
    trainedSingleCount = 0;
    skipTrainFieldCount = 0;

    int threads = 1;//std::thread::hardware_concurrency();
    //if (threads == 0) threads = 4;

    at::set_num_threads(threads);
    torch::set_num_threads(threads);

    model = std::make_shared<GomokuNet>();
    optimizer = std::make_unique<torch::optim::Adam>(
        model->parameters(),
        torch::optim::AdamOptions(1e-4)
    );

    // загрузка модели
    std::ifstream file("gomoku_model.pt");
    if (file.good()) {
        try {
            torch::load(model, "gomoku_model.pt");

            if (std::ifstream("gomoku_optimizer.pt").good()) {
                torch::load(*optimizer, "gomoku_optimizer.pt");
            }

            std::cout << "[AI] Модель загружена" << std::endl;
        } catch (...) {
            std::cout << "[AI] Ошибка загрузки, начинаем заново" << std::endl;
        }
    } else {
        std::cout << "[AI] Новая сеть" << std::endl;
    }
}


// ==========================================================
// Новый вход: 3 канала
// ==========================================================
torch::Tensor Neuro::getTensorFromField() {

    auto options = torch::TensorOptions().dtype(torch::kFloat32);

    // [batch, channels, H, W]
    auto t = torch::zeros({1, 3, 15, 15}, options);
    float* data = t.data_ptr<float>();

    bool myTurn = (count % 2 == 0); // крестик ходит

    for (int i = 0; i < 225; ++i) {

        int y = i / 15;
        int x = i % 15;

        int idx_my   = 0 * 225 + i;
        int idx_op   = 1 * 225 + i;
        int idx_turn = 2 * 225 + i;

        // канал 3: чей ход
        data[idx_turn] = myTurn ? 1.0f : 0.0f;

        if (kl[i] == 1) {
            // крестик
            if (myTurn)
                data[idx_my] = 1.0f;
            else
                data[idx_op] = 1.0f;
        }
        else if (kl[i] == 2) {
            // нолик
            if (!myTurn)
                data[idx_my] = 1.0f;
            else
                data[idx_op] = 1.0f;
        }
    }

    return t;
}


// ==========================================================
// Обучение (почти без изменений)
// ==========================================================
void Neuro::trainNetworkOnCurrentPosition() {
    // 1. Проверка на повторное обучение той же позиции
    static TNode* prev[250] = { nullptr };
    TNode *par = current()->node;
    if (count > 3 && prev[count] == par) {
        ++skipTrainFieldCount;
        return;
    }
    prev[count] = par;

    // 2. Подготовка входных данных
    torch::Tensor input = getTensorFromField();

    // Создаем таргеты и маску (все во float32)
    auto targetRatings = torch::zeros({1, 225});
    auto mask = torch::zeros({1, 225});

    int knownNodesCount = 0;

    // 3. Заполнение таргетов и маски
    for (int i = 0; i < 225; ++i) {

        if (kl[i] > 1) {
            // Клетка занята: фиксируем штраф, чтобы сеть не предлагала туда ходить
            targetRatings[0][i] = -1.0f;
            mask[0][i] = 1.0f;
        }
        else {
            TNode* node = getChild(par, i);
            if (node) {
                // Данные из дерева поиска (самые ценные)
                targetRatings[0][i] = (float)(node->rating / 32768.0f);
                mask[0][i] = 1.0f;
                knownNodesCount++;
            }
            // Все остальные клетки игнорируем (mask = 0).
            // Это решает проблему высокого Loss на пустых полях.
        }
    }

    // 4. Шаг обучения
    model->train();
    optimizer->zero_grad();

    auto output = model->forward(input);

    // Считаем MSE вручную только для маскированных элементов
    // (выход - цель)^2 * маска
    auto loss_map = torch::pow(output - targetRatings, 2) * mask;

    // Средний лосс только по активным клеткам (чтобы не делить на 225)
    auto loss = loss_map.sum() / (mask.sum() + 1e-6);

    loss.backward();
    optimizer->step();

    // 5. Логирование
    static int iter = 0;
    if (++iter % 2000 == 0 || count <= 1) {
        std::cout << "[AI] Полевое обучение: Ход: " << (int)count
                  << " | Обучено клеток: " << knownNodesCount
                  << " / " << (int)par->totalDirectChilds
                  << " | Loss: " << loss.item<float>()
                  << " | Avg.Loss: " << lossTracker->toLossString()
                  << std::endl;
    }

    save(loss.item<float>());
    ++trainedFieldCount;
}

void Neuro::save(float loss) {

    lossTracker->addLoss(loss);

    static int iter = 0;
    if (++iter % 500 == 0) {
        try {
            // 1. Сохраняем веса нейросети
            torch::save(model, "gomoku_model.pt");
            // 2. Сохраняем состояние Adam (важно для продолжения обучения)
            torch::save(*optimizer, "gomoku_optimizer.pt");
            TNode *n = current()->node;
            if (++iter % 2000 == 0) {
                std::cout << "[AI] Модель сохранена. Ходов " << count
                  << " ХэшХ " << n->hashCodeX
                  << " ХэшO " << n->hashCodeO
                  << " Rating " << n->rating
                  //<< " Ход: " << (int)lastMove
                  //<< " | Рейтинг: " << normRating
                  << " | Loss: " << loss
                  << " | Avg.Loss: " << lossTracker->toString()
                  << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[AI] Ошибка сохранения: " << e.what() << std::endl;
        }
    }
};

void Neuro::trainNetworkOnSingleMove(TMove move, TRating rating) {
    // 1. Подготовка данных
    torch::Tensor input = getTensorFromField();
    float normRating = (float)rating / 32768.0f;

    // 2. Включаем режим обучения
    model->train();
    optimizer->zero_grad();

    // 3. Прямой проход (forward)
    auto output = model->forward(input);

    // 4. Выделяем предсказание только для нужной клетки (индекса хода)
    // Используем squeeze/unsqueeze для корректности размерностей тензора [1]
    auto predicted = output[0][(int)move].unsqueeze(0);
    auto target = torch::tensor({normRating}, torch::kFloat32).to(output.device());

    // 5. Считаем ошибку ТОЛЬКО по этому ходу
    // Остальные 224 клетки не участвуют в backward(), их градиент будет 0
    auto loss = torch::mse_loss(predicted, target);

    // 6. Обратный проход и шаг оптимизатора
    loss.backward();
    optimizer->step();

    // 7. Логирование и статистика
    ++trainedSingleCount;
    if (trainedSingleCount % 2000 == 0 || count < 4) {
        std::cout << "[AI] Точечное обучение: Ход " << (int)count
                  << " | Индекс: " << (int)move
                  << " | Рейтинг: " << rating
                  << " | Loss: " << loss.item<float>() << std::endl;
    }

    // Сохраняем состояние (опционально, если loss не слишком шумный)
    save(loss.item<float>());
}


int Neuro::moveNeuro() {

    trainNetworkOnCurrentPosition();

    TMove move = predictBestMove();
    TNode* node = current()->node;
    do {
        if (kl[move] > 1) {
            std::cout << "cell is already occupied " << move << std::endl;
            return 0;
        }

        TNode* child = getChild(node, move);

        if (child == NULL || child != NULL && std::abs(node->rating + child->rating) >= 600) {
            if (child == NULL)
                trainNetworkOnCurrentPosition();
            else
                trainNetworkOnSingleMove(move, child->rating);

            TMove move1 = predictBestMove();
            if (move1 != move) {
                std::cout << "predicted cell changed " << (int)move << " -> " << (int)move1 << std::endl;
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

    torch::Tensor output = model->forward(getTensorFromField()).view({-1});

    TNode *node = current()->node;
    // 🔥 Отфильтровываем неликвиды
    for (int i = 0; i < 225; ++i) {
        if (kl[i] > 1) {
            // клетка занята — худший вариант
            output[i] = -2.0f;
        }
        else if (kl[i] == 0) {
            // далеко от игры
            output[i] = -1.1f;
        }
        else if (!isExpected(node, i)) {
            // неинтересный ход
            output[i] = -1.0f;
        }
    }

    int64_t bestMoveIdx = output.argmax(0).item<int64_t>();

    model->train();
    return (TMove)bestMoveIdx;
}

// ==========================================================
// Получение рейтинга
// ==========================================================
TRating Neuro::getNNRating(TMove move) {

    torch::NoGradGuard no_grad;
    model->eval();

    auto output = model->forward(getTensorFromField());

    float normVal = output[0][(int)move].item<float>();

    model->train();

    return (TRating)(normVal * 32768.0f);
}