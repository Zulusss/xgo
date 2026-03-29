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
        // Увеличиваем фильтры до 64 и добавляем BatchNorm
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 3).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));

        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));

        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(128));

        // Линейный слой без tanh на конце
        fc = register_module("fc", torch::nn::Linear(128 * 15 * 15, 225));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(bn1->forward(conv1->forward(x)));
        x = torch::relu(bn2->forward(conv2->forward(x)));
        x = torch::relu(bn3->forward(conv3->forward(x)));
        x = x.view({x.size(0), -1});
        return fc->forward(x); // Убрали tanh, MSE лучше работает с сырыми числами
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
    torch::nn::Linear fc{nullptr};
};

Neuro::Neuro(SimplyNumbers* s, Hashtable* h, int gameMode)
        : GameBoard(s, h, gameMode){

    lossTracker = new LossTracker(3000);

    trainedFieldCount = 0;
    trainedSingleCount = 0;
    skipTrainFieldCount = 0;

    // 1. Отключаем использование NNPACK для всех операций
    at::set_num_threads(1); // Для 15x15 одного потока за глаза, это уберет лишние проверки CPU

    // 2. Глобальный флаг отключения оптимизаций, вызывающих NNPACK
    // (актуально для сверточных сетей на CPU)
    torch::set_num_threads(1);

    model = std::make_shared<GomokuNet>();
    optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(1e-4));

    // Пытаемся загрузить существующую модель
    std::ifstream file("gomoku_model.pt");
    if (file.good()) {
        try {
            torch::load(model, "gomoku_model.pt");

            bool opt = std::ifstream("gomoku_optimizer.pt").good();
            // Опционально: загрузка оптимизатора, если сохраняли его
            if (opt) {
                torch::load(*optimizer, "gomoku_optimizer.pt");
            }
            std::cout << "[AI] Загружена обученная модель из файла. "
                << (opt ? " +optimizer " : "") << std::endl;
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
    if (++iter % 200 == 0 || count < 4) {
        std::cout << "[AI] Полевое обучение: Ход: " << (int)count
                  << " | Обучено клеток: " << knownNodesCount
                  << " / " << (int)par->totalDirectChilds
                  << " | Loss: " << loss.item<float>()
                  << " | Avg.Loss: " << lossTracker->toString()
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

    TNode *node = current()->node;
    // 1. Применяем маску занятых клеток
    for (int i = 0; i < 225; ++i) {
        if (kl[i] > 1) {
            output[i] = -2.0f; // Занято -> худший возможный рейтинг
        } else if (kl[i] == 0) {
            output[i] = -1.1f;// слишком далеко от других камней
        } else if (!isExpected(node, i)) {
            output[i] = -1.0f;
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
