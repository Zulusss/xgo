//---------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <thread>

#pragma hdrstop

#include "Neuro.h"

//---------------------------------------------------------------------------

#pragma package(smart_init)

// -----------------------------
// GomokuNet (policy head без ReLU)
// -----------------------------
struct GomokuNet : torch::nn::Module {
    GomokuNet() {
        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 64, 3).padding(1)));
        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
        conv3 = register_module("conv3",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)));

        // policy head
        policy_conv = register_module("policy_conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 2, 1)));
        policy_fc = register_module("policy_fc",
            torch::nn::Linear(2 * 15 * 15, 225)); // линейный, без ReLU

        // value head
        value_conv = register_module("value_conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 1, 1)));
        value_fc1 = register_module("value_fc1",
            torch::nn::Linear(15 * 15, 64));
        value_fc2 = register_module("value_fc2",
            torch::nn::Linear(64, 1));
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        // --- общие conv слои ---
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = torch::relu(conv3->forward(x));

        // --- policy head (без ReLU) ---
        auto p = policy_conv->forward(x);
        p = p.view({p.size(0), -1});
        p = policy_fc->forward(p);

        // --- value head ---
        auto v = torch::relu(value_conv->forward(x));
        v = v.view({v.size(0), -1});
        v = torch::relu(value_fc1->forward(v));

        // 🔥 Dropout только во время обучения
        if (this->is_training()) {
            v = torch::dropout(v, /*p=*/0.15, /*train=*/true); // оставляем 85% нейронов
        }

        v = torch::tanh(value_fc2->forward(v)); // [-1, 1]

        return {p, v};
    }
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Conv2d policy_conv{nullptr}, value_conv{nullptr};
    torch::nn::Linear policy_fc{nullptr}, value_fc1{nullptr}, value_fc2{nullptr};
};


std::vector<TrainSample> gameHistory;//Хранилище партии

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
        torch::optim::AdamOptions(3e-4)
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
    auto t = torch::zeros({1, 2, 15, 15}, options);
    float* data = t.data_ptr<float>();

    bool isXTurn = (count % 2 == 0);

    for (int i = 0; i < 225; ++i) {

        int y = i / 15;
        int x = i % 15;

        int idx_my = 0 * 225 + i;
        int idx_op = 1 * 225 + i;
        int cell = kl[i];

        if (cell == 4) { // X
            if (isXTurn)
                data[idx_my] = 1.0f;
            else
                data[idx_op] = 1.0f;
        }
        else if (cell == 8) { // O
            if (!isXTurn)
                data[idx_my] = 1.0f;
            else
                data[idx_op] = 1.0f;
        }

        // пустые игнорируем (0 или 1)
    }

    return t;
}


void Neuro::save(float loss) {

    static int iter = 0;
    lossTracker->addLoss(loss);

    if (++iter % 200 == 0) {
        try {
            torch::save(model, "gomoku_model.pt");
            torch::save(*optimizer, "gomoku_optimizer.pt");
            TNode *n = current()->node;

                std::cout << "[AI] Модель сохранена. Ходов " << count
                  << " ХэшХ " << n->hashCodeX
                  << " ХэшO " << n->hashCodeO
                  << " Rating " << n->rating
                  //<< " Ход: " << (int)lastMove
                  //<< " | Рейтинг: " << normRating
                  << " | Loss: " << loss
                  << " | Avg.Loss: " << lossTracker->toString()
                  << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "[AI] Save error: " << e.what() << std::endl;
        }
    }
}


// -----------------------------
// addSample с временно убранным ходом
// -----------------------------
void Neuro::addSample(TMove move) {
    TrainSample sample;

    // --- временно убрать ход с поля ---
    auto saved = kl[move];
    kl[move] = 1; // считаем клетку пустой

    // --- формируем тензор для сети ---
    sample.state = getTensorFromField(); // move ещё не сделан
    sample.move = move;
    sample.result = 0;
    sample.isXTurn = (count % 2 == 0);
    sample.node = current()->node;

    // --- вернуть ход на место ---
    kl[move] = saved;

    // --- добавить сэмпл в историю ---
    gameHistory.push_back(sample);
}

int Neuro::moveNeuro() {

    static int lastCount = 0;

    //сохраняем в сэмпл предыдущий ход легаси
    if (lastCount < count || count < lastCount && count < 2) {
        addSample(current()->move);
        lastCount = count;
    }

    TMove move = predictBestMove();
    TNode* node = current()->node;

//    static int debugGames = 0;
//    if (count <= 3 && debugGames < 20) {
//        std::cout << "[DEBUG] chosen move: " << (int)move << std::endl;
//        if (count == 0) debugGames++; // считаем партии
//    }

    put(move);

    //сохраняем сэмпл хода нейронки
    addSample(move);
    lastCount = count;

    int rat = node->rating;
    //std::cout << "[AI] best move " << (int)move << " node rating = " << rat << std::endl;
    return rat;
};

// -----------------------------
// predictBestMove с фильтром пустых клеток
// -----------------------------
TMove Neuro::predictBestMove() {
    torch::NoGradGuard no_grad;
    model->eval();

    // Получаем предсказания нейросети
    auto [policy_logits, value] = model->forward(getTensorFromField());

    // используем чистые логиты через softmax, без температуры
    auto p = torch::softmax(policy_logits[0], 0).clone();

    // Маскируем недопустимые ходы (зануляем их вероятность)
    auto node = current()->node;
    for (int i = 0; i < 225; ++i) {
        if (kl[i] != 1 || getChild(node,i) == NULL){// || !isExpected(node, i)) {
            p[i] = 0.0f;
        }
    }

    // Проверка на случай, если все ходы были отфильтрованы
    float sum = p.sum().item<float>();
    if (sum < 1e-6f) {
        // Если легальных ходов не нашлось в маске, берем любую свободную клетку
        std::cout << "легальных ходов не нашлось в маске, берем любую свободную клетку" << std::endl;
        for (int i = 0; i < 225; ++i)
            if (kl[i] == 0) p[i] = 1.0f;
    }

    // Находим индекс самого сильного хода (вместо случайного выбора multinomial)
    auto idx = torch::argmax(p).item<int>();

    if (kl[idx] != 1)
        std::cout << "Не удалось попасть в свободную клетку" << std::endl;

    return (TMove)idx;
}


//случайный выбор multinomial)
//TMove Neuro::predictBestMove() {
//    torch::NoGradGuard no_grad;
//    model->eval();
//
//    auto [policy_logits, value] = model->forward(getTensorFromField());
//
//    float T = 1.5f;
//    auto p = torch::softmax(policy_logits[0] / T, 0).clone();
//
//    auto node = current()->node;
//    for (int i = 0; i < 225; ++i) {
//        if (kl[i] != 1 || !isExpected(node, i)) { // только пустые и ожидаемые
//            p[i] = 0.0f;
//        }
//    }
//
//    float sum = p.sum().item<float>();
//    if (sum > 1e-6f) {
//        p = p / sum;
//    } else {
//        for (int i = 0; i < 225; ++i)
//            if (kl[i] == 0) p[i] = 1.0f;
//        p = p / p.sum();
//    }
//
//    auto idx = torch::multinomial(p, 1).item<int>();
//    return (TMove)idx;
//}

//// ==========================================================
//// Получение рейтинга
//// ==========================================================
//TRating Neuro::getNNRating(TMove move) {
//
//    torch::NoGradGuard no_grad;
//    model->eval();
//
//    auto output = model->forward(getTensorFromField());
//
//    float normVal = output[0][(int)move].item<float>();
//
//    model->train();
//
//    return decodeRating(normVal);
//}

void Neuro::trainFromGame(bool lastPlayerWon) {
    size_t n = gameHistory.size();

    static int iter = 0;

    for (size_t i = 0; i < n; ++i) {
        bool samePlayerAsWinner = ((n - 1 - i) % 2 == 0) == lastPlayerWon;
        //if (!samePlayerAsWinner) continue;

        // Победный ход игрока
        gameHistory[i].result = samePlayerAsWinner ? 1.0f : 0.0f;

//        if (samePlayerAsWinner) {
//            if (++iter < 20) {
//                std::cout << "training good move " << (int)(gameHistory[i].move) << std::endl;
//            }
//        }
        trainSample(gameHistory[i]);
    }
    gameHistory.clear();
}

void Neuro::trainSample(const TrainSample& s) {
    model->train();
    optimizer->zero_grad();

    // 1️⃣ Forward
    auto [policy_logits, value] = model->forward(s.state);

    // 2️⃣ Value target [-1,1]
    float nodeRating = decodeRating(s.node->rating);
    float gameResult = s.result * 2.0f - 1.0f; // 1 → +1, 0 → -1

    const float alpha = 0.6f;
    float combinedValue = alpha * nodeRating + (1.0f - alpha) * gameResult;

    combinedValue = std::max(-1.0f, std::min(1.0f, combinedValue));
    auto target_value = torch::tensor({combinedValue}, torch::kFloat32).to(value.device());

    // 3️⃣ Policy loss
    torch::Tensor policy_loss = torch::tensor(0.0f, torch::kFloat32).to(policy_logits.device());

    if (s.result > 0 && s.move >= 0) {
        // 🔥 учим ТОЛЬКО хорошие ходы
        auto target_move = torch::tensor({s.move}, torch::kLong).to(policy_logits.device());

        policy_loss = torch::nn::functional::cross_entropy(policy_logits, target_move);

        // 🔥 добавим небольшую энтропию (чтобы не схлопывалось)
        auto log_probs = torch::log_softmax(policy_logits, 1);
        auto entropy = -torch::sum(torch::softmax(policy_logits,1) * log_probs);

        policy_loss = policy_loss - 0.01f * entropy;
    }

    // 4️⃣ Value loss (ВСЕГДА обучается)
    auto value_loss = torch::mse_loss(value, target_value);

    // 5️⃣ Балансировка
    const float beta = 0.3f; // policy слабее, чем value
    auto loss = beta * policy_loss + (1.0f - beta) * value_loss;

    float lambda = 1e-4f; // сила L2
    for (auto& param : model->parameters()) {
        loss = loss + lambda * param.pow(2).sum();
    }
    // 6️⃣ Backprop
    loss.backward();
    optimizer->step();

    // 7️⃣ Debug
//    static int iter = 0;
//    if (iter < 20) {
//        std::cout << "[DEBUG] move=" << (int)s.move
//                  << " | result=" << s.result
//                  << " | nodeRating=" << nodeRating
//                  << " | combined=" << combinedValue
//                  << " | policy_loss=" << policy_loss.item<float>()
//                  << std::endl;
//    }
//
//    if (++iter % 500 == 0) {
//        std::cout << "[AI] trainSample | loss=" << loss.item<float>()
//                  << " | value=" << value.item<float>()
//                  << " | policy_loss=" << policy_loss.item<float>()
//                  << " | combinedValue=" << combinedValue
//                  << std::endl;
//    }

    save(loss.item<float>());
}

//---------------------------------------------------------------------------
// Преобразуем рейтинг узла в диапазон [-1, +1]
float Neuro::decodeRating(int rating) {
    const float maxRating = 32768.0f;  // нормализация по максимальному ожидаемому рейтингу
    float val = static_cast<float>(rating) / maxRating;

    // Ограничиваем диапазон
    if (val > 1.0f) val = 1.0f;
    else if (val < -1.0f) val = -1.0f;

    return val;
}