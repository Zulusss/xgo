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
        // Общие сверточные слои
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 64, 3).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)));

        // Policy head
        policy_conv = register_module("policy_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 2, 1)));
        policy_fc   = register_module("policy_fc", torch::nn::Linear(2 * 15 * 15, 225));

        // Value head
        value_conv = register_module("value_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 1, 1)));
        value_fc1  = register_module("value_fc1", torch::nn::Linear(15 * 15, 64));
        value_fc2  = register_module("value_fc2", torch::nn::Linear(64, 1));
    }

    // --- Forward pass ---
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        // Общие conv слои с leaky ReLU
        x = torch::leaky_relu(conv1->forward(x), 0.01);
        x = torch::leaky_relu(conv2->forward(x), 0.01);
        x = torch::leaky_relu(conv3->forward(x), 0.01);

        // Policy head
        auto p = policy_conv->forward(x);
        p = p.view({p.size(0), -1});
        p = policy_fc->forward(p);

        // Value head
        auto v = value_conv->forward(x);
        v = v.view({v.size(0), -1});
        v = torch::leaky_relu(value_fc1->forward(v), 0.01);

        // Dropout только в режиме обучения
        if (this->is_training()) {
            v = torch::dropout(v, /*p=*/0.15, /*train=*/true);
        }

        v = value_fc2->forward(v);

        return std::make_tuple(p, v);
    }

    torch::Tensor decodeAbsRating(torch::Tensor raw) {
        return 0.63661978 * torch::atan(raw / 2600.0f);
    }

    // Слои
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Conv2d policy_conv{nullptr}, value_conv{nullptr};
    torch::nn::Linear policy_fc{nullptr}, value_fc1{nullptr}, value_fc2{nullptr};
};

// ==========================================================
// Конструктор
// ==========================================================
Neuro::Neuro(SimplyNumbers* s, Hashtable* h, int gameMode)
        : GameBoard(s, h, gameMode){

#ifdef __linux__
    setenv("NNPACK_MODE", "0", 1);
#endif

    lossTracker = new MovingAverage(3000);

    trackerNX = new MovingAverage(30);
    trackerNO = new MovingAverage(30);
    trackerLX = new MovingAverage(30);
    trackerLO = new MovingAverage(30);
    trackerNNX = new MovingAverage(30);
    trackerNNO = new MovingAverage(30);

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
    lossTracker->put(loss);

    if (++iter % 600 == 0) {
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

int Neuro::moveNeuro() {

    TMove move = predictBestMove();
    TNode* node = current()->node;

    putWithoutSwap(move);

    int rat = node->rating;
    //std::cout << "[AI] best move " << (int)move << " node rating = " << rat << std::endl;
    return rat;
};

/// -----------------------------
// predictBestMove (оптимизировано)
// -----------------------------
TMove Neuro::predictBestMove() {
    torch::NoGradGuard no_grad;
    model->eval();

    // 1️⃣ Forward
    auto [policy_logits, value] = model->forward(getTensorFromField());

    // =========================================================
    // 🔥 EXPLOIT (argmax) — kl[i] == 1 + getChild != NULL
    // =========================================================

    auto p = torch::softmax(policy_logits[0], 0).clone();

    auto node = current()->node;
    bool isCheck = node->isCheck();
    int free = -1;

    for (int i = 0; i < 225; ++i) {
     if (kl[i] != 1 || isCheck && getChild(node, i) == NULL) {
         p[i] = 0.0f;
     } else {
         free = i;
     }
    }

    float sum = p.sum().item<float>();

    if (sum < 1e-6f) {
     if (free >= 0) {
         std::cout << "fallback (argmax): берем последнюю допустимую клетку" << std::endl;
         return (TMove)free;
     }

     std::cout << "fallback (argmax): нет допустимых ходов!" << std::endl;
     return (TMove)0;
    }

    p = p / sum;

    auto idx = torch::argmax(p).item<int>();

    if (kl[idx] != 1 || isCheck && getChild(node, idx) == NULL) {
     std::cout << "argmax попал в запрещённую клетку, ищем вручную" << std::endl;

     float best = -1.0f;
     int bestIdx = -1;

     for (int i = 0; i < 225; ++i) {
         if (kl[i] == 1 && (!isCheck || getChild(node, i) != NULL)) {
             float val = p[i].item<float>();
             if (val > best) {
                 best = val;
                 bestIdx = i;
             }
         }
     }

     if (bestIdx >= 0)
         return (TMove)bestIdx;

     if (free >= 0)
         return (TMove)free;
    }

    return (TMove)idx;
}

void Neuro::trainNetworkOnCurrentPosition() {
    auto node = current()->node;

    model->train();
    optimizer->zero_grad();

    // 1️⃣ Forward
    auto state = getTensorFromField();
    auto [policy_logits, value] = model->forward(state);

    // 2️⃣ VALUE
    TRating currentRating = -node->rating;//rating for current player is opposite to rating of opponent
    auto target_value = model->decodeAbsRating(torch::tensor({(float)currentRating}, torch::kFloat32).to(value.device()));
    auto value_loss = torch::mse_loss(value, target_value);

    // 3️⃣ Собираем кандидатов
    std::vector<std::pair<int, float>> candidates;

    for (int i = 0; i < 225; ++i) {
        if (kl[i] != 1) continue;

        auto child = getChild(node, i);
        if (!child) continue;

        //для policy head, относительная оценка <=0, ноль означает лучший ход
        float r = std::clamp((child->rating - currentRating) / 2000.0f, -4.0f, 0.0f);
        candidates.push_back({i, r});
    }

    if (candidates.empty()) return;

    // Сортировка по рейтингу (убывание)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto &a, const auto &b){ return a.second > b.second; });

    int maxK = IS_X_TURN ? 5 : 6;

    int adaptiveK = std::min((int)candidates.size(), std::min(maxK, (int)node->totalDirectChilds));

    float bestVal = candidates[0].second;
    float kthVal  = candidates[adaptiveK - 1].second;

    float spread = std::max(0.0f, bestVal - kthVal);

    float T;
    if (spread > 0.5f)       T = 0.35f;
    else if (spread > 0.2f)  T = 0.45f;
    else                     T = 0.5f;

    // 4️⃣ Формируем целевое распределение для policy
    auto target_probs = torch::zeros({225}, torch::kFloat32).to(policy_logits.device());
    float sumExp = 0.0f;

    for (int j = 0; j < adaptiveK; ++j) {
        float val = candidates[j].second;
        int idx = candidates[j].first;

        float rel = bestVal - val;

        float bonus = 1.0f + std::exp(-rel * 12.0f);

        float x = std::clamp(val / T, -10.0f, 10.0f);
        float e = bonus * std::exp(x);

        target_probs[idx] = e;
        sumExp += e;
    }

    if (sumExp > 1e-6f)
        target_probs /= sumExp;

    // 🔥 Немного размазываем распределение
    target_probs = target_probs * 0.92f + 0.08f / 225;

    // ===== POLICY LOSS =====
    auto log_probs = torch::log_softmax(policy_logits, 1);
    auto policy_loss = -torch::sum(target_probs * log_probs[0]);

    auto probs = torch::softmax(policy_logits, 1);
    auto entropy = -torch::sum(probs * log_probs);

    float entropyCoef = (!IS_X_TURN ? 0.04f : 0.02f);
    policy_loss -= entropyCoef * entropy;

    // ===== TOTAL LOSS =====
    float beta = (!IS_X_TURN) ? 0.6f
        : (node->totalChilds > 50000) ? 0.4f : 0.5f;

    auto loss = beta * policy_loss + (1.0f - beta) * value_loss;
    if (!IS_X_TURN) loss *= 1.6f;

    // L2 регуляризация
    float lambda = 1e-4f;
    for (auto& param : model->parameters())
        loss += lambda * param.pow(2).sum();

    loss.backward();
    optimizer->step();

    // 🔹 Логгирование
    static int dbg = 0;
    if (++dbg % 200 == 0) {
        std::cout << "[TREE TRAIN " << dbg
                  << "] spread=" << spread
                  //<< " avg target=" << target_probs.mean().item<float>() //always gives 0.0044
                  << " childs=" << node->totalChilds
                  << " direct=" << (int)node->totalDirectChilds
                  << " adaptiveK=" << adaptiveK
                  << " value=" << value.item<float>()
                  << " target=" << target_value.item<float>()
                  << " loss=" << loss.item<float>()
                  << " policy_loss=" << policy_loss.item<float>()
                  << " delta=" << (value - target_value).abs().item<float>()
                  << std::endl;
    }

    save(loss.item<float>());
}
