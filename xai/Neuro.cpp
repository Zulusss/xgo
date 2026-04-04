//---------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <thread>

#pragma hdrstop

#include "Neuro.h"

//---------------------------------------------------------------------------

#pragma package(smart_init)


// ==========================================================
// архитектура сети (альфа-зеро)
// ==========================================================
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
            torch::nn::Linear(2 * 15 * 15, 225));

        // value head
        value_conv = register_module("value_conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 1, 1)));
        value_fc1 = register_module("value_fc1",
            torch::nn::Linear(15 * 15, 64));
        value_fc2 = register_module("value_fc2",
            torch::nn::Linear(64, 1));
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {

        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = torch::relu(conv3->forward(x));

        // policy
        auto p = torch::relu(policy_conv->forward(x));
        p = p.view({p.size(0), -1});
        p = policy_fc->forward(p);
        p = torch::softmax(p, 1);

        // value
        auto v = torch::relu(value_conv->forward(x));
        v = v.view({v.size(0), -1});
        v = torch::relu(value_fc1->forward(v));
        v = torch::tanh(value_fc2->forward(v));

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

    if (++iter % 20 == 0) {
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


void Neuro::addSample() {
    TrainSample sample;
    sample.state = getTensorFromField();
    sample.move = -1;
    sample.result = 0;
    sample.isXTurn = (this->count % 2 == 0);// кто сейчас ходит:
    gameHistory.push_back(sample);

}

int Neuro::moveNeuro() {

    //сохраняем в сэмпл ход легаси
    if (count>1) {
        if (gameHistory.empty()) addSample();
        gameHistory.back().move = current()->move;
    }

    //готовим сэмпл для хода нейронки
    addSample();

    TMove move = predictBestMove();
    TNode* node = current()->node;

    put(move);

    //сохраняем сэмпл хода нейронки
    gameHistory.back().move = move;

    //готовим сэмпл для хода легаси
    addSample();

    int rat = node->rating;
    //std::cout << "[AI] best move " << (int)move << " node rating = " << rat << std::endl;
    return rat;
};

TMove Neuro::predictBestMove() {

    torch::NoGradGuard no_grad;
    model->eval();

    auto [policy, value] = model->forward(getTensorFromField());

    auto p = policy[0].clone();

    // маска недопустимых
    for (int i = 0; i < 225; ++i) {
        if (kl[i] > 1)
            p[i] = 0;
    }

    // нормализация
    p = p / p.sum();

//    // 🔥 exploration
//    if (rand() % 100 < 20) {
//        auto idx = torch::multinomial(p, 1).item<int>();
//        return (TMove)idx;
//    }

    return (TMove)p.argmax().item<int>();
}

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
    // Победитель = last player
    float z = lastPlayerWon ? 1.0f : 0.0f;
    size_t n = gameHistory.size();

    for (size_t i = 0; i < n; ++i) {
        // Определяем, какой игрок сделал ход (последний ход в истории — победитель)
        bool samePlayerAsWinner = ((n - 1 - i) % 2 == 0);

        // Результат для value head всегда:
        gameHistory[i].result = samePlayerAsWinner ? z : 1.0f - z;

        // Обучаем каждый сэмпл
        trainSample(gameHistory[i], samePlayerAsWinner);
    }

    gameHistory.clear();
}

void Neuro::trainSample(const TrainSample& s, bool isWinnerMove) {
    model->train();
    optimizer->zero_grad();

    auto [policy, value] = model->forward(s.state);

    // Value target всегда
    auto target_value = torch::tensor({s.result}, torch::kFloat32).to(value.device());
    auto value_loss = torch::mse_loss(value, target_value);

    // Policy target
    torch::Tensor policy_loss = torch::tensor({0.0f}, torch::kFloat32).to(policy.device());
    if (isWinnerMove && s.move >= 0) {
        torch::Tensor target_policy = torch::zeros({1, 225}, torch::kFloat32).to(policy.device());
        target_policy[0][s.move] = 1.0f;
        policy_loss = -torch::sum(target_policy * torch::log(policy + 1e-6));
    } else {
        // Мягкий штраф за "далёкий" ход проигравшего
        // Чтобы градиент не нулевой, увеличиваем веса для того, где ход не близко к центру
        float penalty_scale = 0.01f; // можно подстроить
        int y = s.move / 15;
        int x = s.move % 15;
        float dist = std::sqrt((y - 7)*(y - 7) + (x - 7)*(x - 7)) / 10.0f; // нормируем до ~1
        policy_loss = penalty_scale * dist * torch::sum(policy);
    }

    auto loss = value_loss + policy_loss;

    loss.backward();
    optimizer->step();

    // Лог
    static int iter = 0;
    if (++iter % 500 == 0) {
        std::cout << "[AI] trainSample | loss=" << loss.item<float>()
                  << " | value=" << value.item<float>()
                  << " | policy_loss=" << policy_loss.item<float>()
                  << std::endl;
    }

    save(loss.item<float>());
}