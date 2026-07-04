/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "config.h"

#include "command/command.h"
#include "core/compressor.h"
#include "core/iohelper.h"
#include "database/dbclient.h"
#include "database/dbconfig.h"
#include "database/dbstorage.h"
#include "database/yxdbstorage.h"
#include "eval/evalconfig.h"
#include "eval/evaluator.h"
#include "eval/mix10nnue.h"
#include "eval/mix9svqnnue.h"
#include "game/pattern.h"
#include "search/ab/searcher.h"
#include "search/hashtable.h"
#include "search/mcts/searcher.h"
#include "search/searchconfig.h"
#include "search/searchthread.h"

#ifdef USE_ORT_EVALUATOR
    #include "eval/onnxevaluator.h"
#endif

#include <cpptoml.h>
#include <fstream>
#include <functional>
#include <limits>
#ifdef MULTI_THREADING
    #include <thread>
#endif

namespace {

static constexpr uint32_t N            = PATTERN_NB;
static constexpr uint32_t TABLE1_COUNT = combineNumber(N, 1);
static constexpr uint32_t TABLE2_COUNT = combineNumber(N, 2);

constexpr uint32_t I(uint32_t y, uint32_t x)
{
    // x >= y
    assert(x >= y);
    return y * N + x - combineNumber(y, 2);
}

}  // namespace

namespace Evaluation {

/// Scaling Factor is used for conversion between eval value and win rate.
/// Formula: win rate = sigmoid(eval / ScalingFactor)
float ScalingFactor = 200.0f;

// Classical evaluation and score tables
// Note that Renju has asymmetry eval and score

Eval          EVALS[RULE_NB + 1][PCODE_NB];
Eval          EVALS_THREAT[RULE_NB + 1][THREAT_NB];
Pattern4Score P4SCORES[RULE_NB + 1][PCODE_NB];

}  // namespace Evaluation

namespace Config {

// -------------------------------------------------
// General options

/// Should we reload config file before searching each move.
bool ReloadConfigEachMove = false;
/// Should we clear hash after each time config file is loaded.
bool ClearHashAfterConfigLoaded = true;
/// Default number of therads if not specified (0 means max hardware concurrency).
size_t DefaultThreadNum = 1;
/// Message output mode.
MsgMode MessageMode = MsgMode::BRIEF;
/// Coordinate convertion mode for protocol I/O.
CoordConvertionMode IOCoordMode = CoordConvertionMode::NONE;
/// Default candidate range mode if not specified when creating board.
CandidateRange DefaultCandidateRange = CandidateRange::SQUARE3_LINE4;
/// Memory reserved for stuff other than hash table in max_memory option.
size_t MemoryReservedMB[RULE_NB] = {0};
/// Default hash table size (zero for not setting).
size_t DefaultTTSizeKB = 0;

// -------------------------------------------------
// Search options

const char *DefaultSearcherName = "alphabeta";

// -------------------------------------------------
// Database options

/// Database storage factory, which takes the url (in utf-8 encoding)
/// and returns a unique pointer to an instance of DBStorage.
std::function<std::unique_ptr<::Database::DBStorage>(std::string)> DatabaseMaker;

// -------------------------------------------------

void readRequirement(const cpptoml::table &t);
void readGeneral(const cpptoml::table &t);
void readSearch(const cpptoml::table &t);
void readModel(const cpptoml::table &t);
void readEvaluator(const cpptoml::table &t);
void readDatabase(const cpptoml::table &t);
template <typename ValueType,
          ValueType MinVal    = std::numeric_limits<ValueType>::lowest(),
          ValueType MaxVal    = std::numeric_limits<ValueType>::max(),
          typename SetterType = void(PatternCode, ValueType)>
void readValueModel(const cpptoml::table &t, SetterType setter);

}  // namespace Config

/// Load config from a stream.
/// @param configStream A input stream that contains a config file.
/// @return Returns true if loading succeeded, otherwise returns false.
bool Config::loadConfig(std::istream &configStream)
{
    Search::Engine.setupEvaluator(nullptr);
    Search::Engine.setupDatabase(nullptr);

    try {
        auto c = cpptoml::parser(configStream).parse();

        if (auto requirement = c->get_table("requirement"))
            readRequirement(*requirement);

        if (auto general = c->get_table("general"))
            readGeneral(*general);

        if (auto search = c->get_table("search"))
            readSearch(*search);

        if (auto database = c->get_table("database"))
            readDatabase(*database);

        if (auto model = c->get_table("model"))
            readModel(*model);
    }
    catch (const std::exception &e) {
        ERRORL("Failed to load config: " << e.what());
        return false;
    }

    return true;
}

/// Read requirement table of the config.
/// This is used to check if the config file is suitable for current version of Rapfi.
void Config::readRequirement(const cpptoml::table &t)
{
    auto [major, minor, revision] = getVersionNumbers();
    uint64_t rapVer = ((uint64_t)major << 32) | ((uint64_t)minor << 16) | (uint64_t)revision;
    if (auto minVer = t.get_array_of<int64_t>("min_version")) {
        if (minVer->size() != 3)
            throw std::runtime_error("illegal min_version");
        uint64_t cfgVer = ((*minVer)[0] << 32) | ((*minVer)[1] << 16) | (*minVer)[2];
        if (cfgVer > rapVer)
            throw std::runtime_error("config requires newer version of rapfi");
    }
    if (auto maxVer = t.get_array_of<int64_t>("max_version")) {
        if (maxVer->size() != 3)
            throw std::runtime_error("illegal max_version");
        uint64_t cfgVer = ((*maxVer)[0] << 32) | ((*maxVer)[1] << 16) | (*maxVer)[2];
        if (cfgVer < rapVer)
            throw std::runtime_error("config requires older version of rapfi");
    }
}

/// Read general table of the config.
void Config::readGeneral(const cpptoml::table &t)
{
    ReloadConfigEachMove = t.get_as<bool>("reload_config_each_move").value_or(ReloadConfigEachMove);
    ClearHashAfterConfigLoaded =
        t.get_as<bool>("clear_hash_after_config_loaded").value_or(ClearHashAfterConfigLoaded);

    // Read Default Thread Num
    DefaultThreadNum = t.get_as<uint64_t>("default_thread_num").value_or(DefaultThreadNum);
    if (DefaultThreadNum == 0) {
#ifdef MULTI_THREADING
        DefaultThreadNum = std::thread::hardware_concurrency();
        MESSAGEL("Setting default thread num to " << DefaultThreadNum << ".");
#else
        DefaultThreadNum = 1;
#endif
    }

    // Read Message Mode
    if (t.get_as<std::string>("message_mode")) {
        std::string msgModeStr = *t.get_as<std::string>("message_mode");
        if (msgModeStr == "normal")
            MessageMode = MsgMode::NORMAL;
        else if (msgModeStr == "brief")
            MessageMode = MsgMode::BRIEF;
        else if (msgModeStr == "ucilike")
            MessageMode = MsgMode::UCILIKE;
        else {
            if (msgModeStr != "none")
                MESSAGEL("Warning: unknown message mode [" << msgModeStr << "], reset to [none].");
            MessageMode = MsgMode::NONE;
        }
    }

    // Read Coord Conversion Mode
    if (t.get_as<std::string>("coord_conversion_mode")) {
        std::string coordModeStr = *t.get_as<std::string>("coord_conversion_mode");
        if (coordModeStr == "X_flipY")
            IOCoordMode = CoordConvertionMode::X_FLIPY;
        else if (coordModeStr == "flipY_X")
            IOCoordMode = CoordConvertionMode::FLIPY_X;
        else {
            if (coordModeStr != "none")
                MESSAGEL("Warning: unknown coordinate conversion mode [" << coordModeStr
                                                                         << "], reset to [none].");
            IOCoordMode = CoordConvertionMode::NONE;
        }
    }

    // Read Default Condidate Range Mode
    if (t.get_as<std::string>("default_candidate_range")) {
        std::string candRangeModeStr = *t.get_as<std::string>("default_candidate_range");
        if (candRangeModeStr == "square2")
            DefaultCandidateRange = CandidateRange::SQUARE2;
        else if (candRangeModeStr == "square2_line3")
            DefaultCandidateRange = CandidateRange::SQUARE2_LINE3;
        else if (candRangeModeStr == "square3")
            DefaultCandidateRange = CandidateRange::SQUARE3;
        else if (candRangeModeStr == "square3_line4")
            DefaultCandidateRange = CandidateRange::SQUARE3_LINE4;
        else if (candRangeModeStr == "square4")
            DefaultCandidateRange = CandidateRange::SQUARE4;
        else if (candRangeModeStr == "full_board")
            DefaultCandidateRange = CandidateRange::FULL_BOARD;
        else {
            MESSAGEL("Warning: unknown candidate range [" << candRangeModeStr
                                                          << "], reset to [square2_line3].");
            DefaultCandidateRange = CandidateRange::SQUARE2_LINE3;
        }
    }

    // Read memory reserved for each rule
    if (auto table = t.get_array("memory_reserved_mb")) {
        if (auto array = table->get_array_of<int64_t>()) {
            for (int i = 0; i < RULE_NB; i++)
                MemoryReservedMB[i] = array->at(std::min<size_t>(i, array->size() - 1));
        }
    }
    else {
        auto v = t.get_as<uint64_t>("memory_reserved_mb");
        for (int i = 0; i < RULE_NB; i++)
            MemoryReservedMB[i] = v.value_or(MemoryReservedMB[i]);
    }

    DefaultTTSizeKB = t.get_as<uint64_t>("default_tt_size_kb").value_or(DefaultTTSizeKB);
    // Resize TT according to default TT size (overriding previous size)
    if (DefaultTTSizeKB > 0)
        Search::Engine.searcher()->setMemoryLimit(DefaultTTSizeKB);
}

/// Read search table of the config.
void Config::readSearch(const cpptoml::table &t)
{
    if (auto v = t.get_as<std::string>("default_searcher"); v)
        Search::Engine.setupSearcher(createSearcher(*v));

    using Search::SearchCfg, Search::TimeCfg;

    // Parameters for alpha-beta search
    SearchCfg.aspirationWindow =
        t.get_as<bool>("aspiration_window").value_or(SearchCfg.aspirationWindow);
    SearchCfg.filterSymmetryRootMoves =
        t.get_as<bool>("filter_symmetry_root_moves").value_or(SearchCfg.filterSymmetryRootMoves);
    SearchCfg.numIterationAfterMate =
        t.get_as<int>("num_iteration_after_mate").value_or(SearchCfg.numIterationAfterMate);
    SearchCfg.numIterationAfterSingularRoot =
        t.get_as<int>("num_iteration_after_singular_root")
            .value_or(SearchCfg.numIterationAfterSingularRoot);
    SearchCfg.maxSearchDepth =
        t.get_as<int>("max_search_depth").value_or(SearchCfg.maxSearchDepth);

    // Parameters for MCTS search
    SearchCfg.expandWhenFirstEvaluate =
        t.get_as<bool>("expand_when_first_evaluate").value_or(SearchCfg.expandWhenFirstEvaluate);
    SearchCfg.maxNumVisitsPerPlayout =
        t.get_as<int>("max_num_visits_per_playout").value_or(SearchCfg.maxNumVisitsPerPlayout);
    SearchCfg.nodesToPrintMCTSRootmoves = t.get_as<int>("nodes_to_print_mcts_rootmoves")
                                              .value_or(SearchCfg.nodesToPrintMCTSRootmoves);
    SearchCfg.timeToPrintMCTSRootmoves =
        t.get_as<int>("time_to_print_mcts_rootmoves").value_or(SearchCfg.timeToPrintMCTSRootmoves);
    SearchCfg.maxNonPVRootmovesToPrint = t.get_as<int>("max_non_pv_rootmoves_to_print")
                                             .value_or(SearchCfg.maxNonPVRootmovesToPrint);
    SearchCfg.numNodesAfterSingularRoot = t.get_as<int>("num_nodes_after_singular_root")
                                              .value_or(SearchCfg.numNodesAfterSingularRoot);
    SearchCfg.numNodeTableShardsPowerOfTwo =
        t.get_as<int>("num_node_table_shards_power_of_two")
            .value_or(SearchCfg.numNodeTableShardsPowerOfTwo);
    SearchCfg.drawUtilityPenalty =
        t.get_as<double>("draw_utility_penalty").value_or(SearchCfg.drawUtilityPenalty);

    // Read time management options
    if (auto tm = t.get_table("timectl")) {
        TimeCfg.turnTimeReserved =
            tm->get_as<int>("turn_time_reserved").value_or(TimeCfg.turnTimeReserved);
        TimeCfg.matchSpace = tm->get_as<double>("match_space").value_or(TimeCfg.matchSpace);
        TimeCfg.matchSpaceMin =
            tm->get_as<double>("match_space_min").value_or(TimeCfg.matchSpaceMin);
        TimeCfg.averageBranchFactor =
            tm->get_as<double>("average_branch_factor").value_or(TimeCfg.averageBranchFactor);
        TimeCfg.advancedStopRatio =
            tm->get_as<double>("advanced_stop_ratio").value_or(TimeCfg.advancedStopRatio);
        TimeCfg.moveHorizon = tm->get_as<int>("move_horizon").value_or(TimeCfg.moveHorizon);

        TimeCfg.timeDivisorScale =
            tm->get_as<double>("time_divisor_scale").value_or(TimeCfg.timeDivisorScale);
        TimeCfg.timeDivisorBias =
            tm->get_as<double>("time_divisor_bias").value_or(TimeCfg.timeDivisorBias);
        TimeCfg.timeDivisorDepthPow =
            tm->get_as<double>("time_divisor_depth_pow").value_or(TimeCfg.timeDivisorDepthPow);

        TimeCfg.fallingFactorScale =
            tm->get_as<double>("falling_factor_scale").value_or(TimeCfg.fallingFactorScale);
        TimeCfg.fallingFactorBias =
            tm->get_as<double>("falling_factor_bias").value_or(TimeCfg.fallingFactorBias);

        TimeCfg.bestmoveStableReductionScale =
            tm->get_as<double>("bestmove_stable_reduction_scale")
                .value_or(TimeCfg.bestmoveStableReductionScale);
        TimeCfg.bestmoveStablePrevReductionPow =
            tm->get_as<double>("bestmove_stable_prev_reduction_pow")
                .value_or(TimeCfg.bestmoveStablePrevReductionPow);
    }
}

/// Read model table of all rules in the config.
void Config::readModel(const cpptoml::table &t)
{
    const Rule  Rules[]    = {FREESTYLE, STANDARD, RENJU};
    const char *RuleName[] = {"freestyle", "standard", "renju"};

    std::string modelPath = t.get_as<std::string>("binary_file").value_or("");
    if (!modelPath.empty()) {
        if (!Command::loadModelFromFile(modelPath))
            throw std::runtime_error("failed to load classic model file");
    }
    else {
        // Read Eval & Score
        if (auto eval = t.get_table("eval")) {
            for (Rule r : Rules) {
                bool hasAsymmetryRenjuEval = false;
                auto setEvalBlack = [r](PatternCode pcode, Eval ev) {
                    Evaluation::EVALS[r + BLACK][pcode] = ev;
                };
                auto setEvalWhite = [r](PatternCode pcode, Eval ev) {
                    Evaluation::EVALS[r + WHITE][pcode] = ev;
                };
                auto ruleEval = eval->get_table(RuleName[r]);
                if (!ruleEval)  // fallback
                    ruleEval = eval;
                else if (r == RENJU) {
                    auto blackEval    = ruleEval->get_table("black");
                    auto whiteEval    = ruleEval->get_table("white");
                    auto fallbackEval = ruleEval->get_as<int64_t>("model_type") ? ruleEval : eval;
                    if (hasAsymmetryRenjuEval = blackEval || whiteEval; hasAsymmetryRenjuEval) {
                        readValueModel<Eval, -16384, 16383>(*(blackEval ? blackEval : fallbackEval),
                                                            setEvalBlack);
                        readValueModel<Eval, -16384, 16383>(*(whiteEval ? whiteEval : fallbackEval),
                                                            setEvalWhite);
                    }
                }

                if (!hasAsymmetryRenjuEval) {
                    if (r == RENJU) {
                        readValueModel<Eval, -16384, 16383>(*ruleEval, setEvalBlack);
                        readValueModel<Eval, -16384, 16383>(*ruleEval, setEvalWhite);
                    }
                    else {
                        readValueModel<Eval, -16384, 16383>(*ruleEval, setEvalBlack);
                    }
                }
            }
        }

        if (auto score = t.get_table("score")) {
            auto readScore = [](const cpptoml::table &t, int tableIdx) {
                auto selfTable = t.get_table("self");
                auto oppoTable = t.get_table("oppo");
                readValueModel<Score, -8192, 8191>(selfTable ? *selfTable : t,
                                                   [tableIdx](PatternCode pcode, Score score) {
                                                       Evaluation::P4SCORES[tableIdx][pcode][0] =
                                                           score;
                                                   });
                readValueModel<Score, -8192, 8191>(oppoTable ? *oppoTable : t,
                                                   [tableIdx](PatternCode pcode, Score score) {
                                                       Evaluation::P4SCORES[tableIdx][pcode][1] =
                                                           score;
                                                   });
            };
            for (Rule r : Rules) {
                bool hasAsymmetryRenjuScore = false;
                auto ruleScore = score->get_table(RuleName[r]);
                if (!ruleScore)  // fallback
                    ruleScore = score;
                else if (r == RENJU) {
                    auto blackScore = ruleScore->get_table("black");
                    auto whiteScore = ruleScore->get_table("white");
                    auto fallbackScore =
                        ruleScore->get_as<int64_t>("model_type") ? ruleScore : score;
                    if (hasAsymmetryRenjuScore = blackScore || whiteScore; hasAsymmetryRenjuScore) {
                        readScore(*(blackScore ? blackScore : fallbackScore), r + BLACK);
                        readScore(*(whiteScore ? whiteScore : fallbackScore), r + WHITE);
                    }
                }

                if (!hasAsymmetryRenjuScore) {
                    if (r == RENJU) {
                        readScore(*ruleScore, r + BLACK);
                        readScore(*ruleScore, r + WHITE);
                    }
                    else {
                        readScore(*ruleScore, r);
                    }
                }
            }
        }
    }

    // Read scalingFactor
    Evaluation::ScalingFactor =
        (float)t.get_as<double>("scaling_factor").value_or(Evaluation::ScalingFactor);

    // Read evaluator
    if (auto evaluator = t.get_table("evaluator"))
        readEvaluator(*evaluator);
}

/// Read evaluator table in the config.
void Config::readEvaluator(const cpptoml::table &t)
{
    using namespace std::filesystem;

    auto evaluatorType = t.get_as<std::string>("type");
    auto weights       = t.get_table_array("weights");
    if (!evaluatorType || !weights || weights->begin() == weights->end())
        return;

    auto warpEvaluatorMaker = [weights,
                               evaluatorName = *evaluatorType](auto maker,
                                                               bool seperateBlackAndWhiteWeights) {
        return [=](int              boardSize,
                   Rule             rule,
                   Numa::NumaNodeId numaId) -> std::unique_ptr<Evaluation::Evaluator> {
            try {
                for (auto weightCfg : *weights) {
                    path weightPath;
                    path blackWeightPath, whiteWeightPath;

                    if (auto weightFile = weightCfg->get_as<std::string>("weight_file"))
                        weightPath = Command::getModelFullPath(u8path(*weightFile));
                    else if (!seperateBlackAndWhiteWeights)
                        throw std::runtime_error("must specify weight_file in weight configs.");

                    if (seperateBlackAndWhiteWeights && weightPath.empty()) {
                        auto weightFileBlack = weightCfg->get_as<std::string>("weight_file_black");
                        auto weightFileWhite = weightCfg->get_as<std::string>("weight_file_white");

                        blackWeightPath = Command::getModelFullPath(u8path(*weightFileBlack));
                        whiteWeightPath = Command::getModelFullPath(u8path(*weightFileWhite));
                        if (!weightFileBlack || !weightFileWhite)
                            throw std::runtime_error(
                                "must specify weight_file or weight_file_black "
                                "and weight_file_white in weight configs.");
                    }
                    else {
                        blackWeightPath = weightPath;
                        whiteWeightPath = weightPath;
                    }

                    try {
                        return maker(boardSize,
                                     rule,
                                     numaId,
                                     weightPath,
                                     std::make_pair(blackWeightPath, whiteWeightPath),
                                     *weightCfg);
                    }
                    catch (const Evaluation::UnsupportedRuleError &e) {
                    }
                    catch (const Evaluation::UnsupportedBoardSizeError &e) {
                    }
                    catch (const std::exception &e) {
                        if (MessageMode != MsgMode::NONE)
                            MESSAGEL("Failed to load from "
                                     << (!weightPath.empty()
                                             ? pathToConsoleString(weightPath)
                                             : pathToConsoleString(blackWeightPath) + " and "
                                                   + pathToConsoleString(whiteWeightPath))
                                     << " due to error: " << e.what());
                    }
                }

                if (MessageMode != MsgMode::NONE)
                    MESSAGEL("Evaluator " << evaluatorName
                                          << " disabled: no compatible weight config found.");
                return nullptr;
            }
            catch (const std::exception &e) {
                ERRORL("Evaluator " << evaluatorName << " failed to initialized: " << e.what());
                return nullptr;
            }
        };
    };

    if (*evaluatorType == "mix9svq") {
        Search::Engine.setupEvaluator(warpEvaluatorMaker(
            [=](int                   boardSize,
                Rule                  rule,
                Numa::NumaNodeId      numaId,
                path                  weightPath,
                std::pair<path, path> blackAndWhiteWeightPath,
                const cpptoml::table &weightCfg) {
                return std::make_unique<Evaluation::mix9svq::Evaluator>(
                    boardSize,
                    rule,
                    numaId,
                    blackAndWhiteWeightPath.first,
                    blackAndWhiteWeightPath.second);
            },
            true));
    }
    else if (*evaluatorType == "mix10") {
        Search::Engine.setupEvaluator(warpEvaluatorMaker(
            [=](int                   boardSize,
                Rule                  rule,
                Numa::NumaNodeId      numaId,
                path                  weightPath,
                std::pair<path, path> blackAndWhiteWeightPath,
                const cpptoml::table &weightCfg) {
                return std::make_unique<Evaluation::mix10::Evaluator>(
                    boardSize,
                    rule,
                    numaId,
                    blackAndWhiteWeightPath.first,
                    blackAndWhiteWeightPath.second);
            },
            true));
    }
#ifdef USE_ORT_EVALUATOR
    else if (*evaluatorType == "ort") {
        std::string deviceName = t.get_as<std::string>("ort_device").value_or("");

        Search::Engine.setupEvaluator(warpEvaluatorMaker(
            [=](int                   boardSize,
                Rule                  rule,
                Numa::NumaNodeId      numaId,
                path                  weightPath,
                std::pair<path, path> blackAndWhiteWeightPath,
                const cpptoml::table &weightCfg) {
                return std::make_unique<Evaluation::onnx::OnnxEvaluator>(boardSize,
                                                                         rule,
                                                                         weightPath,
                                                                         deviceName);
            },
            false));
    }
#endif
    else {
        throw std::runtime_error("unsupported evaluator type " + *evaluatorType);
    }

    // Read classical/evaluator switching margin
    auto &evalCfg               = Evaluation::EvalCfg;
    evalCfg.marginWinLossScale  = (float)t.get_as<double>("margin_winloss_scale")
                                     .value_or(evalCfg.marginWinLossScale);
    evalCfg.marginWinLossExponent =
        (float)t.get_as<double>("margin_winloss_exp").value_or(evalCfg.marginWinLossExponent);
    evalCfg.marginScale = (float)t.get_as<double>("margin_scale").value_or(evalCfg.marginScale);
    evalCfg.drawBlackWinRate =
        (float)t.get_as<double>("draw_black_winrate").value_or(evalCfg.drawBlackWinRate);
    evalCfg.drawRatio        = (float)t.get_as<double>("draw_ratio").value_or(evalCfg.drawRatio);
    evalCfg.drawBlackWinRate = std::clamp(evalCfg.drawBlackWinRate, 0.0f, 1.0f);
    evalCfg.drawRatio        = std::clamp(evalCfg.drawRatio, 0.0f, 1.0f);

    MESSAGEL("Evaluator set to " << *evaluatorType << ".");
}

/// Read database table in the config.
void Config::readDatabase(const cpptoml::table &t)
{
    using Database::DatabaseCfg;

    DatabaseCfg.defaultEnabled =
        t.get_as<bool>("enable_by_default").value_or(DatabaseCfg.defaultEnabled);
    DatabaseCfg.type      = t.get_as<std::string>("type").value_or(DatabaseCfg.type);
    DatabaseCfg.url       = t.get_as<std::string>("url").value_or(DatabaseCfg.url);
    DatabaseCfg.cacheSize = t.get_as<size_t>("cache_size").value_or(DatabaseCfg.cacheSize);
    DatabaseCfg.recordCacheSize =
        t.get_as<size_t>("record_cache_size").value_or(DatabaseCfg.recordCacheSize);
    DatabaseCfg.legacyFileCodePage =
        t.get_as<int>("legacy_file_code_page").value_or(DatabaseCfg.legacyFileCodePage);
    DatabaseMaker = nullptr;

    if (DatabaseCfg.type == "yixindb") {
        if (DatabaseCfg.url.empty())
            DatabaseCfg.url = "rapfi.db";

        DatabaseCfg.yixindb = {};
        if (auto args = t.get_table("yixindb")) {
            DatabaseCfg.yixindb.compressedSave =
                args->get_as<bool>("compressed_save").value_or(DatabaseCfg.yixindb.compressedSave);
            DatabaseCfg.yixindb.saveOnClose =
                args->get_as<bool>("save_on_close").value_or(DatabaseCfg.yixindb.saveOnClose);
            DatabaseCfg.yixindb.numBackupsOnSave = args->get_as<int>("num_backups_on_save")
                                                       .value_or(DatabaseCfg.yixindb.numBackupsOnSave);
            DatabaseCfg.yixindb.ignoreCorrupted =
                args->get_as<bool>("ignore_corrupted").value_or(DatabaseCfg.yixindb.ignoreCorrupted);
        }

        DatabaseMaker = [compressedSave   = DatabaseCfg.yixindb.compressedSave,
                         saveOnClose      = DatabaseCfg.yixindb.saveOnClose,
                         numBackupsOnSave = DatabaseCfg.yixindb.numBackupsOnSave,
                         ignoreCorrupted  = DatabaseCfg.yixindb.ignoreCorrupted](
                            std::string utf8URL) -> std::unique_ptr<::Database::DBStorage> {
            try {
                auto dbPath    = std::filesystem::u8path(utf8URL);
                bool existing  = std::filesystem::exists(dbPath);
                auto startTime = now();
                MESSAGEL("Opening yixin database at " << pathToConsoleString(dbPath) << " ...");
                auto dbStorage = std::make_unique<::Database::YXDBStorage>(dbPath,
                                                                           compressedSave,
                                                                           saveOnClose,
                                                                           numBackupsOnSave,
                                                                           ignoreCorrupted);
                if (existing)
                    MESSAGEL("Loaded Yixin database (" << dbStorage->size() << " records) using "
                                                       << (now() - startTime) << " ms.");
                return std::move(dbStorage);
            }
            catch (const std::exception &e) {
                ERRORL("Failed to create yixin database: " << e.what());
                return nullptr;
            }
        };
    }
    else if (!DatabaseCfg.type.empty()) {
        throw std::runtime_error("unsupported database type " + DatabaseCfg.type);
    }

    if (auto s = t.get_table("search")) {
        auto &dbs        = DatabaseCfg.search;
        dbs.readonlyMode = s->get_as<bool>("readonly_mode").value_or(false);
        dbs.mandatoryParentWrite = s->get_as<bool>("mandatory_parent_write").value_or(true);
        dbs.queryPly             = s->get_as<int>("query_ply").value_or(dbs.queryPly);
        dbs.queryPVIterPerPlyIncrement =
            s->get_as<int>("pv_iter_per_ply_increment").value_or(dbs.queryPVIterPerPlyIncrement);
        dbs.queryNonPVIterPerPlyIncrement = s->get_as<int>("nonpv_iter_per_ply_increment")
                                                .value_or(dbs.queryNonPVIterPerPlyIncrement);

        dbs.pvWritePly      = s->get_as<int>("pv_write_ply").value_or(dbs.pvWritePly);
        dbs.pvWriteMinDepth = s->get_as<int>("pv_write_min_depth").value_or(dbs.pvWriteMinDepth);

        dbs.nonPVWritePly = s->get_as<int>("nonpv_write_ply").value_or(dbs.nonPVWritePly);
        dbs.nonPVWriteMinDepth =
            s->get_as<int>("nonpv_write_min_depth").value_or(dbs.nonPVWriteMinDepth);

        dbs.writeValueRange = s->get_as<int>("write_value_range").value_or(dbs.writeValueRange);

        dbs.mateWritePly = s->get_as<int>("mate_write_ply").value_or(dbs.mateWritePly);
        dbs.mateWriteMinDepthExact =
            s->get_as<int>("mate_write_min_depth_exact").value_or(dbs.mateWriteMinDepthExact);
        dbs.mateWriteMinDepthNonExact = s->get_as<int>("mate_write_min_depth_nonexact")
                                            .value_or(dbs.mateWriteMinDepthNonExact);
        dbs.mateWriteMinStep =
            s->get_as<int>("mate_write_min_step").value_or(dbs.mateWriteMinStep);

        dbs.exactOverwritePly =
            s->get_as<int>("exact_overwrite_ply").value_or(dbs.exactOverwritePly);
        dbs.nonExactOverwritePly =
            s->get_as<int>("nonexact_overwrite_ply").value_or(dbs.nonExactOverwritePly);

        if (auto overwriteRule =
                s->get_as<std::string>("overwrite_rule").value_or("better_value_depth_bound");
            overwriteRule == "better_value_depth_bound")
            dbs.overwriteRule = ::Database::OverwriteRule::BetterValueDepthBound;
        else if (overwriteRule == "better_depth_bound")
            dbs.overwriteRule = ::Database::OverwriteRule::BetterDepthBound;
        else if (overwriteRule == "better_value")
            dbs.overwriteRule = ::Database::OverwriteRule::BetterValue;
        else if (overwriteRule == "better_label")
            dbs.overwriteRule = ::Database::OverwriteRule::BetterLabel;
        else if (overwriteRule == "always")
            dbs.overwriteRule = ::Database::OverwriteRule::Always;
        else if (overwriteRule == "disabled")
            dbs.overwriteRule = ::Database::OverwriteRule::Disabled;
        else
            MESSAGEL("unknown database overwrite rule " << overwriteRule << ", keep it unchanged.");

        dbs.overwriteExactBias =
            s->get_as<int>("overwrite_exact_bias").value_or(dbs.overwriteExactBias);
        dbs.overwriteDepthBoundBias =
            s->get_as<int>("overwrite_depth_bound_bias").value_or(dbs.overwriteDepthBoundBias);
        dbs.queryResultDepthBoundBias =
            s->get_as<int>("query_result_depth_bound_bias").value_or(dbs.queryResultDepthBoundBias);
    }

    if (auto s = t.get_table("libfile")) {
        auto &lib           = DatabaseCfg.libfile;
        lib.blackWinMark    = s->get_as<std::string>("black_win_mark").value_or("a")[0];
        lib.whiteWinMark    = s->get_as<std::string>("white_win_mark").value_or("a")[0];
        lib.blackLoseMark   = s->get_as<std::string>("black_lose_mark").value_or("c")[0];
        lib.whiteLoseMark   = s->get_as<std::string>("white_lose_mark").value_or("c")[0];
        lib.ignoreComment   = s->get_as<bool>("ignore_comment").value_or(false);
        lib.ignoreBoardText = s->get_as<bool>("ignore_board_text").value_or(false);
    }

    if (DatabaseCfg.defaultEnabled)
        Search::Engine.setupDatabase(createDefaultDBStorage());
}

/// Read a value model from a model table.
/// @tparam ValueType Type of read values.
/// @tparam MinVal Minimal of the read values. Values smaller than this will be clamped.
/// @tparam MaxVal Maximal of the read values. Values greater than this will be clamped.
/// @tparam SetterType Type of value setter, must be compatible with void(PatternCode, ValueType).
/// @param setter setter(pcode, value) is called when saving values.
template <typename ValueType, ValueType MinVal, ValueType MaxVal, typename SetterType>
void Config::readValueModel(const cpptoml::table &t, SetterType setter)
{
    enum ComposeFunc { SUM, MAX };
    auto getComposeFunc = [](const std::string &key) -> ComposeFunc {
        if (key == "sum")
            return SUM;
        else if (key == "max")
            return MAX;
        else
            throw std::runtime_error("unknown value compose function");
    };

    auto modelType = t.get_as<int64_t>("model_type");
    if (!modelType)
        throw std::runtime_error("model_type not specified");

    size_t overflowCount = 0;
    switch (*modelType) {
    case 0:
        if (auto raw = t.get_array_of<int64_t>("raw")) {
            if (raw->size() != PCODE_NB)
                throw std::runtime_error("number of values in raw model is not correct");
            for (size_t pcode = 0; pcode < PCODE_NB; pcode++) {
                int64_t val = (*raw)[pcode];
                overflowCount += val < MinVal || val > MaxVal;
                setter(pcode, ValueType(val));
            }
        }
        else
            throw std::runtime_error("raw values are missing or incorrect");
        break;
    case 1:
        if (auto table1 = t.get_array_of<int64_t>("table1")) {
            if (table1->size() != TABLE1_COUNT)
                throw std::runtime_error("size of values of table1 not correct");

            double      valScale = t.get_as<double>("table1_scale").value_or(1.0);
            int64_t     valMin   = t.get_as<int64_t>("table1_min").value_or(MinVal);
            int64_t     valMax   = t.get_as<int64_t>("table1_max").value_or(MaxVal);
            ComposeFunc compose =
                getComposeFunc(t.get_as<std::string>("table1_compose_func").value_or("sum"));

            for (int a = 0; a < N; a++)
                for (int b = a; b < N; b++)
                    for (int c = b; c < N; c++)
                        for (int d = c; d < N; d++) {
                            int64_t val = 0;
                            switch (compose) {
                            case SUM:
                                val = (*table1)[a] + (*table1)[b] + (*table1)[c] + (*table1)[d];
                                break;
                            case MAX:
                                val = std::max(
                                    {(*table1)[a], (*table1)[b], (*table1)[c], (*table1)[d]});
                                break;
                            }
                            val = std::clamp((int64_t)std::round(val * valScale), valMin, valMax);
                            overflowCount += val < MinVal || val > MaxVal;
                            setter(PatternConfig::PCODE[a][b][c][d], ValueType(val));
                        }
        }
        else
            throw std::runtime_error("table1 is missing or incorrect");
        break;
    case 2:
        if (auto table2 = t.get_array_of<int64_t>("table2")) {
            if (table2->size() != TABLE2_COUNT)
                throw std::runtime_error("size of values of table2 not correct");

            double      valScale = t.get_as<double>("table2_scale").value_or(1.0);
            int64_t     valMin   = t.get_as<int64_t>("table2_min").value_or(MinVal);
            int64_t     valMax   = t.get_as<int64_t>("table2_max").value_or(MaxVal);
            ComposeFunc compose =
                getComposeFunc(t.get_as<std::string>("table2_compose_func").value_or("sum"));

            for (int a = 0; a < N; a++)
                for (int b = a; b < N; b++)
                    for (int c = b; c < N; c++)
                        for (int d = c; d < N; d++) {
                            int64_t val = 0;
                            switch (compose) {
                            case SUM:
                                val = (*table2)[I(a, b)] + (*table2)[I(a, c)] + (*table2)[I(a, d)]
                                      + (*table2)[I(b, c)] + (*table2)[I(b, d)]
                                      + (*table2)[I(c, d)];
                                break;
                            case MAX:
                                val = std::max({(*table2)[I(a, b)],
                                                (*table2)[I(a, c)],
                                                (*table2)[I(a, d)],
                                                (*table2)[I(b, c)],
                                                (*table2)[I(b, d)],
                                                (*table2)[I(c, d)]});
                                break;
                            }
                            val = std::clamp((int64_t)std::round(val * valScale), valMin, valMax);
                            overflowCount += val < MinVal || val > MaxVal;
                            setter(PatternConfig::PCODE[a][b][c][d], ValueType(val));
                        }
        }
        else
            throw std::runtime_error("table2 is missing or incorrect");
        break;
    default: throw std::runtime_error("unknown valuation model");
    }

    if (overflowCount)
        MESSAGEL("Warning: " << overflowCount
                             << " values in (converted) raw model overflows. "
                                "Value should be in range ["
                             << MinVal << ", " << MaxVal
                             << "]. "
                                "This will cause incorrect evaluation and move sorting. "
                                "Please try to turn down large values or adding limits.");
}

bool Config::loadModel(std::istream &inStream)
{
    Compressor    compressor(inStream, Compressor::Type::LZ4_DEFAULT);
    std::istream *in = compressor.openInputStream();
    if (!in)
        return false;

    double scalingFactorF64;
    in->read(reinterpret_cast<char *>(&scalingFactorF64), sizeof(scalingFactorF64));
    Evaluation::ScalingFactor = scalingFactorF64;

    in->read(reinterpret_cast<char *>(Evaluation::EVALS), sizeof(Evaluation::EVALS));
    in->read(reinterpret_cast<char *>(Evaluation::EVALS_THREAT),
             sizeof(Evaluation::EVALS_THREAT));

    Score scores[PCODE_NB][2];
    for (int rule = 0; rule < RULE_NB + 1; rule++) {
        in->read(reinterpret_cast<char *>(scores), sizeof(scores));

        // Set score table to P4SCORES
        for (size_t pcode = 0; pcode < PCODE_NB; pcode++) {
            Evaluation::P4SCORES[rule][pcode][0] = scores[pcode][0];
            Evaluation::P4SCORES[rule][pcode][1] = scores[pcode][1];
        }
    }

    return *in && in->peek() == std::ios::traits_type::eof();
}

void Config::exportModel(std::ostream &outStream)
{
    Compressor    compressor(outStream, Compressor::Type::LZ4_DEFAULT);
    std::ostream *out = compressor.openOutputStream();
    assert(out);

    double scalingFactorF64 = Evaluation::ScalingFactor;
    out->write(reinterpret_cast<char *>(&scalingFactorF64), sizeof(scalingFactorF64));
    out->write(reinterpret_cast<char *>(Evaluation::EVALS), sizeof(Evaluation::EVALS));
    out->write(reinterpret_cast<char *>(Evaluation::EVALS_THREAT),
               sizeof(Evaluation::EVALS_THREAT));

    Score scores[PCODE_NB][2];
    for (int rule = 0; rule < RULE_NB + 1; rule++) {
        // Get score table out of P4SCORES
        for (size_t pcode = 0; pcode < PCODE_NB; pcode++) {
            scores[pcode][0] = (Score)Evaluation::P4SCORES[rule][pcode][0];
            scores[pcode][1] = (Score)Evaluation::P4SCORES[rule][pcode][1];
        }

        out->write(reinterpret_cast<char *>(scores), sizeof(scores));
    }
}

std::unique_ptr<::Search::Searcher> Config::createSearcher(std::string searcherName)
{
    if (searcherName.empty())
        searcherName = DefaultSearcherName;

    upperInplace(searcherName);

    if (searcherName == "ALPHABETA")
        return std::make_unique<Search::AB::ABSearcher>();
    if (searcherName == "MCTS")
        return std::make_unique<Search::MCTS::MCTSSearcher>();

    ERRORL("Unknown search type: " << searcherName
                                   << ", must be one of [alphabeta, mcts]."
                                      " Use alphabeta searcher as default.");
    return std::make_unique<Search::AB::ABSearcher>();
}

std::unique_ptr<::Database::DBStorage> Config::createDefaultDBStorage(std::string utf8URL)
{
    return DatabaseMaker ? DatabaseMaker(utf8URL.empty() ? Database::DatabaseCfg.url : utf8URL)
                         : nullptr;
}
