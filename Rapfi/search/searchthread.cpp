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

#include "searchthread.h"

#include "../config.h"
#include "../core/iohelper.h"
#include "../core/platform.h"
#include "../game/board.h"
#include "movepick.h"
#include "opening.h"
#include "searcher.h"

#include <algorithm>
#include <unordered_set>

namespace Search {

namespace {

/// Generate the root move list for the main thread from the current board
/// position and options: applies block-move filtering, balance2 move pair
/// expansion, and root symmetry filtering.
void generateRootMoves(SearchThread &th)
{
    auto addMoveToRootMoves = [&th](Pos m) {
        // Ignore blocked moves
        if (std::count(th.options().blockMoves.begin(),
                       th.options().blockMoves.end(),
                       m))
            return;

        if (th.options().balanceMode == Search::SearchOptions::BALANCE_TWO) {
            // Use candidates before first move
            std::unordered_set<Pos> cands;
            FOR_EVERY_CAND_POS(th.board, pos)
            {
                cands.insert(pos);
            }
            th.board->move(th.options().rule, m);
            // Generate second move for balance2
            MovePicker movePicker2(th.options().rule,
                                   *th.board,
                                   MovePicker::ExtraArgs<MovePicker::ROOT> {});
            while (Pos m2 = movePicker2()) {
                if (cands.find(m2) != cands.end()) {
                    Search::Balance2Move bm {m, m2};
                    th.rootMoves.emplace_back(bm);
                    th.balance2Moves[bm] = th.rootMoves.size() - 1;
                }
            }
            th.board->undo(th.options().rule);
        }
        else {
            th.rootMoves.emplace_back(m);
        }
    };

    // Generate root moves for main search thread
    MovePicker movePicker(th.options().rule,
                          *th.board,
                          MovePicker::ExtraArgs<MovePicker::ROOT> {});
    while (Pos m = movePicker()) {
        addMoveToRootMoves(m);
    }

    // If all legal moves are blocked, we select all candidate moves as root moves
    if (th.rootMoves.empty() && th.options().blockMoves.size() > 0) {
        std::unordered_set<Pos> cands;
        FOR_EVERY_CAND_POS(th.board, pos)
        {
            cands.insert(pos);
        }

        for (const auto &m : cands) {
            addMoveToRootMoves(m);
        }
    }

    // Filter root moves with symmetry (not for balance two)
    if (th.options().balanceMode != Search::SearchOptions::BALANCE_TWO) {
        std::vector<Pos> rootMoveList;
        for (const auto &rm : th.rootMoves) {
            rootMoveList.push_back(rm.pv[0]);
        }

        Opening::filterSymmetryMoves(*th.board, rootMoveList);

        if (rootMoveList.size() < th.rootMoves.size()) {
            // Remove root moves that are not in the filtered list
            auto pred = [&rootMoveList](const Search::RootMove &rm) -> bool {
                return std::find(rootMoveList.begin(), rootMoveList.end(), rm.pv[0])
                       == rootMoveList.end();
            };
            th.rootMoves.erase(std::remove_if(th.rootMoves.begin(), th.rootMoves.end(), pred),
                               th.rootMoves.end());
        }
    }
}

}  // namespace

/// Global search thread pool
ThreadPool Threads;

void SearchContext::reset(const SearchOptions &opts, bool ponder)
{
    options                  = opts;
    inPonder                 = ponder;
    callsCnt                 = 0;
    startPonderAfterThinking = false;
    resultAction             = ActionType::Move;
    bestMove                 = Pos::NONE;
}

SearchThread::SearchThread(ThreadPool &threadPool, uint32_t id)
    : numaId(Numa::DefaultNumaNodeId)
    , running(false)
    , exit(false)
    , id(id)
    , threads(threadPool)
{}

void SearchThread::init(bool bindGroup)
{
#ifdef MULTI_THREADING
    thread = std::thread(&SearchThread::threadLoop, this);
#endif

    runTask([bindGroup](SearchThread &th) {
        // Create search data for this thread
        th.searchData = th.threads.searcher()->makeSearchData(th);

        // Set thread affinity to a specific group if needed
        if (bindGroup) {
            // If OS already scheduled us on a different group than 0 then don't overwrite
            // the choice, eventually we are one of many one-threaded processes running on
            // some Windows NUMA hardware, for instance in fishtest. To make it simple,
            // just check if running threads are below a threshold, in this case all this
            // NUMA machinery is not needed. We also store this thread's numa ID for the
            // later NUMA-aware loading of evaluator weights.
            th.numaId = Numa::bindThisThread(th.id);
        }
    });
}

SearchThread::~SearchThread()
{
    exit = true;

#ifdef MULTI_THREADING
    runTask(nullptr);
    thread.join();
#endif
}

void SearchThread::runTask(std::function<void(SearchThread &)> task)
{
#ifdef MULTI_THREADING
    if (std::this_thread::get_id() == thread.get_id()) {
        // We *are* the worker => enqueue "tail task" without waiting.
        std::lock_guard<std::mutex> lock(mutex);
        // at this point running is still true, we simply replace the functor
        taskFunc = std::move(task);
    }
    else {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return !running; });
        taskFunc = std::move(task);
        running  = true;
        lock.unlock();
        cv.notify_one();
    }
#else
    if (task)
        task(*this);
#endif
}

void SearchThread::waitForIdle()
{
#ifdef MULTI_THREADING
    // Check deadlock if we are already in the worker thread
    assert(std::this_thread::get_id() != thread.get_id());

    if (!running)
        return;

    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return !running; });
#endif
}

#ifdef MULTI_THREADING
void SearchThread::threadLoop()
{
    while (true) {
        std::function<void(SearchThread &)> task;

        {
            std::unique_lock<std::mutex> lock(mutex);
            if (!taskFunc) {
                running = false;
                cv.notify_all();
                cv.wait(lock, [&] { return running || exit; });
            }

            if (exit)
                return;

            std::swap(task, taskFunc);
        }

        if (task)
            task(*this);
    }
}
#endif

void SearchThread::clear()
{
    if (searchData)
        searchData->clearData(*this);
    rootMoves.clear();
    balance2Moves.clear();
    numNodes = 0;
    selDepth = 0;

    // Setup dbClient for each thread
    if (threads.dbStorage() && (!dbClient || &dbClient->getStorage() != threads.dbStorage())) {
        dbClient = std::make_unique<Database::DBClient>(*threads.dbStorage(),
                                                        Database::RECORD_MASK_LVDB,
                                                        Config::DatabaseCacheSize,
                                                        Config::DatabaseRecordCacheSize);
    }
}

void SearchThread::setBoardAndEvaluator(const Board &board)
{
    // Reset board instance in this thread to be null
    this->board.reset();

    // Setup evaluator in this thread
    if (!threads.evaluatorMaker)
        evaluator.reset();
    else {
        const int  boardSize = board.size();
        const Rule rule      = threads.ctx.options.rule;

        // Clear loaded evaluator that does not match
        if (evaluator && (evaluator->boardSize != boardSize || evaluator->rule != rule))
            evaluator.reset();

        if (!evaluator)
            evaluator = threads.evaluatorMaker(boardSize, rule, numaId);
    }

    // Clone the board (this will also sync the evaluator to the board state)
    this->board = std::make_unique<Board>(board, this);
}

void SearchContext::checkExit(uint32_t elapsedCalls)
{
    // We only check exit condition after a number of calls.
    // This is to avoid expensive calculation in timeup condition checking.
    if (callsCnt > elapsedCalls) {
        callsCnt -= elapsedCalls;
        return;
    }

    // Resets callsCnt
    if (options.maxNodes)
        callsCnt = std::min(1024U, uint32_t(options.maxNodes / 1024U));
    else
        callsCnt = 1024;

    // Do not stop searching in pondering mode
    if (inPonder.load(std::memory_order_relaxed))
        return;

    // Check if we have reached node/time limits
    if (options.maxNodes && pool->nodesSearched() >= options.maxNodes
        || options.timeLimit && pool->searcher()->checkTimeupCondition(timectl)) {
        markPonderingAvailable();
        pool->stopThinking();
    }
}

void SearchContext::markPonderingAvailable()
{
    if (options.pondering && !inPonder.load(std::memory_order_relaxed))
        startPonderAfterThinking.store(true, std::memory_order_relaxed);
}

void ThreadPool::runOnAllThreads(std::function<void(SearchThread &)> task, bool includeSelf)
{
    if (!task)
        return;

    // Run task in non-main threads
    for (size_t i = 1; i < size(); i++)
        (*this)[i]->runTask(task);

    // Run task in the calling (main) thread
    if (includeSelf)
        task(*main());

    // Wait for all other threads to finish
    waitForIdle();
}

void ThreadPool::waitForIdle()
{
#ifdef MULTI_THREADING
    // Iterate all other threads and wait for them to finish
    for (auto &th : *this)
        if (th->thread.get_id() != std::this_thread::get_id())
            th->waitForIdle();
#endif
}

void ThreadPool::setNumThreads(size_t numThreads)
{
    // Destroy all threads first (which will also wait for them to be idle)
    while (!empty())
        pop_back();  // std::unique_ptr will automatically destroy thread

    // Create requested amount of threads
    if (numThreads > 0) {
        bool bindGroup = numThreads > Numa::BindGroupThreshold;

        // The first thread created is the main search thread (id == 0)
        push_back(std::make_unique<SearchThread>(*this, 0u));

#ifdef MULTI_THREADING
        while (size() < numThreads) {
            push_back(std::make_unique<SearchThread>(*this, (uint32_t)size()));
        }
#endif

        // Init the threads
        for (auto &th : *this)
            th->init(bindGroup);
    }
}

void ThreadPool::setupSearcher(std::unique_ptr<Searcher> newSearcher)
{
    waitForIdle();

    size_t memLimitKB = 0;
    if (searcher())
        memLimitKB = searcher()->getMemoryLimit();

    assert(newSearcher);
    searcherPtr = std::move(newSearcher);

    if (memLimitKB)
        searcher()->setMemoryLimit(memLimitKB);

    // Re-instantiate all threads
    setNumThreads(size());
}

void ThreadPool::setupDatabase(std::unique_ptr<Database::DBStorage> dbStorage)
{
    if (!empty()) {
        waitForIdle();
        for (const auto &th : *this)
            th->dbClient.reset();
    }

    dbStoragePtr = std::move(dbStorage);
}

void ThreadPool::setupEvaluator(std::function<EvaluatorMaker> maker)
{
    if (!empty()) {
        waitForIdle();
        for (const auto &th : *this) {
            th->board.reset();
            th->evaluator.reset();
        }
    }

    evaluatorMaker = maker;
}

void ThreadPool::startThinking(const Board          &board,
                               const SearchOptions  &options,
                               bool                  inPonder,
                               std::function<void()> onStop)
{
    assert(size() > 0);
    assert(searcher());

    // If we are already thinking, wait for it first
    waitForIdle();
    terminate = false;

    // Reset the search context and clean up main thread state
    ctx.reset(options, inPonder);
    main()->clear();

    // Clone the input board to main thread and update evaluator
    main()->setBoardAndEvaluator(board);

    // Expand board candidate if needed
    Opening::expandCandidate(*main()->board);

    // Generate root moves for the main search thread
    generateRootMoves(*main());

    // Launch a small task to clear threads state and copy state from main thread
    runOnAllThreads(
        [mainTh = main()](SearchThread &th) {
            th.clear();
            th.setBoardAndEvaluator(*mainTh->board);
            th.rootMoves     = mainTh->rootMoves;
            th.balance2Moves = mainTh->balance2Moves;
        },
        false);

    // Start the main search thread
    main()->runTask([this, onStop = std::move(onStop)](SearchThread &th) {
        runSearch(th);
        if (onStop)  // If onStop is set, queue a tail task to call it
            main()->runTask([onStop = std::move(onStop)](SearchThread &th) { onStop(); });
    });
}

bool ThreadPool::tryTrivialBestmove(SearchThread &main)
{
    SearchOptions &opts = ctx.options;

    // Probe opening database and find if there is a prepared opening
    if (!opts.disableOpeningQuery
        && Opening::probeOpening(*main.board, opts.rule, ctx.resultAction, ctx.bestMove)) {
        ctx.markPonderingAvailable();
        return true;
    }

    // Check for immediate move
    if (main.rootMoves.empty()) {
        // If there is no stones on board, it is possible that the opponent played a pass
        // move at the start of one game. We just choose the center location to play.
        if (main.board->nonPassMoveCount() == 0) {
            ctx.bestMove = main.board->centerPos();
            return true;
        }

        // Return the first empty position if we might find a forced forbidden
        // point mate in Renju, or all legal points have been blocked.
        FOR_EVERY_EMPTY_POS(main.board, pos)
        {
            ctx.bestMove = pos;
            ctx.printer.printBestmoveWithoutSearch(main, pos, mated_in(0), 0, nullptr);
            return true;
        }

        return true;  // abnormal case: GUI might have a bug
    }
    // If we are winning, return directly
    else if (main.board->p4Count(main.board->sideToMove(), A_FIVE)) {
        assert(main.board->cell(main.rootMoves[0].pv[0]).pattern4[main.board->sideToMove()]
               == A_FIVE);
        main.rootMoves[0].value = mate_in(1);
        ctx.bestMove            = main.rootMoves[0].pv[0];

        if (searcher()->printsTrivialBestmove)
            ctx.printer.printBestmoveWithoutSearch(main,
                                                   main.rootMoves[0].pv[0],
                                                   main.rootMoves[0].value,
                                                   1,
                                                   &main.rootMoves[0].pv);
        return true;
    }

    return false;
}

void ThreadPool::finalizeResult(SearchThread &main, const RootMove &finalMove)
{
    // Do not record bestmove in pondering
    if (ctx.inPonder.load(std::memory_order_relaxed))
        return;

    // Record best move
    ctx.bestMove = finalMove.pv[0];

    // If swap check is needed, make swap decision according to the rule
    if (ctx.options.swapable)
        ctx.resultAction = Opening::decideAction(*main.board, ctx.options.rule, finalMove.value);
    else if (ctx.options.balanceMode == SearchOptions::BalanceMode::BALANCE_TWO)
        ctx.resultAction = ActionType::Move2;
    else
        ctx.resultAction = ActionType::Move;
}

void ThreadPool::runSearch(SearchThread &main)
{
    // Handle trivial cases that need no real search
    if (tryTrivialBestmove(main))
        return;

    // Init time management of this search
    ctx.timectl.init(ctx.options.turnTime,
                     ctx.options.matchTime,
                     ctx.options.timeLeft,
                     {main.board->ply(), main.board->movesLeft()});

    // Run the algorithm's main search routine and finalize its result
    if (const RootMove *finalMove = searcher()->searchMain(main))
        finalizeResult(main, *finalMove);
}

void ThreadPool::clear(bool clearAllMemory)
{
    if (empty())
        setNumThreads(Config::DefaultThreadNum);

    if (searcher())
        searcher()->clear(*this, clearAllMemory);
}

ThreadPool::ThreadPool()
{
    ctx.pool = this;

    // Set default searcher
    setupSearcher(Config::createSearcher());
}

ThreadPool::~ThreadPool()
{
#ifdef MULTI_THREADING
    // Stop if there are still some threads thinking
    stopThinking();
#endif
    // Explicitly free all threads
    setNumThreads(0);
}

}  // namespace Search
