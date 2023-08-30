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

#include "movepick.h"

#include "../eval/evaluator.h"
#include "../game/board.h"
#include "../game/movegen.h"
#include "searchthread.h"

#include <algorithm>

namespace {

/// Move picking stages.
/// Usual procedure: X_TT -> X_MOVES -> ALLMOVES.
enum Stages {
    MAIN_TT,
    MAIN_MOVES,
    DEFENDFIVE_TT,
    DEFENDFIVE_MOVES,
    DEFENDFOUR_TT,
    DEFENDFOUR_MOVES,
    DEFENDB4F3_TT,
    DEFENDB4F3_MOVES,
    QVCF_TT,
    QVCF_MOVES,
    ALLMOVES,
};

/// Partial sort the move list up to the score limit. It dynamiclly decides
/// which sorting algorithm to use based on how many moves are in the list.
void fastPartialSort(Move *begin, Move *end, Score limit)
{
    // heruistic values
    constexpr size_t InsertionSortLimit = MAX_MOVES / 4;
    constexpr size_t SortLimit          = MAX_MOVES * 2 / 3;

    size_t nMoves = end - begin;
    if (nMoves <= InsertionSortLimit) {
        // Sorts moves in descending order up to and including a given limit.
        // The order of moves smaller than the limit is left unspecified.
        for (Move *sortedEnd = begin, *p = begin + 1; p < end; ++p)
            if (p->score >= limit) {
                Move tmp = *p, *q;
                *p       = *++sortedEnd;
                for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
                    *q = *(q - 1);
                *q = tmp;
            }
    }
    else if (nMoves <= SortLimit) {
        std::sort(begin, end, std::greater<>());
    }
    else {
        std::partial_sort(begin, begin + SortLimit, end, std::greater<>());
    }
}

}  // namespace

namespace Search {

/// MovePicker constructor at the root node.
/// Generate all legal moves for current side to move.
template <>
MovePicker::MovePicker(Rule rule, const Board &board, ExtraArgs<MovePicker::ROOT> args)
    : board(board)
    , mainHistory(nullptr)
    , counterMoveHistory(nullptr)
    , continuationHistory(nullptr)
    , stage(ALLMOVES)
    , rule(rule)
    , ttMove(Pos::NONE)
    , allowPlainB4InVCF(false)
    , hasPolicy(false)
{
    Color self = board.sideToMove(), oppo = ~self;
    curMove = moves;

    if (board.p4Count(self, A_FIVE))
        endMove = generate<WINNING>(board, curMove);
    else if (board.p4Count(oppo, A_FIVE))
        endMove = generate<DEFEND_FIVE>(board, curMove);
    else if (board.p4Count(self, B_FLEX4))
        endMove = generate<WINNING>(board, curMove);
    else if (board.p4Count(oppo, B_FLEX4)) {
        endMove = generate<DEFEND_FOUR | ALL>(board, curMove);
        endMove = generate<VCF>(board, endMove);
    }
    else if (board.p4Count(oppo, C_BLOCK4_FLEX3)
             && (rule != Rule::RENJU || validateOpponentCMove(board))) {
        switch (rule) {
        default:
        case FREESTYLE: endMove = generate<DEFEND_B4F3 | RULE_FREESTYLE>(board, curMove); break;
        case STANDARD: endMove = generate<DEFEND_B4F3 | RULE_STANDARD>(board, curMove); break;
        case RENJU: endMove = generate<DEFEND_B4F3 | RULE_RENJU>(board, curMove); break;
        }
        if (endMove == curMove)
            endMove = generate<ALL>(board, curMove);
        else
            endMove = generate<VCF>(board, endMove);
    }
    else
        endMove = generate<ALL>(board, curMove);
}

/// MovePicker constructor for the main search.
template <>
MovePicker::MovePicker(Rule rule, const Board &board, ExtraArgs<MovePicker::MAIN> args)
    : board(board)
    , mainHistory(args.mainHistory)
    , counterMoveHistory(args.counterMoveHistory)
    , continuationHistory(args.continuationHistory)
    , rule(rule)
    , allowPlainB4InVCF(false)
    , hasPolicy(false)
{
    assert(mainHistory);
    Color oppo = ~board.sideToMove();
    bool  ttmValid;

    if (board.p4Count(oppo, A_FIVE)) {
        stage    = DEFENDFIVE_TT;
        ttmValid = board.cell(args.ttMove).pattern4[oppo] == A_FIVE;
    }
    else if (board.p4Count(oppo, B_FLEX4)) {
        stage = DEFENDFOUR_TT;

        const Cell &ttCell = board.cell(args.ttMove);
        ttmValid           = ttCell.pattern4[BLACK] >= E_BLOCK4 || ttCell.pattern4[BLACK] == FORBID
                   || ttCell.pattern4[WHITE] >= E_BLOCK4;
    }
    else if (board.p4Count(oppo, C_BLOCK4_FLEX3)
             && (rule != Rule::RENJU || validateOpponentCMove(board))) {
        stage    = DEFENDB4F3_TT;
        ttmValid = true;
    }
    else {
        stage    = MAIN_TT;
        ttmValid = true;
    }

    // check legality for defence ttmove
    ttmValid = ttmValid && board.isEmpty(args.ttMove);

    stage += !ttmValid;
    ttMove = ttmValid ? args.ttMove : Pos::NONE;
}

/// MovePicker constructor for quick vcf search.
template <>
MovePicker::MovePicker(Rule rule, const Board &board, ExtraArgs<MovePicker::QVCF> args)
    : board(board)
    , mainHistory(nullptr)
    , continuationHistory(nullptr)
    , rule(rule)
    , allowPlainB4InVCF(
          args.depth >= DEPTH_QVCF_FULL
          || (args.previousSelfP4[0] >= D_BLOCK4_PLUS && args.previousSelfP4[1] >= D_BLOCK4_PLUS))
    , hasPolicy(false)
{
    Color self = board.sideToMove(), oppo = ~self;
    bool  ttmValid;

    if (board.p4Count(oppo, A_FIVE)) {
        stage    = DEFENDFIVE_TT;
        ttmValid = board.cell(args.ttMove).pattern4[oppo] == A_FIVE;
    }
    else {
        stage    = QVCF_TT;
        ttmValid = board.cell(args.ttMove).pattern4[self] >= E_BLOCK4;
    }

    // check legality for defence ttmove
    ttmValid = ttmValid && board.isEmpty(args.ttMove);

    stage += !ttmValid;
    ttMove = ttmValid ? args.ttMove : Pos::NONE;
}

/// Return the next move satisfying a predicate function.
/// Selected move is recorded in curMoves. It never returns the TT move.
/// If there is no move left, it returns Pos::NONE.
template <MovePicker::PickType T, typename Pred>
Pos MovePicker::pickNextMove(Pred filter)
{
    bool forbidden = rule == Rule::RENJU && board.sideToMove() == BLACK;

    while (curMove < endMove) {
        if constexpr (T == Best)
            std::swap(*curMove, *std::max_element(curMove, endMove));

        if (curMove->pos != ttMove && (!forbidden || !board.checkForbiddenPoint(*curMove))
            && filter()) {
            curScore       = curMove->score;
            curPolicyScore = curMove->rawScore;
            return *curMove++;
        }

        curMove++;
    }

    return Pos::NONE;
}

/// Score all remaining moves according to score type.
template <MovePicker::ScoreType Type>
void MovePicker::scoreMoves()
{
    using Evaluation::Evaluator;
    using Evaluation::PolicyBuffer;
    using PolicyBufferStorage = std::aligned_storage_t<sizeof(PolicyBuffer), alignof(PolicyBuffer)>;

    PolicyBufferStorage policyBufferStorage;
    Color               self = board.sideToMove(), oppo = ~self;
    PolicyBuffer       *policyBuf = reinterpret_cast<PolicyBuffer *>(&policyBufferStorage);
    Evaluator *evaluator = board.thisThread() ? board.thisThread()->evaluator.get() : nullptr;

    if (bool(Type & POLICY) && evaluator) {
        new (policyBuf) Evaluation::PolicyBuffer(board.size());

        // Set compute flag for all moves in move list
        for (auto &m : *this)
            policyBuf->setComputeFlag(m.pos);

        evaluator->evaluatePolicy(board, *policyBuf);
        hasPolicy      = true;
        maxPolicyScore = std::numeric_limits<Score>::lowest() / 2;  // avoid underflow
    }

    for (auto &m : *this) {
        const Cell &c = board.cell(m);

        if (bool(Type & POLICY) && evaluator) {
            m.score = m.rawScore = policyBuf->score(m.pos);
            maxPolicyScore       = std::max(maxPolicyScore, m.rawScore);
        }
        else if constexpr (bool(Type & BALANCED))
            m.score = m.rawScore = c.score[self];
        else if constexpr (bool(Type & ATTACK))
            m.score = m.rawScore = (c.score[self] * 2 + c.score[oppo]) / 3;
        else if constexpr (bool(Type & DEFEND))
            m.score = m.rawScore = (c.score[self] + c.score[oppo] * 2) / 3;
        else
            assert(false && "incorrect score type");

        if constexpr (bool(Type & MAIN_HISTORY)) {
            if (c.pattern4[self] >= H_FLEX3)
                m.score += (*mainHistory)[self][m.pos][HIST_ATTACK] / 128;
            else
                m.score += (*mainHistory)[self][m.pos][HIST_QUIET] / 256;
        }

        if constexpr (bool(Type & COUNTER_MOVE)) {
            assert(counterMoveHistory);

            if (Pos lastMove = board.getLastMove(); board.isInBoard(lastMove)) {
                const int CounterMoveBonus = 21;
                auto [counterMove, counterMoveP4] =
                    *((*counterMoveHistory)[oppo][lastMove.moveIndex()]);

                if (counterMove == m.pos && counterMoveP4 <= c.pattern4[self])
                    m.score += CounterMoveBonus;
            }
        }

        if constexpr (bool(Type & CONT_HISTORY)) {
            assert(continuationHistory);

            // m.score += (*continuationHistory[0])[m.pos] / 1024     //
            //            + (*continuationHistory[1])[m.pos] / 2048   //
            //            + (*continuationHistory[3])[m.pos] / 2048   //
            //            + (*continuationHistory[5])[m.pos] / 2048;  //

            // m.score += (*continuationHistory[0])[m.pos] / 2048;
        }
    }
}

// Pick the next legal move until there is no legal move left.
/// @return Next legal move, or Pos::NONE if there is no legal move left.
Pos MovePicker::operator()()
{
top:
    switch (stage) {
    case MAIN_TT:
    case DEFENDFIVE_TT:
    case DEFENDFOUR_TT:
    case DEFENDB4F3_TT:
    case QVCF_TT: ++stage; return ttMove;

    case MAIN_MOVES:
        assert(!board.p4Count(~board.sideToMove(), A_FIVE));
        assert(!board.p4Count(~board.sideToMove(), B_FLEX4));

        curMove = moves;
        endMove = generate<ALL>(board, curMove);

        scoreMoves<ScoreType(BALANCED | POLICY | MAIN_HISTORY | COUNTER_MOVE | CONT_HISTORY)>();
        fastPartialSort(curMove, endMove, 0);

        stage = ALLMOVES;
        goto top;

    case DEFENDFIVE_MOVES:
        assert(board.p4Count(~board.sideToMove(), A_FIVE));

        curMove = moves;
        endMove = !ttMove ? generate<DEFEND_FIVE>(board, moves) : moves;

        stage = ALLMOVES;
        goto top;

    case DEFENDFOUR_MOVES:
        assert(board.p4Count(~board.sideToMove(), B_FLEX4));

        curMove = moves;
        endMove = generate<DEFEND_FOUR>(board, curMove);
        endMove = generate<VCF>(board, endMove);

        scoreMoves<ScoreType(BALANCED | POLICY | MAIN_HISTORY)>();
        fastPartialSort(curMove, endMove, 0);

        stage = ALLMOVES;
        goto top;

    case DEFENDB4F3_MOVES:
        assert(board.p4Count(~board.sideToMove(), C_BLOCK4_FLEX3));

        curMove = moves;
        switch (rule) {
        default:
        case FREESTYLE: endMove = generate<DEFEND_B4F3 | RULE_FREESTYLE>(board, curMove); break;
        case STANDARD: endMove = generate<DEFEND_B4F3 | RULE_STANDARD>(board, curMove); break;
        case RENJU: endMove = generate<DEFEND_B4F3 | RULE_RENJU>(board, curMove); break;
        }

        if (endMove == curMove) {
            stage = MAIN_MOVES;
            goto top;
        }

        endMove = generate<VCF>(board, endMove);

        scoreMoves<ScoreType(BALANCED | POLICY | MAIN_HISTORY)>();
        fastPartialSort(curMove, endMove, 0);

        stage = ALLMOVES;
        goto top;

    case QVCF_MOVES:
        curMove = moves;
        {
            Pos selfLast = board.getLastActualMoveOfSide(board.sideToMove());
            endMove =
                (allowPlainB4InVCF ? generateNeighbors<VCF>
                                   : generateNeighbors<VCF | COMB>)(board,
                                                                    curMove,
                                                                    selfLast,
                                                                    RANGE_SQUARE2_LINE4,
                                                                    arraySize(RANGE_SQUARE2_LINE4));
        }

        scoreMoves<BALANCED>();
        fastPartialSort(curMove, endMove, 0);

        stage = ALLMOVES;
        [[fallthrough]];

    case ALLMOVES: return pickNextMove<Next>([]() { return true; });
    }

    // This should never be reached, unless a bug occurs
    assert(false && "unknown MovePicker stage occurred");
    return Pos::NONE;
}

}  // namespace Search
