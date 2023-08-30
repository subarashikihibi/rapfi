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

#pragma once

#include "../game/movegen.h"
#include "history.h"

namespace Search {

/// MovePicker class is used to pick one legal move at a time from the current
/// position. In order to improve the efficiency of the alpha beta algorithm,
/// it attempts to return the moves which are most likely to get a cut-off first.
class MovePicker
{
public:
    enum SearchType { ROOT, MAIN, QVCF };
    template <SearchType ST>
    struct ExtraArgs
    {};

    /// Construct a MovePicker object with arguments used to decide which
    /// generation phase to set and provide information for move ordering.
    template <SearchType ST>
    MovePicker(Rule rule, const Board &board, ExtraArgs<ST> args);
    MovePicker(const MovePicker &)            = delete;
    MovePicker &operator=(const MovePicker &) = delete;

    [[nodiscard]] Pos operator()();
    bool              hasPolicyScore() const { return hasPolicy; }
    Score             maxMovePolicy() const { return maxPolicyScore; }
    Score             curMovePolicy() const { return curPolicyScore; }
    Score             curMoveScore() const { return curScore; }
    Score             curMovePolicyDiff() const { return maxPolicyScore - curPolicyScore; }
    Score             curMoveScoreDiff() const { return maxPolicyScore - curScore; }

private:
    enum PickType { Next, Best };
    enum ScoreType {
        ATTACK       = 0b01,
        DEFEND       = 0b10,
        BALANCED     = ATTACK | DEFEND,
        POLICY       = 0b100,
        MAIN_HISTORY = 0b1000,
        COUNTER_MOVE = 0b10000,
        CONT_HISTORY = 0b100000,
    };

    template <PickType T, typename Pred>
    Pos pickNextMove(Pred);
    template <ScoreType T>
    void  scoreMoves();
    Move *begin() { return curMove; }
    Move *end() { return endMove; }

    const Board              &board;
    const MainHistory        *mainHistory;
    const CounterMoveHistory *counterMoveHistory;
    const MoveHistory       **continuationHistory;
    int8_t                    stage;
    Rule                      rule;
    Pos                       ttMove;
    bool                      allowPlainB4InVCF;
    bool                      hasPolicy;
    Score                     curScore, curPolicyScore, maxPolicyScore;
    Move                     *curMove, *endMove;
    Move                      moves[MAX_MOVES];
};

template <>
struct MovePicker::ExtraArgs<MovePicker::ROOT>
{};

template <>
struct MovePicker::ExtraArgs<MovePicker::MAIN>
{
    Pos                       ttMove;
    const MainHistory        *mainHistory;
    const CounterMoveHistory *counterMoveHistory;
    const MoveHistory       **continuationHistory;
};

template <>
struct MovePicker::ExtraArgs<MovePicker::QVCF>
{
    Pos      ttMove;
    Depth    depth;  // negative depth in qvcf search
    Pattern4 previousSelfP4[2];
};

}  // namespace Search
