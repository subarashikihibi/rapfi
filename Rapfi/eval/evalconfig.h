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

namespace Evaluation {

/// EvaluatorConfig groups the classical/evaluator switching margins and draw
/// adjustments read from "[model.evaluator]". One mutable global instance
/// (EvalCfg) holds the live configuration; config.cpp's readEvaluator fills it
/// (behind the type+weights gate - a margins-only table has no effect), and
/// the gomocup INFO handlers assign the draw fields directly.
struct EvaluatorConfig
{
    float marginWinLossScale    = 1.18f;
    float marginWinLossExponent = 3.07f;
    float marginScale           = 395.0f;
    float drawBlackWinRate      = 0.5f;
    float drawRatio             = 1.0f;
};

/// The global live evaluator configuration.
extern EvaluatorConfig EvalCfg;

}  // namespace Evaluation
