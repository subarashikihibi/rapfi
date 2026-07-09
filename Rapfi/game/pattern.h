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

#include "../core/math.h"
#include "../core/platform.h"
#include "../core/types.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>

/// Packs the line patterns of both colors for one cell into a single byte (4 bits each), so the
/// board's per-cell pattern array can store all four directions in 4 bytes.
struct Pattern2x
{
    Pattern patBlack : 4;  ///< Line pattern seen from black's perspective.
    Pattern patWhite : 4;  ///< Line pattern seen from white's perspective.

    Pattern operator[](Color side) const { return side == BLACK ? patBlack : patWhite; }

    /// Whole-byte equality (both colors at once); compiles to a single byte compare.
    friend bool operator==(Pattern2x a, Pattern2x b)
    {
        uint8_t ra, rb;
        std::memcpy(&ra, &a, sizeof(a));
        std::memcpy(&rb, &b, sizeof(b));
        return ra == rb;
    }
    friend bool operator!=(Pattern2x a, Pattern2x b) { return !(a == b); }
};
static_assert(sizeof(Pattern2x) == sizeof(uint8_t));

namespace PatternConfig {

/// Number of cells on each side of the center cell in a line. Freestyle only cares about five in
/// a row (4 neighbours suffice); standard and renju must also detect overlines, needing 5.
template <Rule R>
constexpr int HalfLineLen = R == Rule::FREESTYLE ? 4 : 5;

/// Number of dense codes for one half-line: an (L,R)-anchored ternary encoding stores, per
/// half, the run of cells visible before the board edge (v cells, each empty/black/white)
/// and nothing beyond it, giving sum over v=0..H of 3^v = (3^(H+1)-1)/2 codes.
template <Rule R>
constexpr int DenseHalfCnt = (power(3, unsigned(HalfLineLen<R> + 1)) - 1) / 2;

/// Number of distinct dense line codes, i.e. the size of every per-rule lookup table.
/// Only lines that can reach a runtime lookup are representable: the center cell is empty
/// (it is never encoded) and walls appear only as a contiguous run at each outer end.
template <Rule R>
constexpr int DenseKeyCnt = DenseHalfCnt<R> * DenseHalfCnt<R>;

static_assert(DenseHalfCnt<FREESTYLE> == 121 && DenseKeyCnt<FREESTYLE> == 14641);
static_assert(DenseHalfCnt<STANDARD> == 364 && DenseKeyCnt<STANDARD> == 132496);

namespace detail {

    /// Dense code of one half-line, or -1 if the half violates wall contiguity (a non-wall
    /// cell beyond the first wall cannot occur on a real board line).
    /// @tparam H             Number of cells in the half.
    /// @tparam NearAtLowBits Bit order of the 2H-bit half: true if the center-adjacent cell
    ///                       sits at the lowest bits (the right/HI half of a line key);
    ///                       false if the farthest cell does (the left/LO half).
    template <int H, bool NearAtLowBits>
    constexpr int denseHalfCode(uint32_t half)
    {
        int  code = 0, pow3 = 1;
        bool wallSeen = false;
        for (int i = 0; i < H; i++) {  // i = distance from center, minus 1
            uint32_t cell = (half >> (NearAtLowBits ? 2 * i : 2 * (H - 1 - i))) & 0b11;
            if (cell == 0b00) {  // wall terminates the visible run
                wallSeen = true;
                continue;
            }
            if (wallSeen)
                return -1;
            // Ternary digit (internal convention shared by fill and lookup):
            // empty=0, black=1, white=2. Nearest cell = least significant digit.
            int digit = cell == 0b11 ? 0 : cell == 0b10 ? 1 : 2;
            code += digit * pow3;
            pow3 *= 3;
        }
        // pow3 == 3^v (v = visible run length); all shorter-run blocks precede this one,
        // occupying (3^v - 1) / 2 codes.
        return (pow3 - 1) / 2 + code;
    }

    /// Build one half-code lookup table. Entries are scaled by `Scale` (the premultiplied
    /// row stride for LO tables, 1 for HI tables). Invalid half-keys store 0 — they are
    /// unreachable at runtime (see isValidLineKey).
    template <int H, bool NearAtLowBits, class T, int Scale>
    constexpr auto makeHalfCodeTable()
    {
        std::array<T, 1 << (2 * H)> table {};
        for (uint32_t half = 0; half < uint32_t(table.size()); half++) {
            int code    = denseHalfCode<H, NearAtLowBits>(half);
            table[half] = code < 0 ? T(0) : T(code * Scale);
        }
        return table;
    }

}  // namespace detail

/// Half-line key -> dense half code, one LO/HI pair per half-line width (the H=5 pair is
/// shared by standard and renju). LO tables are premultiplied by the DenseHalfCnt stride so
/// the full dense index is a plain add of two table reads.
static_assert((DenseHalfCnt<FREESTYLE> - 1) * DenseHalfCnt<FREESTYLE> <= UINT16_MAX,
              "premultiplied freestyle LO entries must fit uint16");
static_assert(DenseHalfCnt<STANDARD> - 1 <= UINT16_MAX);
alignas(64) inline constexpr auto HALF_CODE_LO_F =
    detail::makeHalfCodeTable<4, false, uint16_t, DenseHalfCnt<FREESTYLE>>();
alignas(64) inline constexpr auto HALF_CODE_HI_F =
    detail::makeHalfCodeTable<4, true, uint16_t, 1>();
alignas(64) inline constexpr auto HALF_CODE_LO_S =
    detail::makeHalfCodeTable<5, false, uint32_t, DenseHalfCnt<STANDARD>>();
alignas(64) inline constexpr auto HALF_CODE_HI_S =
    detail::makeHalfCodeTable<5, true, uint16_t, 1>();

/// Dense line code -> line pattern of both colors, one table per rule.
/// (`alignas` must precede `extern` — clang rejects the other order as applying to the type.)
alignas(64) extern Pattern2x PATTERN2x[DenseKeyCnt<FREESTYLE>];
alignas(64) extern Pattern2x PATTERN2xStandard[DenseKeyCnt<STANDARD>];
alignas(64) extern Pattern2x PATTERN2xRenju[DenseKeyCnt<RENJU>];
/// The four line patterns around a cell (one per direction) -> a single packed pattern code,
/// order-independent so the four directions can be supplied in any order.
extern PatternCode PCODE[PATTERN_NB][PATTERN_NB][PATTERN_NB][PATTERN_NB];
/// Dense line code + attacking color -> bitmask of cells that defend against the attack.
alignas(64) extern uint8_t DEFENCE[DenseKeyCnt<FREESTYLE>][2];
alignas(64) extern uint8_t DEFENCEStandard[DenseKeyCnt<STANDARD>][2];
alignas(64) extern uint8_t DEFENCERenju[DenseKeyCnt<RENJU>][2];

/// Debug helper: check the two invariants every runtime line key satisfies — the center
/// cell is empty, and each half is wall-contiguous. Keys violating either cannot be
/// produced by getKeyAt / the move() window for an in-board empty cell.
template <Rule R>
constexpr bool isValidLineKey(uint64_t key)
{
    constexpr int      H = HalfLineLen<R>;
    constexpr uint32_t M = (1u << (2 * H)) - 1;
    if (((key >> (2 * H)) & 0b11) != 0b11)
        return false;
    return detail::denseHalfCode<H, false>(uint32_t(key) & M) >= 0
        && detail::denseHalfCode<H, true>(uint32_t(key >> (2 * H + 2)) & M) >= 0;
}

/// Compute the dense table index from a raw 64-bit line key (layout: left half at bits
/// [0, 2H), center cell at [2H, 2H+2), right half at [2H+2, 4H+2)). The center is implicit
/// (always empty at lookup time); walls are folded into the boundary-run encoding.
template <Rule R>
inline uint32_t denseKey(uint64_t key)
{
    constexpr int      H = HalfLineLen<R>;
    constexpr uint32_t M = (1u << (2 * H)) - 1;
    assert(isValidLineKey<R>(key));

    uint32_t lo = uint32_t(key) & M;
    uint32_t hi = uint32_t(key >> (2 * H + 2)) & M;
    if constexpr (H == 4)
        return uint32_t(HALF_CODE_LO_F[lo]) + HALF_CODE_HI_F[hi];
    else
        return HALF_CODE_LO_S[lo] + HALF_CODE_HI_S[hi];
}

/// Look up the line pattern of both colors from a raw 64-bit line key.
template <Rule R>
inline Pattern2x lookupPattern(uint64_t key)
{
    static_assert(R == FREESTYLE || R == STANDARD || R == RENJU,
                  "incorrect rule to lookup pattern");

    uint32_t idx = denseKey<R>(key);

    if constexpr (R == Rule::FREESTYLE)
        return PATTERN2x[idx];
    else if constexpr (R == Rule::STANDARD)
        return PATTERN2xStandard[idx];
    else if constexpr (R == Rule::RENJU)
        return PATTERN2xRenju[idx];
}

/// Look up the defence-move bitmask for `attackSide` from a raw 64-bit line key. The table
/// carries a separate entry per attacking color rather than normalizing the key to a single
/// perspective.
template <Rule R>
inline uint8_t lookupDefenceTable(uint64_t key, Color attackSide)
{
    static_assert(R == FREESTYLE || R == STANDARD || R == RENJU,
                  "incorrect rule to lookup defence table");
    assert(attackSide == BLACK || attackSide == WHITE);

    uint32_t idx = denseKey<R>(key);

    if constexpr (R == Rule::FREESTYLE)
        return DEFENCE[idx][attackSide];
    else if constexpr (R == Rule::STANDARD)
        return DEFENCEStandard[idx][attackSide];
    else if constexpr (R == Rule::RENJU)
        return DEFENCERenju[idx][attackSide];
}

}  // namespace PatternConfig
