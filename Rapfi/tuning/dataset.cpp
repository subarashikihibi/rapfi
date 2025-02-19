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

#include "dataset.h"

#include "../core/iohelper.h"
#include "../core/utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <lz4Stream.hpp>
#include <npy.hpp>
#include <unordered_set>

namespace {

/// Unpacks a byte array into bit array (in big-endian).
/// @param bytes The source of byte array
/// @param numBits number of bits to unpack
/// @param bits Destination of bits array
void unpackBytesToBits(const uint8_t *bytes, size_t numBits, uint8_t *bits)
{
    size_t numBytesFloored = numBits / 8;
    size_t numBitsRemained = numBits % 8;

    for (size_t byteIdx = 0; byteIdx < numBytesFloored; byteIdx++) {
        uint8_t byte = *bytes++;
        for (int i = 0; i < 8; i++)
            bits[i] = (byte >> (7 - i)) & 0x1;
        bits += 8;
    }

    // Deals with remaining bits that less than a byte
    uint8_t byte = *bytes;
    for (size_t bitIdx = 0; bitIdx < numBitsRemained; bitIdx++)
        bits[bitIdx] = (byte >> (7 - bitIdx)) & 0x1;
}

/// Converting a board array to a pos sequence (in arbitrary order).
void boardArrayToPosSequence(const std::vector<Color> &boardArray,
                             int                       boardSize,
                             std::vector<Pos>         &posSequence)
{
    std::vector<Pos> blackPos, whitePos;
    for (size_t i = 0; i < boardArray.size(); i++) {
        Pos pos(i % boardSize, i / boardSize);
        switch (boardArray[i]) {
        case BLACK: blackPos.push_back(pos); break;
        case WHITE: whitePos.push_back(pos); break;
        default: break;
        }
    }

    assert((int)blackPos.size() - (int)whitePos.size() <= 1);
    posSequence.clear();
    size_t numCommonPos = std::min(blackPos.size(), whitePos.size());
    for (size_t i = 0; i < numCommonPos; i++) {
        posSequence.push_back(blackPos[i]);
        posSequence.push_back(whitePos[i]);
    }

    // Black might have one more move than white
    if (blackPos.size() > numCommonPos)
        posSequence.push_back(blackPos[numCommonPos]);
}

}  // namespace

namespace Tuning {

class PackedBinaryDataset::DataSource
{
public:
    DataSource(std::vector<std::ifstream> &&fileStreams)
        : files(std::move(fileStreams))
        , nextIdx(0)
        , compressor(nullptr)
        , istream(nullptr)
    {
        next();
    }

    ~DataSource() = default;

    /// Goto the next file in the file list.
    /// @return False when curIdx reaches the end, otherwise true.
    bool next()
    {
        if (nextIdx == files.size())
            return false;

        // Delete previous compressor if exists
        if (compressor) {
            istream = nullptr;
            compressor.reset();
        }

        // fileStream will be set std::istream::badbit
        int magic;
        files[nextIdx].read(reinterpret_cast<char *>(&magic), sizeof(magic));
        files[nextIdx].seekg(0);

        // Check LZ4 magic
        compressor = std::make_unique<Compressor>(
            files[nextIdx],
            magic == 0x184D2204 ? Compressor::Type::LZ4_DEFAULT : Compressor::Type::NO_COMPRESS);
        istream = compressor->openInputStream();
        nextIdx++;

        if (!istream)
            throw std::runtime_error("unable to load dataset stream");

        return true;
    }

    /// Reset the state of data source to its initial state.
    void reset()
    {
        istream = nullptr;
        if (compressor)
            compressor.reset();
        for (std::ifstream &fs : files)
            fs.seekg(0);
        nextIdx = 0;
        next();
    }

    std::istream &getStream()
    {
        assert(istream);
        return *istream;
    }

private:
    std::vector<std::ifstream>  files;
    size_t                      nextIdx;
    std::unique_ptr<Compressor> compressor;
    std::istream               *istream;
};

PackedBinaryDataset::PackedBinaryDataset(const std::vector<std::string> &filenames)
{
    if (filenames.empty())
        throw std::runtime_error("no file in binary dataset");

    std::vector<std::ifstream> fileStreams;

    for (const std::string &filename : filenames) {
        std::ifstream fileStream(filename, std::ios::binary);
        if (!fileStream.is_open())
            throw std::runtime_error("unable to open file " + filename);

        fileStream.exceptions(std::istream::badbit | std::istream::failbit);
        fileStreams.push_back(std::move(fileStream));
    }

    dataSource = std::make_unique<DataSource>(std::move(fileStreams));
}

PackedBinaryDataset::~PackedBinaryDataset() {}

bool PackedBinaryDataset::next(DataEntry *entry)
{
    struct EntryHead
    {
        uint16_t result : 2;     // game outcome: 0=loss, 1=draw, 2=win (side to move pov)
        uint16_t ply : 9;        // current number of stones on board
        uint16_t boardsize : 5;  // board size in [5-22]
        uint16_t rule : 3;       // game rule: 0=freestyle, 1=standard, 4=renju
        uint16_t move : 13;      // move output by the engine
    } ehead;
    uint16_t position[MAX_MOVES];  // move sequence that representing a position

    // Check if current stream has reached its EOF, if so proceeds to the next one
    if (dataSource->getStream().eof()
        || dataSource->getStream().peek() == std::ios::traits_type::eof()) {
        if (!dataSource->next())
            return false;
    }

    std::istream &src = dataSource->getStream();

    // Read and process entry header first
    src.read(reinterpret_cast<char *>(&ehead), sizeof(EntryHead));

    // Check legality of entryhead
    if (ehead.boardsize == 0)
        throw std::runtime_error("wrong boardsize in dataset");
    if (ehead.rule != 0 && ehead.rule != 1 && ehead.rule != 4)
        throw std::runtime_error("wrong rule in dataset");
    if (ehead.result != 0 && ehead.result != 1 && ehead.result != 2)
        throw std::runtime_error("wrong result in dataset");
    if (ehead.ply > ehead.boardsize * ehead.boardsize)
        throw std::runtime_error("wrong ply in dataset");

    if (entry) {
        entry->boardsize = ehead.boardsize;
        entry->rule      = ehead.rule == 4 ? RENJU : Rule(ehead.rule);
        entry->result    = Result(ehead.result);
        entry->position.clear();
        entry->position.reserve(ehead.ply);

        // Read position move sequence according the ply in header
        src.read(reinterpret_cast<char *>(&position), ehead.ply * sizeof(uint16_t));

        std::unordered_set<Pos> movedPos;
        movedPos.reserve(MAX_MOVES);

        /// Each move is represented by a 16bit unsigned integer. It's lower 10 bits are
        /// constructed with two index x and y using uint16_t move = (x << 5) | y.
        const auto coordX = [](uint16_t move) { return (move >> 5) & 0x1f; };
        const auto coordY = [](uint16_t move) { return move & 0x1f; };

        for (uint16_t ply = 0; ply < ehead.ply; ply++) {
            int x = coordX(position[ply]);
            int y = coordY(position[ply]);
            Pos pos {x, y};

            if (x < 0 || y < 0 || x >= ehead.boardsize || y >= ehead.boardsize)
                throw std::runtime_error("wrong move sequence in dataset ([" + std::to_string(x)
                                         + "," + std::to_string(y) + "] in boardsize "
                                         + std::to_string(ehead.boardsize) + ")");
            else if (movedPos.find(pos) != movedPos.end()) {
                std::stringstream ss;
                ss << "duplicate move in sequence ([" << pos << "], current sequence ["
                   << entry->position << "])";
                throw std::runtime_error(ss.str());
            }

            movedPos.emplace(pos);
            entry->position.push_back(pos);
        }

        int bestX = coordX(ehead.move);
        int bestY = coordY(ehead.move);
        Pos bestMove {bestX, bestY};
        if (bestX == ehead.boardsize && bestY == ehead.boardsize)
            ;  // Allow special marked no-bestmove
        else if (bestX < 0 || bestY < 0 || bestX >= ehead.boardsize || bestY >= ehead.boardsize
                 || movedPos.find(bestMove) != movedPos.end())
            throw std::runtime_error("wrong best move in dataset ([" + std::to_string(bestX) + ","
                                     + std::to_string(bestY) + "] in boardsize "
                                     + std::to_string(ehead.boardsize) + ")");

        entry->move = bestMove;
    }
    else {
        // Just skip those position move sequence
        src.ignore(ehead.ply * sizeof(uint16_t));
    }

    return true;
}

void PackedBinaryDataset::reset()
{
    dataSource->reset();
}

// ==============================================

class KatagoNumpyDataset::DataSource
{
public:
    struct RawDataEntry
    {
        Color                sideToMove;
        std::vector<Color>   boardInput;
        std::array<float, 3> valueTarget;
        std::vector<int16_t> policyTarget;
    };

    DataSource(std::vector<std::string> filenames)
        : filenames(std::move(filenames))
        , nextFileIdx(0)
        , nextEntryIdx(0)
    {
        nextFile();
    }
    ~DataSource() = default;

    /// Goto the next file in the file list.
    /// @return False when file list reaches the end, otherwise true.
    bool nextFile()
    {
        nextEntryIdx = 0;

        // If we have reached the end of file list
        if (nextFileIdx == filenames.size())
            return false;

        std::ifstream fileStream(filenames[nextFileIdx], std::ios::binary);
        if (!fileStream.is_open())
            throw std::runtime_error("unable to open file " + filenames[nextFileIdx]);

        // fileStream will be set std::istream::badbit
        fileStream.exceptions(std::istream::badbit | std::istream::failbit);

        // Open .npz with ZIP
        Compressor compressor(fileStream, Compressor::Type::ZIP_DEFAULT);

        auto openEntryThen = [&](std::string entryName,
                                 bool (DataSource::*receiver)(std::istream & is)) {
            std::istream *is = compressor.openInputStream(entryName);
            if (!is)
                throw std::runtime_error("unable to open " + entryName + " in file "
                                         + filenames[nextFileIdx]);
            if (!(this->*receiver)(*is))
                throw std::runtime_error("incorrect data in " + entryName + " in file "
                                         + filenames[nextFileIdx]);
            compressor.closeStream(*is);
        };

        openEntryThen("globalInputNC", &DataSource::readSideToMove);
        openEntryThen("binaryInputNCHWPacked", &DataSource::readBoardInput);
        openEntryThen("globalTargetsNC", &DataSource::readValueTarget);
        openEntryThen("policyTargetsNCMove", &DataSource::readPolicyTarget);

        nextFileIdx++;
        return true;
    }

    /// Get the next raw data entry.
    /// @return False when entry list reaches the end, otherwise true.
    bool nextEntry(RawDataEntry &rawDataEntry)
    {
        if (nextEntryIdx == sideToMove.size())
            return false;

        rawDataEntry.sideToMove   = sideToMove[nextEntryIdx];
        rawDataEntry.boardInput   = std::move(boardInput[nextEntryIdx]);
        rawDataEntry.valueTarget  = std::move(valueTarget[nextEntryIdx]);
        rawDataEntry.policyTarget = std::move(policyTarget[nextEntryIdx]);
        nextEntryIdx++;
        return true;
    }

    /// Reset the state of data source to its initial state.
    void reset()
    {
        nextFileIdx = nextEntryIdx = 0;
        sideToMove.clear();
        boardInput.clear();
        valueTarget.clear();
        policyTarget.clear();

        nextFile();
    }

private:
    std::vector<std::string> filenames;
    size_t                   nextFileIdx;
    size_t                   nextEntryIdx;

    std::vector<Color>                sideToMove;    // [N]
    std::vector<std::vector<Color>>   boardInput;    // [N, HW]
    std::vector<std::array<float, 3>> valueTarget;   // [N, 3] win, loss, draw
    std::vector<std::vector<int16_t>> policyTarget;  // [N, HW]

    // Read globalInputNC into (sideToMove)
    bool readSideToMove(std::istream &is)
    {
        // Read ndarray [N, C] float
        std::vector<unsigned long> shape;
        std::vector<float>         data;
        npy::LoadArrayFromNumpy(is, shape, data);
        if (shape.size() != 2)
            return false;

        size_t length      = shape[0];
        int    numChannels = shape[1];

        sideToMove.resize(length);
        for (std::size_t i = 0; i < length; i++) {
            // Channel 5: side to move (black = -1.0, white = 1.0)
            float stmInput = data[i * numChannels + 5];
            sideToMove[i]  = (stmInput < 0 ? BLACK : WHITE);
        }

        return true;
    }

    // Read binaryInputNCHWPacked
    bool readBoardInput(std::istream &is)
    {
        // Read ndarray [N, C, ceil(H*W/8)] uint8
        std::vector<unsigned long> shape;
        std::vector<uint8_t>       data;
        npy::LoadArrayFromNumpy(is, shape, data);
        if (shape.size() != 3)
            return false;

        size_t length      = shape[0];
        int    numChannels = shape[1];
        int    numBytes    = shape[2];
        int    boardSize   = (int)std::sqrt(numBytes * 8);
        int    numCells    = boardSize * boardSize;

        boardInput.resize(length);
        std::vector<uint8_t> boardSelfBits(numCells);
        std::vector<uint8_t> boardOppoBits(numCells);
        uint8_t             *boardSelfBytes = &data[1 * numBytes];  // Channel 1: next player stones
        uint8_t             *boardOppoBytes = &data[2 * numBytes];  // Channel 2: opponent stones
        size_t               stride         = numChannels * numBytes;
        for (std::size_t i = 0; i < length; i++) {
            unpackBytesToBits(boardSelfBytes, numCells, boardSelfBits.data());
            unpackBytesToBits(boardOppoBytes, numCells, boardOppoBits.data());

            // Fill board input according to unpacked bits
            boardInput[i].resize(numCells);
            Color stm = sideToMove[i];
            for (std::size_t j = 0; j < numCells; j++)
                boardInput[i][j] = boardSelfBits[j] ? stm : boardOppoBits[j] ? ~stm : EMPTY;

            // Goto next entry
            boardSelfBytes += stride;
            boardOppoBytes += stride;
        }

        return true;
    }

    // Read globalTargetsNC
    bool readValueTarget(std::istream &is)
    {
        // Read ndarray [N, C] float
        std::vector<unsigned long> shape;
        std::vector<float>         data;
        npy::LoadArrayFromNumpy(is, shape, data);
        if (shape.size() != 2)
            return false;

        size_t length      = shape[0];
        int    numChannels = shape[1];

        valueTarget.resize(length);
        for (std::size_t i = 0; i < length; i++) {
            valueTarget[i][0] = data[i * numChannels + 0];  // Channel 0: win prob
            valueTarget[i][1] = data[i * numChannels + 1];  // Channel 1: loss prob
            valueTarget[i][2] = data[i * numChannels + 2];  // Channel 2: draw prob
        }

        return true;
    }

    // Read policyTargetsNCMove
    bool readPolicyTarget(std::istream &is)
    {
        // Read ndarray [N, C, Pos] int16
        std::vector<unsigned long> shape;
        std::vector<int16_t>       data;
        npy::LoadArrayFromNumpy(is, shape, data);
        if (shape.size() != 3)
            return false;

        size_t length      = shape[0];
        int    numChannels = shape[1];
        int    numCells    = shape[2] - 1;

        // Read policy target without normalize (do that when actually needed)
        policyTarget.resize(length);
        size_t stride = numChannels * (numCells + 1);
        for (std::size_t i = 0; i < length; i++) {
            policyTarget[i].resize(numCells);
            std::copy_n(&data[i * stride], numCells, policyTarget[i].data());
        }

        return true;
    }
};

KatagoNumpyDataset::KatagoNumpyDataset(const std::vector<std::string> &filenames, Rule rule)
    : defaultRule(rule)
{
    if (filenames.empty())
        throw std::runtime_error("no file in katago numpy dataset");

    // Check all file legality
    for (const std::string &filename : filenames) {
        std::ifstream fileStream(filename, std::ios::binary);
        if (!fileStream.is_open())
            throw std::runtime_error("unable to open file " + filename);
    }

    dataSource = std::make_unique<DataSource>(filenames);
}

KatagoNumpyDataset::~KatagoNumpyDataset() {}

bool KatagoNumpyDataset::next(DataEntry *entry)
{
    KatagoNumpyDataset::DataSource::RawDataEntry rawDataEntry;

    // Check if we reached the end of entry list, if so proceeds to the next file
    while (!dataSource->nextEntry(rawDataEntry)) {
        // Check if we reached the end of file list, if so we have completed the whole dataset
        if (!dataSource->nextFile())
            return false;
    }

    if (entry) {
        int numCells  = rawDataEntry.boardInput.size();
        int boardSize = (int)std::sqrt(numCells);  // square board

        boardArrayToPosSequence(rawDataEntry.boardInput, boardSize, entry->position);
        entry->boardsize = boardSize;
        entry->rule      = defaultRule;
        entry->result    = rawDataEntry.valueTarget[0] > 0   ? RESULT_WIN
                           : rawDataEntry.valueTarget[1] > 0 ? RESULT_LOSS
                                                             : RESULT_DRAW;

        // Create and normalize policy target
        auto  policy         = std::make_unique<float[]>(numCells);
        float policySum      = 0.0f;
        float policyMax      = std::numeric_limits<float>::min();
        int   maxPolicyIndex = -1;
        for (int i = 0; i < numCells; i++) {
            policy[i] = (float)rawDataEntry.policyTarget[i];
            policySum += policy[i];
            if (policy[i] > policyMax) {
                policyMax      = policy[i];
                maxPolicyIndex = i;
            }
        }
        float invPolicySum = 1.0f / (policySum + 1e-7);
        for (int i = 0; i < numCells; i++)
            policy[i] *= invPolicySum;

        entry->policy = std::move(policy);
        entry->move   = Pos(maxPolicyIndex % boardSize, maxPolicyIndex / boardSize);
    }

    return true;
}

void KatagoNumpyDataset::reset()
{
    dataSource->reset();
}

}  // namespace Tuning
