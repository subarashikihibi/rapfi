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

#include "dbconfig.h"

#include "../core/iohelper.h"
#include "../core/utils.h"
#include "yxdbstorage.h"

#include <filesystem>

namespace Database {

DatabaseConfig DatabaseCfg;

std::unique_ptr<DBStorage> openDBStorage(const DatabaseConfig &cfg, std::string utf8URL)
{
    if (cfg.type == "yixindb") {
        auto dbPath = std::filesystem::u8path(utf8URL.empty() ? cfg.url : utf8URL);
        return std::make_unique<YXDBStorage>(dbPath,
                                             cfg.yixindb.compressedSave,
                                             cfg.yixindb.saveOnClose,
                                             cfg.yixindb.numBackupsOnSave,
                                             cfg.yixindb.ignoreCorrupted);
    }
    throw std::invalid_argument("unknown database type " + cfg.type);
}

std::unique_ptr<DBStorage> createDBStorage(const DatabaseConfig &cfg, std::string utf8URL)
{
    // Silent nullptr reproduces the old null-DatabaseMaker sentinel exactly: a
    // maker existed iff the last parse got through the yixindb options. Empty
    // type, unknown-type leftovers from a failed parse (readDatabase throws
    // AFTER storing type), and a parse that threw mid-[database] must all stay
    // silent here.
    if (!cfg.factoryEnabled)
        return nullptr;

    // The old maker was a self-contained yixindb closure: a failed reload may
    // have overwritten cfg.type before throwing, yet the live factory kept
    // opening yixindb with the last-committed options. Force the type so the
    // core's dispatch cannot see that transient state (its unknown-type throw
    // stays CLI-only reachable).
    DatabaseConfig liveCfg = cfg;
    liveCfg.type           = "yixindb";

    try {
        auto dbPath    = std::filesystem::u8path(utf8URL.empty() ? liveCfg.url : utf8URL);
        bool existing  = std::filesystem::exists(dbPath);
        auto startTime = now();
        MESSAGEL("Opening yixin database at " << pathToConsoleString(dbPath) << " ...");
        auto dbStorage = openDBStorage(liveCfg, std::move(utf8URL));
        if (existing)
            MESSAGEL("Loaded Yixin database (" << dbStorage->size() << " records) using "
                                               << (now() - startTime) << " ms.");
        return dbStorage;
    }
    catch (const std::exception &e) {
        ERRORL("Failed to create yixin database: " << e.what());
        return nullptr;
    }
}

}  // namespace Database
