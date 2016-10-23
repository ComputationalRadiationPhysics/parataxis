/**
 * Copyright 2015-2016 Alexander Grund
 *
 * This file is part of ParaTAXIS.
 *
 * ParaTAXIS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ParaTAXIS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ParaTAXIS.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#pragma once

#include <stdint.h>
#include <debug/VerboseLogMakros.hpp>

namespace parataxis{

    #ifndef PARATAXIS_VERBOSE_LVL
    #   define PARATAXIS_VERBOSE_LVL 0
    #endif

    /*create verbose class*/
    DEFINE_VERBOSE_CLASS(PARATAXISLogLvl)
    (
        /* define log lvl for later use
         * e.g. log<PMaccLogLvl::NOTHING>("TEXT");*/
        DEFINE_LOGLVL(0, NOTHING);
        DEFINE_LOGLVL(1, SIM_STATE);
        DEFINE_LOGLVL(2, MEMORY);
        DEFINE_LOGLVL(4, DOMAINS);
        DEFINE_LOGLVL(8, IN_OUT);
        DEFINE_LOGLVL(16, PLUGINS);
        DEFINE_LOGLVL(32, TIMING);
        DEFINE_LOGLVL(64, DEBUG);
    )
    /*set default verbose lvl (integer number)*/
    (NOTHING::lvl|PARATAXIS_VERBOSE_LVL);


} // namespace parataxis
