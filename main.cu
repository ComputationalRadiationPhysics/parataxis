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
 
#include "mallocMCConfig.hpp"
#include "xrtTypes.hpp"
#include <mpi.h>

int main( int argc, char **argv )
{
    MPI_CHECK(MPI_Init(&argc, &argv));

    int errorCode;
    try{
        /* Use nested region to make sure all simulation classes are
         * freed before we call MPI_Finalize
         */
        xrt::SimStarter starter;
        xrt::ArgsErrorCode parserCode = starter.parseConfigs(argc, argv);

        switch(parserCode)
        {
            case xrt::ArgsErrorCode::ERROR:
                errorCode = 1;
                break;
            case xrt::ArgsErrorCode::SUCCESS:
                starter.load();
                starter.start();
                starter.unload();
                errorCode = 0;
                break;
            case xrt::ArgsErrorCode::SUCCESS_EXIT:
                errorCode = 0;
                break;
            default:
                std::cerr << "Unhandled parser code: " << int(parserCode) << std::endl;
                errorCode = 99;
        };
    }catch(std::exception& e)
    {
        std::cerr << "Unhandled exception occurred: " << e.what() << std::endl;
        std::cout << "Unhandled exception occurred: " << e.what() << std::endl;
        errorCode = 2;
    }

    MPI_CHECK(MPI_Finalize());
    return errorCode;
}
