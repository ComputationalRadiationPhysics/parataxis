#include "mallocMCConfig.hpp"
#include "xrtTypes.hpp"
#include <mpi.h>

int main( int argc, char **argv )
{
    MPI_CHECK(MPI_Init(&argc, &argv));

    int errorCode;
    {
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
    }

    MPI_CHECK(MPI_Finalize());
    return errorCode;
}
