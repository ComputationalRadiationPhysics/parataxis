#define BOOST_NO_SFINAE_EXPR
#include <boost/utility/result_of.hpp>
#include "mallocMCConfig.hpp"
#include "include/xrtTypes.hpp"
#include "simulationControl/SimulationStarter.hpp"
#include <mpi.h>

int main( int argc, char **argv )
{
    MPI_CHECK(MPI_Init(&argc, &argv));

    int errorCode;
    {
        /* Use nested region to make sure all simulation classes are
         * freed before we call MPI_Finalize
         */
        xrt::SimulationStarter sim;
        xrt::ArgsErrorCode parserCode = sim.parseConfigs(argc, argv);

        switch(parserCode)
        {
            case xrt::ArgsErrorCode::ERROR:
                errorCode = 1;
                break;
            case xrt::ArgsErrorCode::SUCCESS:
                sim.load();
                sim.start();
                sim.unload();
                errorCode = 0;
                break;
            case xrt::ArgsErrorCode::SUCCESS_EXIT:
                errorCode = 0;
                break;
        };
    }

    MPI_CHECK(MPI_Finalize());
    return errorCode;
}
