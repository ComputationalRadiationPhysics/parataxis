#include "include/xrtTypes.hpp"
#include "Simulation.hpp"

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

#include <iostream>

#include <mpi.h>
#include <communication/manager_common.h>

namespace po = boost::program_options;

int main( int argc, char **argv )
{

    MPI_CHECK( MPI_Init( &argc, &argv ) );

    typedef xrt::Space Space;

    std::vector<uint32_t> devices;  /* will be set by boost program argument option "-d 3 3 3" */
    std::vector<uint32_t> gridSize; /* same but with -g */
    std::vector<bool> periodic;
    uint32_t steps;

    po::options_description desc( "Allowed options" );
    desc.add_options( )
            ( "help,h", "produce help message" )
            ( "steps,s", po::value( &steps ), "simulation steps" )
            ( "devices,d", po::value( &devices )->multitoken( ),
              "number of devices in each dimension (only 1D or 2D). If you use more than "
              "one device in total, you will need to run mpirun with \"mpirun -n "
              "<DeviceCount.x*DeviceCount.y> ./xRayTracing\"" )
            ( "grid,g", po::value( &gridSize )->multitoken( ),
              "size of the simulation grid (must be 2D, e.g.: -g 128 128). Because of the border, which is one supercell = 16 cells wide, "
              "the size in each direction should be greater or equal than 3*16=48 per device, so that the core will be non-empty" )
            ( "periodic,p", po::value( &periodic )->multitoken( ),
              "specifying whether the grid is periodic (1) or not (0) in each dimension, default: no periodic dimensions" );

    /* parse command line options and store values in vm */
    po::variables_map vm;
    po::store( po::parse_command_line( argc, argv, desc ), vm );
    po::notify( vm );

    /* print help message and quit simulation */
    if ( vm.count( "help" ) )
    {
        MPI_CHECK( MPI_Finalize() );
        std::cerr << desc << "\n";
        return false;
    }


    /* fill periodic with 0 */
    while ( periodic.size() < xrt::SIMDIM )
        periodic.push_back(false);

    /* check on correct number of devices. fill with default value 1 for missing dimensions */
    if ( devices.size() > xrt::SIMDIM )
        std::cerr << "Invalid number of devices.\nUse [-d dx=1 dy=1 dz=1]" << std::endl;
    else
        while ( devices.size() < xrt::SIMDIM )
            devices.push_back(1);

    /* check on correct grid size. fill with default grid size value 1 for missing 3. dimension */
    if ( gridSize.size() != xrt::SIMDIM )
    {
        std::cerr << "Invalid or missing grid size.\nUse -g width height [depth=1]" << std::endl;
        MPI_CHECK( MPI_Finalize() );
        return 0;
    }


    /* after checking all input values, copy into DataSpace Datatype */
    Space gpus( devices[0], devices[1] );
    Space grid( gridSize[0], gridSize[1] );
    Space endless( periodic[0], periodic[1] );

    {
        /* start simulation
         * Use extra scope to have it destroyed before the MPI_Finalize
         */
        xrt::Simulation sim( steps, grid, gpus, endless );
        sim.init();
        sim.start();
    }

    MPI_CHECK( MPI_Finalize( ) );
    return 0;
}
