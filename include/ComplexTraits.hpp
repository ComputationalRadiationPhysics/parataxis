#pragma once

#include "xrtTypes.hpp"

namespace PMacc{namespace mpi{namespace def{

    template<>
    struct GetMPI_StructAsArray<PMacc::math::Complex<float> >
    {
        MPI_StructAsArray operator()() const
        {
            return MPI_StructAsArray(MPI_FLOAT, 2);
        }
    };

    template<>
    struct GetMPI_StructAsArray<PMacc::math::Complex<double> >
    {
        MPI_StructAsArray operator()() const
        {
            return MPI_StructAsArray(MPI_DOUBLE, 2);
        }
    };

}}}
