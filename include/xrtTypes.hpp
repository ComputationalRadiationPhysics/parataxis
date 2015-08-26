#pragma once

#include "simulation_defines.hpp"
#include <types.h>
#include <dimensions/DataSpace.hpp>
#include <dimensions/GridLayout.hpp>
#include <memory/buffers/GridBuffer.hpp>
#include <Environment.hpp>

namespace xrt {

    typedef PMacc::Environment<simDim> Environment;
    typedef PMacc::DataSpace<simDim> Space;
    typedef PMacc::DataSpace<2> Space2D;
    typedef PMacc::GridController<simDim> GC;
    typedef PMacc::GridBuffer< float_X, simDim > Buffer;
    typedef PMacc::GridLayout<simDim> GridLayout;
    typedef PMacc::SubGrid<simDim> SubGrid;

}  // namespace xrt

/**
 * Appends kernel arguments to generated code and activates kernel task.
 *
 * @param ... parameters to pass to kernel
 */
#define PIC_PMACC_CUDAPARAMS(...) (__VA_ARGS__, mapper);                       \
        PMACC_ACTIVATE_KERNEL                                                  \
    }   /*this is used if call is EventTask.waitforfinished();*/

/**
 * Configures block and grid sizes and shared memory for the kernel.
 *
 * gridSize for kernel call is set by mapper
 *
 * @param block size of block on GPU
 * @param ... amount of shared memory for the kernel (optional)
 */
#define PIC_PMACC_CUDAKERNELCONFIG(block, ...) <<<mapper.getGridDim(), (block), \
    __VA_ARGS__+0,                                                              \
    taskKernel->getCudaStream()>>> PIC_PMACC_CUDAPARAMS

/**
 * Calls a CUDA kernel and creates an EventTask which represents the kernel.
 *
 * Creates a AreaMapping mapper
 * gridSize for kernel call is set by mapper
 * last argument of kernel call is add by mapper and is the mapper
 *
 * @param kernelName name of the CUDA kernel (can also used with templates etc. myKernel<1>)
 * @param description cellDescription aka mapDescription
 * @param area area type for which the kernel is called
 */
#define __cudaKernelArea(kernelName, description, area) {                                                 \
    CUDA_CHECK_KERNEL_MSG(cudaDeviceSynchronize(), "picKernelArea crash before kernel call");             \
    PMacc::AreaMapping<area, MappingDesc> mapper(description);                                            \
    PMacc::TaskKernel *taskKernel =  PMacc::Environment<>::get().Factory().createTaskKernel(#kernelName); \
    kernelName PIC_PMACC_CUDAKERNELCONFIG
