#pragma once

#include "xrtTypes.hpp"
#include "random/RNGState.hpp"
#include "random/methods/XorMin.hpp"
#if XRT_USE_MAPPED_RNG_BUFFER
#   include <memory/buffers/MappedBufferIntern.hpp>
#else
#   include <memory/buffers/GridBuffer.hpp>
#endif
#include <dataManagement/ISimulationData.hpp>
#include <memory>

namespace xrt {
namespace random {

    /**
     * Provider of a per cell random number generator
     */
    class RNGProvider: PMacc::ISimulationData
    {
    public:
        typedef methods::XorMin RNGMethod;
#if XRT_USE_MAPPED_RNG_BUFFER
        typedef PMacc::MappedBufferIntern< RNGState<RNGMethod>, simDim > Buffer;
#else
        typedef PMacc::GridBuffer< RNGState<RNGMethod>, simDim > Buffer;
#endif
        typedef typename Buffer::DataBoxType DataBoxType;

    private:
        MappingDesc cellDescription;
        std::unique_ptr<Buffer> buffer;

    public:

        RNGProvider(const MappingDesc& desc);
        /**
         * Initializes the random number generators
         * Must be called before usage
         * @param seed
         */
        void init(uint32_t seed);

        /**
         * Gets the device data box
         */
        DataBoxType getDeviceDataBox();

        static std::string getName();
        PMacc::SimulationDataId getUniqueId() override;
        void synchronize() override;
    };

}  // namespace random
}  // namespace xrt
