#pragma once

#include "xrtTypes.hpp"
#include "RNGState.hpp"
#include <memory/buffers/GridBuffer.hpp>
#include <dataManagement/ISimulationData.hpp>
#include <nvidia/rng/methods/Xor.hpp>
#include <memory>

namespace xrt {
namespace random {

    /**
     * Provider of a per cell random number generator
     */
    class RNGProvider: PMacc::ISimulationData
    {
    public:
        typedef nvrng::methods::Xor RNGMethod;
        typedef PMacc::GridBuffer< RNGState<RNGMethod>, simDim > Buffer;
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
