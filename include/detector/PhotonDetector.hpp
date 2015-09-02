#pragma once

#include "xrtTypes.hpp"

#include <memory/buffers/GridBuffer.hpp>
#include <dataManagement/ISimulationData.hpp>

namespace xrt {
namespace detector {

    /**
     * Detector for photons that will accumulate incoming photons with the given policy
     * The policy must define an inner type "Type" that is used for each "cell" of the detector
     * and be a functor with signature Type(Type oldVal, Particle, uint32_t timeStep) that returns
     * the new value for the detector cell
     */
    template<class T_AccumPolicy>
    class PhotonDetector: PMacc::ISimulationData
    {
        using AccumPolicy = T_AccumPolicy;
        using Type = typename AccumPolicy::Type;
        using Buffer = PMacc::GridBuffer< Type, 2 >;
        std::unique_ptr< Buffer > buffer;

    public:

        PhotonDetector(const Space& simulationSize)
        {
            /* Detector is right to the buffer
             * => Project from the left so bufX = -simZ && bufY == simY
             */
            Space2D size(
                    simulationSize.z(),
                    simulationSize.y()
                    );
            buffer.reset(new Buffer(size));
        }

        static std::string
        getName()
        {
            return "PhotonDetector";
        }

        PMacc::SimulationDataId getUniqueId() override
        {
            return getName();
        }

        void synchronize() override
        {
            buffer->deviceToHost();
        }

        void init()
        {
            Environment::get().DataConnector().registerData(*this);
        }
    };

}  // namespace detector
}  // namespace xrt
