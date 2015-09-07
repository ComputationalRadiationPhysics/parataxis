#pragma once

#include "xrtTypes.hpp"

#include <memory/buffers/GridBuffer.hpp>
#include <dataManagement/ISimulationData.hpp>

namespace xrt {
namespace detector {

    /**
     * Detector for photons that will accumulate incoming photons with the given policy
     * \tparam T_Config:
     *      policy IncomingParticleHandler_ The policy must define an inner type "Type"
     *          that is used for each "cell" of the detector and be a functor with
     *          signature Type(Type oldVal, Particle, uint32_t timeStep) that returns
     *          the new value for the detector cell
     *      float_X distance Distance from the volume in meters
     */
    template<class T_Config>
    class PhotonDetector: PMacc::ISimulationData
    {
        using Config = T_Config;
        using AccumPolicy = typename Config::IncomingParticleHandler;
        /**
         * Distance of the detector from the right side of the volume
         */
        static constexpr float_X distance = float_X(Config::distance / UNIT_LENGTH);

        using Type = typename AccumPolicy::Type;
        using Buffer = PMacc::GridBuffer< Type, 2 >;
        std::unique_ptr< Buffer > buffer;

        float_X xPosition_;

    public:

        PhotonDetector(const Space& simulationSize)
        {
            /* Detector is right to the volume
             * => Project from the left so bufX = -simZ && bufY == simY
             */
            Space2D size(
                    simulationSize.z(),
                    simulationSize.y()
                    );
            buffer.reset(new Buffer(size));
            xPosition_ = simulationSize.x() * CELL_WIDTH + distance;
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

        float_X
        getXPosition() const
        {
            return xPosition_;
        }
    };

}  // namespace detector
}  // namespace xrt
