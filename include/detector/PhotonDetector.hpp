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

    /**
     * Calculates the cell index on the detector that a photon reaches when it continues
     * with the current speed
     */
    struct GetTargetCellIdx
    {
        /**
         *
         * @param xPosition Position of the detector in x direction
         * @param size      Size of the detector
         * @param globalIdx Global index of the current super cell
         */
        GetTargetCellIdx(float_X xPosition, Space2D size, Space globalIdx):
            xPosition_(xPosition), size_(size), globalIdx_(globalIdx)
        {}

        template<typename T_Particle>
        HDINLINE bool
        operator()(const T_Particle& particle, Space2D& targetIdx, float_X& dt)
        {
            auto mom = particle[momentum_];
            /* Not flying towards detector? -> exit */
            if(mom.x() <= 0)
                return false;
            /* Calculate global position */
            const float_X momAbs = PMaccMath::abs(mom);
            const floatD_X vel   = mom * ( SPEED_OF_LIGHT / momAbs );
            floatD_X pos;
            for(uint32_t i=0; i<simDim; ++i)
                pos[i] = (float_X(globalIdx_[i]) + particle[position_][i]) * cellSize[i];
            /* Required time to reach detector */
            dt = (xPosition_ - pos.x()) / vel.x();
            /* Position at detector plane */
            pos.z() += dt * vel.z();
            pos.y() += dt * vel.y();
            targetIdx.x() = pos.z() / CELL_DEPTH;
            targetIdx.y() = pos.y() / CELL_HEIGHT;
            /* Check bounds */
            return targetIdx.x() >= 0 && targetIdx.x() < size_.x() &&
                   targetIdx.y() >= 0 && targetIdx.y() < size_.y();
        }

    private:
        PMACC_ALIGN(xPosition_, const float_X);
        PMACC_ALIGN(size_, const Space2D);
        PMACC_ALIGN(globalIdx_, const Space);
    };

}  // namespace detector
}  // namespace xrt
