#pragma once

#include "xrtTypes.hpp"

#include <memory/buffers/GridBuffer.hpp>
#include <dataManagement/ISimulationData.hpp>
#include <algorithms/math.hpp>

namespace xrt {
namespace detector {

    namespace detail {

        /**
         * Calculates the cell index on the detector that a photon reaches when it continues
         * with the current speed
         */
        template<class T_Config>
        struct GetTargetCellIdx
        {
            using Config = T_Config;
            static constexpr float_X cellWidth = float_X(Config::cellWidth / UNIT_LENGTH);
            static constexpr float_X cellHeight = float_X(Config::cellHeight / UNIT_LENGTH);

            /**
             *
             * @param xPosition Position of the detector in x direction
             * @param size      Size of the detector
             */
            GetTargetCellIdx(float_X xPosition, Space2D size):
                xPosition_(xPosition), size_(size), simSize_(Environment::get().SubGrid().getTotalDomain().size.shrink<2>(1))
            {}

            /**
             * Calculates the index on the detector and the time of impact
             * Returns true, if the detector was hit, false otherwise.
             * On false \ref targetIdx and \ref dt are undefined
             */
            template<typename T_Particle>
            HDINLINE bool
            operator()(const T_Particle& particle, Space globalIdx, Space2D& targetIdx, float_X& dt) const
            {
                auto mom = particle[momentum_];
                /* Not flying towards detector? -> exit */
                if(mom.x() <= 0)
                    return false;
                /* Calculate global position */
                const float_X momAbs = PMaccMath::abs(mom);
                const auto vel = mom * ( SPEED_OF_LIGHT / momAbs );
                float3_X pos;
                // This loop sets x,y,z for 3D and y,z for 2D
                for(uint32_t i=0; i<simDim; ++i)
                    pos[i + 3 - simDim] = (float_X(globalIdx[i]) + particle[position_][i]) * cellSize[i];
                /* Required time to reach detector */
                dt = (xPosition_ - pos.x()) / vel.x();
                /* Position at detector plane */
                pos.y() += dt * vel.y();
                pos.z() += dt * vel.z();
                using PMacc::algorithms::math::float2int_rn;
                /* We place so that the simulation volume is in the middle -->
                 * Offset = (DetectorSize - SimSize) / 2
                 */
                float_X xPos = pos.shrink<2>(1).x() + (cellWidth  * size_.x() + cellSize[simDim - 2] * simSize_.x()) / 2.f;
                float_X yPos = pos.shrink<2>(1).y() + (cellHeight * size_.y() + cellSize[simDim - 1] * simSize_.y()) / 2.f;
                targetIdx.x() = float2int_rn(xPos / cellWidth);
                targetIdx.y() = float2int_rn(yPos / cellHeight);
                /* Check bounds */
                return targetIdx.x() >= 0 && targetIdx.x() < size_.x() &&
                       targetIdx.y() >= 0 && targetIdx.y() < size_.y();
            }

        private:
            PMACC_ALIGN(xPosition_, const float_X);
            PMACC_ALIGN(size_, const Space2D);
            PMACC_ALIGN(simSize_, const Space2D);
        };

        template<class T_GetTargetCellIdx, class T_AccumPolicy>
        class DetectParticle
        {
            using GetTargetCellIdx = T_GetTargetCellIdx;
            using AccumPolicy = T_AccumPolicy;

            const GetTargetCellIdx getTargetCellIdx_;
            const AccumPolicy accumPolicy_;
            const uint32_t timeStep_;
        public:

            DetectParticle(GetTargetCellIdx getTargetCellIdx, AccumPolicy accumPolicy, uint32_t timeStep):
                getTargetCellIdx_(getTargetCellIdx), accumPolicy_(accumPolicy), timeStep_(timeStep)
            {}

            template<typename T_Particle, typename T_DetectorBox>
            HDINLINE void
            operator()(const T_Particle& particle, const Space superCellPosition, T_DetectorBox& detector) const
            {
                /*calculate global cell index*/
                const Space localCell(PMacc::DataSpaceOperations<simDim>::map<SuperCellSize>(particle[PMacc::localCellIdx_]));
                const Space globalCellIdx = superCellPosition + localCell;
                Space2D targetIdx;
                float_X dt;
                /* Get index on detector, if none found -> go out */
                if(!getTargetCellIdx_(particle, globalCellIdx, targetIdx, dt))
                    return;
                detector(targetIdx) = accumPolicy_(detector(targetIdx), particle, timeStep_ * DELTA_T + dt);
            }
        };

    }  // namespace detail

    /**
     * Detector for photons that will accumulate incoming photons with the given policy
     * \tparam T_Config:
     *      policy IncomingParticleHandler_ The policy must define an inner type "Type"
     *          that is used for each "cell" of the detector and be a functor with
     *          signature Type(Type oldVal, Particle, float_X timeAtDetector) that returns
     *          the new value for the detector cell
     *      float_X distance Distance from the volume in meters
     */
    template<class T_Config, class T_Species>
    class PhotonDetectorImpl: PMacc::ISimulationData
    {
        using Config = T_Config;
        using Species = T_Species;
        using IncomingParticleHandler = Resolve_t<typename Config::IncomingParticleHandler>;
        using AccumPolicy = typename bmpl::apply<IncomingParticleHandler, Species>::type;
        /**
         * Distance of the detector from the right side of the volume
         */
        static constexpr float_X distance = float_X(Config::distance / UNIT_LENGTH);

    public: using Type = typename AccumPolicy::Type;
    private:
        using Buffer = PMacc::GridBuffer< Type, 2 >;
        std::unique_ptr< Buffer > buffer;

        float_X xPosition_;

    public:

        using DetectParticle = detail::DetectParticle<detail::GetTargetCellIdx<Config>, AccumPolicy>;

        PhotonDetectorImpl(const Space2D& size)
        {
            buffer.reset(new Buffer(size));
            Space simSize = Environment::get().SubGrid().getTotalDomain().size;
            xPosition_ = simSize.x() * CELL_WIDTH + distance;
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

        typename Buffer::DataBoxType
        getHostDataBox()
        {
            return buffer->getHostBuffer().getDataBox();
        }

        typename Buffer::DataBoxType
        getDeviceDataBox()
        {
            return buffer->getDeviceBuffer().getDataBox();
        }

        Space2D
        getSize() const
        {
            return buffer->getGridLayout().getDataSpaceWithoutGuarding();
        }

        DetectParticle
        getDetectParticle(uint32_t timeStep) const
        {
            return DetectParticle(detail::GetTargetCellIdx<Config>(xPosition_, getSize()), AccumPolicy(), timeStep);
        }
    };

}  // namespace detector
}  // namespace xrt
