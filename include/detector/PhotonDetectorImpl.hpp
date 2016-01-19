#pragma once

#include "xrtTypes.hpp"
#include "particles/functors/GetWavelength.hpp"
#include "debug/LogLevels.hpp"

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
            GetTargetCellIdx(float_64 xPosition, Space2D size):
                xPosition_(xPosition), size_(size), simSize_(Environment::get().SubGrid().getTotalDomain().size.shrink<2>(1))
            {}

            /**
             * Calculates the index on the detector and the time of impact
             * Returns true, if the detector was hit, false otherwise.
             * On false \ref targetIdx and \ref dt are undefined
             */
            template<typename T_Particle>
            HDINLINE bool
            operator()(const T_Particle& particle, Space globalIdx, Space2D& targetIdx, float_64& dt) const
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
                using PMacc::algorithms::math::float2int_rd;
                /* We place so that the simulation volume is in the middle -->
                 * Offset = (DetectorSize - SimSize) / 2
                 */
                float_X xPos = pos.shrink<2>(1).x() + (cellWidth  * size_.x() - cellSize[simDim - 2] * simSize_.x()) / float_X(2.);
                float_X yPos = pos.shrink<2>(1).y() + (cellHeight * size_.y() - cellSize[simDim - 1] * simSize_.y()) / float_X(2.);
                targetIdx.x() = float2int_rd(xPos / cellWidth);
                targetIdx.y() = float2int_rd(yPos / cellHeight);
                /* Check bounds */
                return targetIdx.x() >= 0 && targetIdx.x() < size_.x() &&
                       targetIdx.y() >= 0 && targetIdx.y() < size_.y();
            }

        private:
            PMACC_ALIGN(xPosition_, const float_64);
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
                float_64 dt;
                /* Get index on detector, if none found -> go out */
                if(!getTargetCellIdx_(particle, globalCellIdx, targetIdx, dt))
                    return;
                accumPolicy_(detector(targetIdx), particle, timeStep_ * DELTA_T + dt);
            }
        };

    }  // namespace detail

    /**
     * Detector for photons that will accumulate incoming photons with the given policy
     * \tparam T_Config:
     *      policy IncomingParticleHandler_ The policy must define an inner type "Type"
     *          that is used for each "cell" of the detector and be a functor with
     *          signature void(Type& value, Particle, float_X timeAtDetector) that returns
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
        static constexpr float_64 distance = float_64(Config::distance / UNIT_LENGTH);

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

            // Report only for rank 0
            const bool doReport = Environment::get().GridController().getGlobalRank() == 0;

            const float_64 wavelength = particles::functors::GetWavelength<Species>()() * UNIT_LENGTH;
            // Angle for first maxima is given by: wavelength = sin(theta) * structSize ~= theta * structSize (theta << 1)
            // --> theta = wavelength / structSize
            const float_64 phiMin = Config::minNumMaxima * wavelength / Config::maxStructureSize;
            const float_64 phiMax = Config::minNumMaxima * wavelength / Config::minStructureSize;
            if(phiMin < 0 || phiMin > PI/2 || phiMax < 0 || phiMax > PI/2)
                throw std::runtime_error("Invalid range for phi. Probably your wavelength or detector constraints config is wrong.");
            // Calculate required ranges from middle of detector to the top
            // Note: We stay in SI units (m) here for simplicity
            const float_64 simSizeWidth = simSize.x() * CELL_WIDTH * UNIT_LENGTH;
            const float_64 distFromSimStart = Config::distance + simSizeWidth;
            const float_64 detRangeMin = tan(phiMin) * Config::distance;
            const float_64 detRangeMax = tan(phiMax) * distFromSimStart;
            assert(detRangeMin <= detRangeMax);

            const float_64 minSize = detRangeMax * 2;
            const float_64 maxCellSize = detRangeMin / (Config::minNumMaxima * Config::resolutionFactor);
            const float_64 maxSize = maxCellSize * std::max(size.x(), size.y());
            const float_64 minCellSize = minSize / std::min(size.x(), size.y());
            const float_64 maxDistance = std::min(size.x() * Config::cellWidth, size.y() * Config::cellHeight) / 2 / tan(phiMax) - simSizeWidth;
            const float_64 minDistance = PMaccMath::max(Config::cellWidth, Config::cellHeight) * (Config::minNumMaxima * Config::resolutionFactor) / tan(phiMin);

            if(doReport)
            {
                PMacc::log< XRTLogLvl::DOMAINS >("[INFO] Constraints for the detector (Resolution: %1%x%2% px) and light of wavelength %3%nm:\n"
                    "Size: %4%m - %5%m (%6% - %7% px)\n"
                    "CellSize: %8%µm - %9%µm\n"
                    "Distance: %10%m - %11%m")
                            % size.x() % size.y() % (wavelength * 1e9)
                            % minSize % maxSize % PMaccMath::ceil(minSize/PMaccMath::max(Config::cellWidth, Config::cellHeight)) % PMaccMath::ceil(maxSize/PMaccMath::min(Config::cellWidth, Config::cellHeight))
                            % (minCellSize * 1e6) % (maxCellSize * 1e6)
                            % minDistance % maxDistance;
            }
            std::string strError;
            if(minSize > size.x() * Config::cellWidth || minSize > size.y() * Config::cellHeight)
            {
                if(doReport)
                {
                    PMacc::log< XRTLogLvl::DOMAINS >("[WARNING] Detector is probably to small or to far away.\n"
                            "Required size: %1%m\n"
                            "Current size: (%2%m x %3%m)\n"
                            "Or maximum distance: %4%m. Current: %5%m")
                        % minSize % (size.x() * Config::cellWidth) % (size.y() * Config::cellHeight)
                        % maxDistance % Config::distance;
                }
                strError += " To small or to far away.";
            }
            // Resolution in pixels
            // PixelSize <= DetectorSize / (ResolutionFactor * minNumMaxima)
            if(PMaccMath::max(Config::cellWidth, Config::cellHeight) > maxCellSize)
            {
                if(doReport)
                {
                    PMacc::log< XRTLogLvl::DOMAINS >("[WARNING] Detector resolution might be to low or it is to close.\n"
                            "Maximum cell size: %1%µm\n"
                            "Current cell size: %2%µm\n"
                            "Or minimum distance: %3%m. Current: %4%m")
                        % (maxCellSize * 1e6)
                        % (PMaccMath::max(Config::cellWidth, Config::cellHeight) * 1e6)
                        % minDistance % Config::distance;
                }
                strError += " To low resolution or to close.";
            }
            if(!strError.empty() && Config::abortOnConstraintError)
                throw std::runtime_error(std::string("Detector constraints violated:") + strError);
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
