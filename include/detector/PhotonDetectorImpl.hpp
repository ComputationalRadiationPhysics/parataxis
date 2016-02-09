#pragma once

#include "xrtTypes.hpp"
#include "particles/functors/GetWavelength.hpp"
#include "debug/LogLevels.hpp"
#include "math/angleHelpers.hpp"

#include <memory/buffers/GridBuffer.hpp>
#include <dataManagement/ISimulationData.hpp>

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
            static constexpr float_X cellWidth  = float_X(Config::cellWidth  / UNIT_LENGTH);
            static constexpr float_X cellHeight = float_X(Config::cellHeight / UNIT_LENGTH);

            /**
             *
             * @param xPosition Position of the detector in x direction
             * @param size      Size of the detector
             */
            GetTargetCellIdx(Space2D size, float_X angleRangeX, float_X angleRangeY):
                size_(size), angleRangeX_(angleRangeX), angleRangeY_(angleRangeY), simSize_(Environment::get().SubGrid().getTotalDomain().size.shrink<2>(1))
            {}

            /**
             * Calculates the index on the detector and the time of impact
             * Returns true, if the detector was hit, false otherwise.
             * On false \ref targetIdx and \ref dt are undefined
             */
            template<typename T_Particle>
            HDINLINE bool
            operator()(const T_Particle& particle, Space globalIdx, Space2D& targetIdx) const
            {
                auto dir = particle[direction_];
                /* Not flying towards detector? -> exit */
                if(dir.x() <= 0)
                    return false;

                using PMacc::algorithms::math::float2int_rd;

                // Calculate angle in "back" dimension (when viewed from front) -> X-dimension of detector
                float_X angleBack = atan(dir.z() / dir.x());
                // Calculate cell index on detector by angle (histogram-like binning)
                float_X cellIdxX  = angleBack / angleRangeX_ + size_.x() / static_cast<float_X>(2);
                // Add (usually very small) offset if the volume is not much smaller than a detector cell
                cellIdxX += (globalIdx.z() - simSize_.y() / 2) * CELL_DEPTH / cellWidth;
                targetIdx.x() = float2int_rd(cellIdxX);
                // Same for "down" dimension -> Y-dimension of detector
                float_X angleDown = atan(dir.y() / dir.x());
                float_X cellIdxY  = angleDown / angleRangeY_ + size_.y() / static_cast<float_X>(2);
                cellIdxY += (globalIdx.y() - simSize_.x() / 2) * CELL_HEIGHT / cellHeight;
                targetIdx.y() = float2int_rd(cellIdxY);

                /* Check bounds */
                return targetIdx.x() >= 0 && targetIdx.x() < size_.x() &&
                       targetIdx.y() >= 0 && targetIdx.y() < size_.y();
            }

        private:
            PMACC_ALIGN(angleRangeX_, const float_X);
            PMACC_ALIGN(angleRangeY_, const float_X);
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
        public:

            DetectParticle(GetTargetCellIdx getTargetCellIdx, AccumPolicy accumPolicy):
                getTargetCellIdx_(getTargetCellIdx), accumPolicy_(accumPolicy)
            {}

            template<typename T_Particle, typename T_DetectorBox>
            HDINLINE void
            operator()(const T_Particle& particle, const Space superCellPosition, T_DetectorBox& detector) const
            {
                /*calculate global cell index*/
                const Space localCell(PMacc::DataSpaceOperations<simDim>::map<SuperCellSize>(particle[PMacc::localCellIdx_]));
                const Space globalCellIdx = superCellPosition + localCell;
                Space2D targetIdx;
                /* Get index on detector, if none found -> go out */
                if(getTargetCellIdx_(particle, globalCellIdx, targetIdx))
                    accumPolicy_(detector(targetIdx), particle, globalCellIdx);
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
    public:
        /**
         * Distance of the detector from the right side of the volume
         */
        static constexpr float_64 distance   = float_64(Config::distance   / UNIT_LENGTH);
        static constexpr float_64 cellWidth  = float_64(Config::cellWidth  / UNIT_LENGTH);
        static constexpr float_64 cellHeight = float_64(Config::cellHeight / UNIT_LENGTH);
        using Type = typename AccumPolicy::Type;
        using OutputTransformer = typename AccumPolicy::OutputTransformer;
    private:
        using Buffer = PMacc::GridBuffer< Type, 2 >;
        std::unique_ptr< Buffer > buffer;
        // which angles are covered by 1 cell
        float_X angleRangePerCellX_, angleRangePerCellY_;

    public:

        using DetectParticle = detail::DetectParticle<detail::GetTargetCellIdx<Config>, AccumPolicy>;

        PhotonDetectorImpl(const Space2D& size)
        {
            buffer.reset(new Buffer(size));
            float_64 angleRangeX = atan(size.x() * cellWidth  / distance);
            float_64 angleRangeY = atan(size.y() * cellHeight / distance);
            angleRangePerCellX_ = angleRangeX / size.x();
            angleRangePerCellY_ = angleRangeY / size.y();

            validateConstraints();
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

        void reset()
        {
            buffer->reset(false);
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
            return DetectParticle(detail::GetTargetCellIdx<Config>(getSize(), angleRangePerCellX_, angleRangePerCellY_), AccumPolicy(timeStep));
        }

    private:
        void validateConstraints()
        {
            const Space simSize = Environment::get().SubGrid().getTotalDomain().size;
            const Space2D size = getSize();
            // Report only for rank 0
            const bool doReport = Environment::get().GridController().getGlobalRank() == 0;

            if(doReport)
            {
                using math::rad2deg;
                PMacc::log< XRTLogLvl::DOMAINS >("Detector detects angles in +- %g/%g(%g°/%g°) with resolution %g/%g(%g°/%g°)")
                            % (angleRangePerCellX_ * size.x() / 2) % (angleRangePerCellY_ * size.y() / 2)
                            % rad2deg(angleRangePerCellX_ * size.x() / 2) % rad2deg(angleRangePerCellY_ * size.y() / 2)
                            % angleRangePerCellX_ % angleRangePerCellY_
                            % rad2deg(angleRangePerCellX_) % rad2deg(angleRangePerCellY_);
            }

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
                            % minSize % maxSize
                            % PMaccMath::ceil(minSize/PMaccMath::max(Config::cellWidth, Config::cellHeight))
                            % PMaccMath::ceil(maxSize/PMaccMath::min(Config::cellWidth, Config::cellHeight))
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
    };

}  // namespace detector
}  // namespace xrt
