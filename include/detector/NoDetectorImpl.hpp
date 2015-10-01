#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace detector {

    class NoDetectorImpl: PMacc::ISimulationData
    {
        using Buffer = PMacc::GridBuffer<int, simDim>;
        struct DetectParticle
        {
            template<typename T_Particle, typename T_DetectorBox>
            HDINLINE void
            operator()(const T_Particle& particle, const Space superCellPosition, T_DetectorBox& detector) const
            {}
        };
    public:

        static std::string
        getName()
        {
            return "NoDetector";
        }

        PMacc::SimulationDataId getUniqueId() override
        {
            return getName();
        }

        void synchronize() override
        {}

        void init()
        {}

        typename Buffer::DataBoxType
        getHostDataBox()
        {
            return Buffer::DataBoxType();
        }

        typename Buffer::DataBoxType
        getDeviceDataBox()
        {
            return Buffer::DataBoxType();
        }

        Space2D
        getSize() const
        {
            return Space2D();
        }

        DetectParticle
        getDetectParticle(uint32_t timeStep) const
        {
            return DetectParticle();
        }
    };

}  // namespace detector
}  // namespace xrt
