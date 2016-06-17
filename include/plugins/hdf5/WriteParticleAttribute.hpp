#pragma once

#include "xrtTypes.hpp"
#include "traits/PICToOpenPMD.hpp"

namespace xrt {
namespace plugins {
namespace hdf5 {

/** write attribute of a particle to hdf5 file
 *
 * @tparam T_Identifier identifier of a particle record
 */
template<typename T_Identifier>
struct WriteParticleAttribute
{
    /** write attribute to hdf5 file
     *
     * @param frame frame with all particles
     * @param elements number of particles in this patch
     * @param elementsOffset number of particles in this patch
     * @param numParticlesGlobal number of particles globally
     */
    template<class T_SplashWriter, typename T_Frame>
    HINLINE void operator()(T_SplashWriter& inWriter,
                            T_Frame& frame,
                            const uint64_t numParticles,
                            const uint64_t localParticlesOffset,
                            const uint64_t numParticlesGlobal
    )
    {
        typedef typename Resolve_t<T_Identifier>::type ValueType;
        constexpr uint32_t numComponents = PMacc::traits::GetNComponents<ValueType>::value;

        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  (begin) write species attribute: %1%") % T_Identifier::getName();

        T_SplashWriter writer = inWriter[traits::OpenPMDName<T_Identifier>::get()];

        const std::string name_lookup[] = {"x", "y", "z"};

        // get the SI scaling, dimensionality of the attribute
        std::vector<float_64> unit = traits::OpenPMDUnit<T_Identifier, T_Frame>::get();
        std::vector<float_64> unitDimension = traits::OpenPMDUnit<T_Identifier, T_Frame>::getDimension();

        assert(unit.size() == numComponents); // unitSI for each component
        assert(unitDimension.size() == traits::NUnitDimension); // seven openPMD base units

        const PMacc::Selection<simDim>& globalDomain = Environment::get().SubGrid().getGlobalDomain();

        typedef typename PMacc::traits::GetComponentsType<ValueType>::type ComponentValueType;
        ComponentValueType* tmpArray = new ComponentValueType[numParticles];

        for (uint32_t d = 0; d < numComponents; d++)
        {
            ValueType* dataPtr = frame.getIdentifier(T_Identifier()).getPointer();
            #pragma omp parallel for
            for(uint64_t i = 0; i < numParticles; ++i)
                tmpArray[i] = ((ComponentValueType*)dataPtr)[i * numComponents + d];

            // Write components only for >1D records
            auto tmpWriter = (numComponents > 1) ? writer[name_lookup[d]] : writer;
            tmpWriter.GetPolyDataWriter()(
                tmpArray,
                1,
                makeSplashDomain(globalDomain),
                splash::Dimensions(numParticlesGlobal, 1, 1),
                splash::Domain(
                    splash::Dimensions(numParticles, 1, 1),
                    splash::Dimensions(localParticlesOffset, 1, 1)
                )
            );
            tmpWriter.GetAttributeWriter()("unitSI", unit.at(d));
        }
        __deleteArray(tmpArray);

        auto writeAttribute = writer.GetAttributeWriter();

        writeAttribute("unitDimension", unitDimension);
        writeAttribute("timeOffset", float_X(0));
        // Single particles
        writeAttribute("macroWeighted", uint32_t(0));
        writeAttribute("weightingPower", float_64(1));

        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  ( end ) write species attribute: %1%") % T_Identifier::getName();
    }

};

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt