#pragma once

#include "xrtTypes.hpp"
#include "fields/IFieldManipulator.hpp"
#include "plugins/ISimulationPlugin.hpp"

namespace xrt {
namespace fields {

    namespace kernel
    {
        template<class T_BoxWriteOnly, class T_Random, class T_Mapping>
        __global__ void updateRandomly(T_BoxWriteOnly buffWrite, T_Random rand, T_Mapping mapper)
        {
            const Space blockCellIdx = mapper.getSuperCellIndex(Space(blockIdx)) * SuperCellSize::toRT();
            const Space cellOffset(threadIdx);
            const Space localCellIdx = blockCellIdx - mapper.getGuardingSuperCells() * SuperCellSize::toRT() + cellOffset;
            rand.init(localCellIdx);
            buffWrite(blockCellIdx + cellOffset) = rand();
        }
    }

/** Manipulator that produces a random field in each timestep */
template<class T_Field>
class RandomManipulator: public IFieldManipulator, public ISimulationPlugin
{
public:

    RandomManipulator()
    {
        Environment::get().PluginConnector().registerPlugin(this);
    }

    void update(uint32_t currentStep) override
    {
        auto& dc = Environment::get().DataConnector();
        T_Field& field = dc.getData<T_Field>(T_Field::getName(), true);
        __cudaKernelArea(kernel::updateRandomly, *cellDescription_, PMacc::CORE + PMacc::BORDER)
                (MappingDesc::SuperCellSize::toRT().toDim3())
                (field.getDeviceDataBox(),
                Random<>());
    }

    template<class T_MaxVal = std::integral_constant<int, 10>>
    struct Random
    {
        using Distribution = PMacc::random::distributions::Uniform<float>;
        using RandomType = typename RNGProvider::GetRandomType<Distribution>::type;

        HINLINE
        Random(): rand(RNGProvider::createRandom<Distribution>())
        {}

        DINLINE void
        init(const Space& localCellIdx)
        {
            rand.init(localCellIdx);
        }

        DINLINE float
        operator()()
        {
            return rand() * T_MaxVal::value;
        }

    private:
        PMACC_ALIGN8(rand, RandomType);
    };
private:
    // Unused methods from IPlugin
    void checkpoint(uint32_t currentStep, const std::string checkpointDirectory) override {}
    void restart(uint32_t restartStep, const std::string restartDirectory) override {}
    void pluginRegisterHelp(boost::program_options::options_description& desc) override {}
    std::string pluginGetName() const override {return "";}
};

}  // namespace fields
}  // namespace xrt
