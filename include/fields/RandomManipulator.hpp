/**
 * Copyright 2015-2016 Alexander Grund
 *
 * This file is part of ParaTAXIS.
 *
 * ParaTAXIS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ParaTAXIS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ParaTAXIS.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#pragma once

#include "parataxisTypes.hpp"
#include "fields/IFieldManipulator.hpp"
#include "plugins/ISimulationPlugin.hpp"

namespace parataxis {
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
    void notify(uint32_t currentStep) override {}
    void checkpoint(uint32_t currentStep, const std::string checkpointDirectory) override {}
    void restart(uint32_t restartStep, const std::string restartDirectory) override {}
    void pluginRegisterHelp(boost::program_options::options_description& desc) override {}
    std::string pluginGetName() const override {return "";}
};

}  // namespace fields
}  // namespace parataxis
