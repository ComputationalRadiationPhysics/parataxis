#pragma once

#include "xrtTypes.hpp"
#include "fields/IFieldManipulator.hpp"

namespace xrt {
namespace fields {

/** Manipulator that does nothing, which results in a static field */
template<class T_Field>
class StaticManipulator: public IFieldManipulator
{
public:

    StaticManipulator(MappingDesc cellDescription){}


    void update(uint32_t currentStep) override
    {}
};

}  // namespace fields
}  // namespace xrt
