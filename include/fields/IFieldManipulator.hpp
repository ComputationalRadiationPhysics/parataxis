#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace fields {

/** Update a field in each timestep before photons are moved */
class IFieldManipulator
{
public:
    virtual ~IFieldManipulator(){}
    virtual void update(uint32_t currentStep) = 0;
};

}  // namespace fields
}  // namespace xrt
