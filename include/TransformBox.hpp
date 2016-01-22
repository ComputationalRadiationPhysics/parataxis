#pragma once

#include <xrtTypes.hpp>

namespace xrt {

    /**
     * A simple wrapper around a DataBox that transforms elements on access
     */
    template<class T_BaseBox, class T_Transform>
    class TransformBox: private T_BaseBox
    {
        using BaseValueType = typename T_BaseBox::ValueType;
        T_Transform transformation;
    public:
        using ValueType = std::result_of_t<T_Transform(BaseValueType)>;

        HDINLINE TransformBox(const T_BaseBox& base, const T_Transform& transformation): T_BaseBox(base), transformation(transformation)
        {}

        template<unsigned T_Dim>
        HDINLINE ValueType operator()(const PMacc::DataSpace<T_Dim>& idx) const
        {
            return transformation(T_BaseBox::operator()(idx));
        }
    };

    template<class T_BaseBox, class T_Transform>
    HDINLINE TransformBox<T_BaseBox, T_Transform>
    makeTransformBox(T_BaseBox&& box, T_Transform&& transformation = T_Transform())
    {
        return TransformBox<T_BaseBox, T_Transform>(box, transformation);
    }
}  // namespace xrt
