#pragma once

#include <xrtTypes.hpp>
#include <type_traits>

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
        static constexpr unsigned Dim = T_BaseBox::Dim;
        using RefValueType = std::result_of_t<T_Transform(BaseValueType&)>;
        using ValueType = std::remove_reference_t<RefValueType>;

        HDINLINE TransformBox(const T_BaseBox& base, const T_Transform& transformation): T_BaseBox(base), transformation(transformation)
        {}

        template<unsigned T_Dim>
        HDINLINE auto operator()(const PMacc::DataSpace<T_Dim>& idx) const
        -> std::result_of_t<T_Transform(std::result_of_t<T_BaseBox(decltype(idx))>)>
        {
            return transformation(T_BaseBox::operator()(idx));
        }
    };

    namespace detail {

        /** Wrapper to Host-Only transformation such as lambda to avoid the warning */
        template<typename T_Transform>
        struct HostTransformWrapper
        {
            T_Transform transformation;

            HDINLINE HostTransformWrapper(T_Transform& trans): transformation(trans)
            {}

            PMACC_NO_NVCC_HDWARNING
            template<typename T>
            HDINLINE std::result_of_t<T_Transform(T)> operator()(T&& input) const
            {
                return transformation(input);
            }

        };

    }  // namespace detail

    template<class T_BaseBox, class T_Transform>
    HDINLINE TransformBox<std::remove_cv_t<std::remove_reference_t<T_BaseBox>>, T_Transform>
    makeTransformBox(T_BaseBox&& box, T_Transform&& transformation = T_Transform())
    {
        using BaseBox = std::remove_cv_t<std::remove_reference_t<T_BaseBox>>;
        return TransformBox<BaseBox, T_Transform>(box, transformation);
    }

    template<class T_BaseBox, class T_Transform>
    TransformBox<std::remove_cv_t<std::remove_reference_t<T_BaseBox>>, detail::HostTransformWrapper<T_Transform>>
    makeHostTransformBox(T_BaseBox&& box, T_Transform&& transformation = T_Transform())
    {
        using BaseBox = std::remove_cv_t<std::remove_reference_t<T_BaseBox>>;
        return TransformBox<BaseBox, detail::HostTransformWrapper<T_Transform>>(box, transformation);
    }
}  // namespace xrt
