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
#include "traits/stdRenamings.hpp"

namespace parataxis {

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
        using RefValueType = traits::result_of_t<T_Transform(BaseValueType&)>;
        using ValueType = traits::remove_reference_t<RefValueType>;

        HDINLINE TransformBox(const T_BaseBox& base, const T_Transform& transformation): T_BaseBox(base), transformation(transformation)
        {}

        template<unsigned T_Dim>
        HDINLINE auto operator()(const PMacc::DataSpace<T_Dim>& idx) const
        -> traits::result_of_t<T_Transform(traits::result_of_t<T_BaseBox(decltype(idx))>)>
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
            HDINLINE traits::result_of_t<T_Transform(T)> operator()(T&& input) const
            {
                return transformation(input);
            }

        };

    }  // namespace detail

    template<class T_BaseBox, class T_Transform>
    HDINLINE TransformBox<traits::remove_cv_t<traits::remove_reference_t<T_BaseBox>>, T_Transform>
    makeTransformBox(T_BaseBox&& box, T_Transform&& transformation = T_Transform())
    {
        using BaseBox = traits::remove_cv_t<traits::remove_reference_t<T_BaseBox>>;
        return TransformBox<BaseBox, T_Transform>(box, transformation);
    }

    template<class T_BaseBox, class T_Transform>
    TransformBox<traits::remove_cv_t<traits::remove_reference_t<T_BaseBox>>, detail::HostTransformWrapper<T_Transform>>
    makeHostTransformBox(T_BaseBox&& box, T_Transform&& transformation = T_Transform())
    {
        using BaseBox = traits::remove_cv_t<traits::remove_reference_t<T_BaseBox>>;
        return TransformBox<BaseBox, detail::HostTransformWrapper<T_Transform>>(box, transformation);
    }
}  // namespace parataxis
