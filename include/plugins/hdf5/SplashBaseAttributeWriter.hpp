#pragma once

#include "xrtTypes.hpp"
#include "traits/PICToSplash.hpp"
#include <splash/splash.h>
#include <array>
#include <type_traits>

namespace xrt {
namespace plugins {
namespace hdf5 {
namespace detail {

    /** CRTP based base functor for writing attributes.
     *  Provides functor-functionality, but delegates actual writing to T_Writer::writeImpl
     */
    template<class T_Writer>
    struct SplashBaseAttributeWriter
    {
        /// Write an arithmetic attribute
        template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
        void operator()(const std::string& attrName, const T value);
        /// Write a string
        void operator()(const std::string& attrName, const std::string& value);
        /// Write a 1D array of strings of same size separated by #0 chars
        void operator()(const std::string& attrName, const char* value, size_t numValues);
        /// Write a 1D array
        template<typename T, size_t T_size>
        void operator()(const std::string& attrName, const std::array<T, T_size>& value);
    };

    template<class T_Writer>
    template<typename T, typename>
    void SplashBaseAttributeWriter<T_Writer>::operator()(const std::string& attrName, const T value)
    {
        typename traits::PICToSplash<T>::type splashType;
        static_cast<T_Writer&>(*this).writeImpl(splashType, attrName, &value);
    }

    template<class T_Writer>
    void SplashBaseAttributeWriter<T_Writer>::operator()(const std::string& name, const std::string& value)
    {
        splash::ColTypeString colType(value.length());
        static_cast<T_Writer&>(*this).writeImpl(colType, name, value.c_str());
    }

    template<class T_Writer>
    void SplashBaseAttributeWriter<T_Writer>::operator()(const std::string& attrName, const char* value, size_t numValues)
    {
        splash::ColTypeString colType(std::strlen(value));
        static_cast<T_Writer&>(*this).writeImpl(colType, attrName, 1u, splash::Dimensions(numValues,0,0), value);
    }

    template<class T_Writer>
    template<typename T, size_t T_size>
    void SplashBaseAttributeWriter<T_Writer>::operator()(const std::string& attrName, const std::array<T, T_size>& value)
    {
        typename traits::PICToSplash<T>::type splashType;
        static_cast<T_Writer&>(*this).writeImpl(splashType, attrName, 1u, splash::Dimensions(T_size,0,0), &value);
    }

}  // namespace detail
}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
