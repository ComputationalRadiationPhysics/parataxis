#pragma once

#include "xrtTypes.hpp"
#include "traits/PICToSplash.hpp"
#include <splash/splash.h>
#include <array>
#include <vector>
#include <type_traits>

namespace xrt {
namespace plugins {
namespace hdf5 {
namespace detail {

    /** CRTP based base functor for reading attributes.
     *  Provides functor-functionality, but delegates actual reading to T_Reader::readImpl
     */
    template<class T_Reader>
    struct SplashBaseAttributeReader
    {
        /// Read a arithmetic attribute
        template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
        void operator()(const std::string& attrName, T& value);
        /// Read a string
        void operator()(const std::string& attrName, std::string& value);
        /// Read a 1D array of strings of same size separated by #0 chars
        void operator()(const std::string& attrName, char* value, size_t numValues);
        /// Read a 1D array
        template<typename T, size_t T_size>
        void operator()(const std::string& attrName, std::array<T, T_size>& value);
        /// Read a 1D array
        template<typename T>
        void operator()(const std::string& attrName, std::vector<T>& value);
    };

    template<class T_Reader>
    template<typename T, typename>
    void SplashBaseAttributeReader<T_Reader>::operator()(const std::string& attrName, T& value)
    {
        typename traits::PICToSplash<T>::type splashType;
        static_cast<T_Reader&>(*this).readImpl(splashType, attrName, &value);
    }

    template<class T_Reader>
    void SplashBaseAttributeReader<T_Reader>::operator()(const std::string& name, std::string& value)
    {
        char tmp[1024];
        splash::ColTypeString colType(1337); // TODO
        static_cast<T_Reader&>(*this).readImpl(colType, name, tmp);
        value = tmp;
    }

    template<class T_Reader>
    void SplashBaseAttributeReader<T_Reader>::operator()(const std::string& attrName, char* value, size_t numValues)
    {
        splash::ColTypeString colType(1337); // TODO
        static_cast<T_Reader&>(*this).readImpl(colType, attrName, 1u, splash::Dimensions(numValues,0,0), value);
    }

    template<class T_Reader>
    template<typename T, size_t T_size>
    void SplashBaseAttributeReader<T_Reader>::operator()(const std::string& attrName, std::array<T, T_size>& value)
    {
        typename traits::PICToSplash<T>::type splashType;
        static_cast<T_Reader&>(*this).readImpl(splashType, attrName, 1u, splash::Dimensions(T_size,0,0), &value);
    }

    template<class T_Reader>
    template<typename T>
    void SplashBaseAttributeReader<T_Reader>::operator()(const std::string& attrName, std::vector<T>& value)
    {
        assert(!value.empty()); // TODO: size must be known in current implementation
        typename traits::PICToSplash<T>::type splashType;
        static_cast<T_Reader&>(*this).readImpl(splashType, attrName, 1u, splash::Dimensions(value.size(),0,0), &value.front());
    }

}  // namespace detail
}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
