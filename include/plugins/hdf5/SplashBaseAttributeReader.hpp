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
        void operator()(const std::string& name, T& value);
        /// Read a string
        void operator()(const std::string& name, std::string& value);
        /// Read a 1D array of strings of same size separated by #0 chars
        void operator()(const std::string& name, char* value, size_t sizes, size_t numValues);
        /// Read a 1D array
        template<typename T, size_t T_size>
        void operator()(const std::string& name, std::array<T, T_size>& value);
        /// Read a 1D array
        template<typename T>
        void operator()(const std::string& name, std::vector<T>& value);

        std::string readString(const std::string& name){ std::string result; (*this)(name, result); return result; }
    private:
        std::unique_ptr<splash::DCAttributeInfo> getAttribute(const std::string& name)
        {
            return std::unique_ptr<splash::DCAttributeInfo>(static_cast<T_Reader&>(*this).readImpl(name));
        }
    };

    template<class T_Reader>
    template<typename T, typename>
    void SplashBaseAttributeReader<T_Reader>::operator()(const std::string& name, T& value)
    {
        typename traits::PICToSplash<T>::type splashType;
        std::unique_ptr<splash::DCAttributeInfo> attr = getAttribute(name);
        if(!attr->isScalar())
            throw std::runtime_error("Unexpected multi-dim attribute");
        attr->read(splashType, &value);
    }

    template<class T_Reader>
    void SplashBaseAttributeReader<T_Reader>::operator()(const std::string& name, std::string& value)
    {
        std::unique_ptr<splash::DCAttributeInfo> attr = getAttribute(name);
        if(typeid(attr->getType()) != typeid(splash::ColTypeString))
            throw std::runtime_error("Attribute is not a string");
        if(!attr->isScalar())
            throw std::runtime_error("Unexpected multi-dim attribute");
        if(attr->isVarSize())
        {
            const char* sPtr = NULL;
            attr->read(&sPtr, sizeof(sPtr));
            value = sPtr;
        }else if(attr->getMemSize() > 0)
        {
            std::vector<char> readVal(attr->getMemSize());
            attr->read(&readVal[0], readVal.size());
            // Remove NULL terminator
            readVal.resize(readVal.size() - 1);
            value.assign(readVal.begin(), readVal.end());
        }else
            value = "";
    }

    template<class T_Reader>
    void SplashBaseAttributeReader<T_Reader>::operator()(const std::string& name, char* value, size_t sizes, size_t numValues)
    {
        std::unique_ptr<splash::DCAttributeInfo> attr = getAttribute(name);
        if(typeid(attr->getType()) != typeid(splash::ColTypeString))
            throw std::runtime_error("Attribute is not a string");
        if(attr->getNDims() != 1 || attr->getDims()[0] != numValues)
            throw std::runtime_error("Wrong string array size");
        attr->read(value, (sizes + 1) * numValues);
    }

    template<class T_Reader>
    template<typename T, size_t T_size>
    void SplashBaseAttributeReader<T_Reader>::operator()(const std::string& name, std::array<T, T_size>& value)
    {
        std::unique_ptr<splash::DCAttributeInfo> attr = getAttribute(name);
        if(attr->getNDims() != 1 || attr->getDims()[0] != T_size)
            throw std::runtime_error("Wrong attribute array size");
        // TODO: Implement possible type conversion by using CollectionType
        attr->read(&value.front(), sizeof(value));
    }

    template<class T_Reader>
    template<typename T>
    void SplashBaseAttributeReader<T_Reader>::operator()(const std::string& name, std::vector<T>& value)
    {
        std::unique_ptr<splash::DCAttributeInfo> attr = getAttribute(name);
        if(attr->getNDims() != 1)
            throw std::runtime_error("Wrong attribute array size");
        value.resize(attr->getDims()[0]);
        // TODO: Implement possible type conversion by using CollectionType
        attr->read(&value.front(), sizeof(T) * value.size());
    }

}  // namespace detail
}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
