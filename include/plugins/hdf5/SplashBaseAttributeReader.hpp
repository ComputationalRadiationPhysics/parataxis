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
#include "traits/PICToSplash.hpp"
#include <splash/splash.h>
#include <array>
#include <vector>
#include <type_traits>

namespace parataxis {
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
        void operator()(const std::string& name, size_t numValues, char* value, size_t bufSize);
        /// Read a 1D array
        template<typename T, size_t T_size>
        void operator()(const std::string& name, std::array<T, T_size>& value);
        /// Read a 1D array
        template<typename T, int T_size>
        void operator()(const std::string& name, PMacc::math::Vector<T, T_size>& value);
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
            // Note: The string may or may not end in a NULL terminator.
            // (Writing with splash: yes (NULL_TERMINATED), h5py: No (NULL_PADDED))
            // Currently there is no way knowing if there should be one.
            attr->read(attr->getType(), &readVal[0]);
            value.assign(readVal.begin(), readVal.end());
        }else
            value = "";
    }

    template<class T_Reader>
    void SplashBaseAttributeReader<T_Reader>::operator()(const std::string& name, size_t numValues, char* value, size_t bufSize)
    {
        std::unique_ptr<splash::DCAttributeInfo> attr = getAttribute(name);
        if(typeid(attr->getType()) != typeid(splash::ColTypeString))
            throw std::runtime_error("Attribute is not a string");
        if(attr->getNDims() != 1 || attr->getDims()[0] != numValues)
            throw std::runtime_error("Wrong string array size");
        // We need space for at least 1 char and 1 NULL terminator per value
        if(bufSize < numValues * 2)
            throw std::runtime_error("Buffer is to small for string array");
        // remove NULL terminators and divide by number of values
        size_t strLen = (bufSize - numValues) / numValues;
        attr->read(splash::ColTypeString(strLen), value);
    }

    template<class T_Reader>
    template<typename T, size_t T_size>
    void SplashBaseAttributeReader<T_Reader>::operator()(const std::string& name, std::array<T, T_size>& value)
    {
        std::unique_ptr<splash::DCAttributeInfo> attr = getAttribute(name);
        if(attr->getNDims() != 1 || attr->getDims()[0] != T_size)
            throw std::runtime_error("Wrong attribute array size");
        typename traits::PICToSplash<T>::type splashType;
        attr->read(splashType, &value.front());
    }

    template<class T_Reader>
    template<typename T, int T_size>
    void SplashBaseAttributeReader<T_Reader>::operator()(const std::string& name, PMacc::math::Vector<T, T_size>& value)
    {
        static_assert(T_size > 0, "Invalid size");
        std::array<T, T_size> tmp;
        (*this)(name, tmp);
        for(int i=0; i<T_size; ++i)
            value[i] = tmp[i];
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
}  // namespace parataxis
