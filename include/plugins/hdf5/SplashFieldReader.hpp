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

#include "xrtTypes.hpp"
#include "traits/PICToSplash.hpp"
#include "plugins/hdf5/splashUtils.hpp"
#include <splash/splash.h>

namespace xrt {
namespace plugins {
namespace hdf5 {

    /** Functor for writing a domain to hdf5 */
    class SplashFieldReader
    {
    public:
        SplashFieldReader(splash::ParallelDataCollector& hdfFile, int32_t id, const std::string& datasetName):
            hdfFile_(hdfFile), id_(id), datasetName_(datasetName){}
        /// Read the field
        /// @param data         Pointer to contiguous data
        /// @param globalDomain Offset and Size of the field over all processes
        /// @param localDomain  Offset and Size of the field on the current process (Size must match extents of data)
        template<typename T>
        void operator()(T* data, unsigned numDims, const splash::Dimensions& globalSize, const splash::Domain& localDomain);
        /** Return the global size of the dataset */
        splash::Dimensions getGlobalSize();
    private:
        template<typename T_In, typename T_Out>
        bool tryConvertedRead(T_Out* data, const splash::Domain& domain, const splash::CollectionType& colTypeData);
        splash::ParallelDataCollector& hdfFile_;
        const int32_t id_;
        std::string datasetName_;
    };

    splash::Dimensions SplashFieldReader::getGlobalSize()
    {
        // sizeRead will be set
        splash::Dimensions globalSizeOut;
        std::unique_ptr<splash::CollectionType> colType(hdfFile_.readMeta(
            id_,
            datasetName_.c_str(),
            splash::Dimensions(0, 0, 0),
            splash::Dimensions(0, 0, 0),
            globalSizeOut));
        return globalSizeOut;
    }

    template<typename T>
    void SplashFieldReader::operator()(T* data, unsigned numDims, const splash::Dimensions& globalSize, const splash::Domain& localDomain)
    {
        PMacc::log<XRTLogLvl::DEBUG>("HDF5: reading %4%D record %1% (globalSize: %2%, localDomain: %3%")
                % datasetName_ % globalSize.toString() % localDomain.toString() % numDims;

        typename traits::PICToSplash<T>::type splashType;
        assert(isSizeValid(globalSize, numDims));
        assert(isDomainValid(localDomain, numDims));

        // Validate dataset
        // sizeRead will be set
        splash::Dimensions sizeRead;
        std::unique_ptr<splash::CollectionType> colType(hdfFile_.readMeta(
            id_,
            datasetName_.c_str(),
            globalSize,
            splash::Dimensions(0, 0, 0),
            sizeRead));

        if(sizeRead != globalSize)
            throw std::runtime_error(std::string("Invalid global size: ") + sizeRead.toString() + "!=" + globalSize.toString());

        // Simple case: Same types
        if(tryConvertedRead<T>(data, localDomain, *colType))
            return;
        // float64->float32
        if(std::is_same<T, float_32>::value && tryConvertedRead<float_64>(data, localDomain, *colType))
            return;
        // float32->float64
        if(std::is_same<T, float_64>::value && tryConvertedRead<float_32>(data, localDomain, *colType))
            return;

        // No match -> Error
        throw std::runtime_error(std::string("Invalid data type: ") + colType->toString() + "!=" + splashType.toString());
    }

    template<typename T_In, typename T_Out>
    bool SplashFieldReader::tryConvertedRead(T_Out* data, const splash::Domain& domain, const splash::CollectionType& colTypeData)
    {
        using ColTypeIn = typename traits::PICToSplash<T_In>::type;

        if(typeid(ColTypeIn) != typeid(colTypeData))
            return false;
        std::unique_ptr<T_In[]> tmpData;
        if(!std::is_same<T_In, T_Out>::value)
            tmpData.reset(new T_In[domain.getSize().getScalarSize()]);

        splash::Dimensions sizeRead;
        // Read dataset
        hdfFile_.read(
            id_,
            domain.getSize(),
            domain.getOffset(),
            datasetName_.c_str(),
            sizeRead,
            tmpData ? (void*)tmpData.get() : (void*)data);

        if(sizeRead != domain.getSize())
             throw std::runtime_error(std::string("Invalid local size: ") + sizeRead.toString() + "!=" + domain.getSize().toString());

        if(!std::is_same<T_In, T_Out>::value)
        {
            // Copy-Convert data
            std::copy(tmpData.get(), tmpData.get() + domain.getSize().getScalarSize(), data);
        }

        return true;
   }

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
