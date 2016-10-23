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
#include "plugins/hdf5/SplashBaseAttributeReader.hpp"
#include <splash/splash.h>

namespace parataxis {
namespace plugins {
namespace hdf5 {

/** Functor for reading an attribute for a given dataset or (when dataSetName is empty) the current iteration */
class SplashAttributeReader: public detail::SplashBaseAttributeReader<SplashAttributeReader>
{
public:
    SplashAttributeReader(splash::DataCollector& hdfFile, int32_t id, const std::string& dataSetName):
        hdfFile_(&hdfFile), id_(id), dataSetName_(dataSetName){}

private:
    friend struct detail::SplashBaseAttributeReader<SplashAttributeReader>;
    splash::DCAttributeInfo* readImpl(const std::string& name);

    splash::DataCollector* hdfFile_;
    int32_t id_;
    std::string dataSetName_;
};

class SplashGlobalAttributeReader: public detail::SplashBaseAttributeReader<SplashGlobalAttributeReader>
{
public:
    SplashGlobalAttributeReader(splash::DataCollector& hdfFile, int32_t id, const std::string& dataSetName):
        hdfFile_(&hdfFile), id_(id), dataSetName_(dataSetName){}

private:
    friend struct detail::SplashBaseAttributeReader<SplashGlobalAttributeReader>;
    splash::DCAttributeInfo* readImpl(const std::string& name);

    splash::DataCollector* hdfFile_;
    int32_t id_;
    std::string dataSetName_;
};

splash::DCAttributeInfo* SplashAttributeReader::readImpl(const std::string& name)
{
    return hdfFile_->readAttributeMeta(id_, dataSetName_.empty() ? nullptr : dataSetName_.c_str(), name.c_str());
}

splash::DCAttributeInfo* SplashGlobalAttributeReader::readImpl(const std::string& name)
{
    return hdfFile_->readGlobalAttributeMeta(id_, name.c_str());
}

}  // namespace openPMD
}  // namespace plugins
}  // namespace parataxis
