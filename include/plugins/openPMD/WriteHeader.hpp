#pragma once

#include "xrtTypes.hpp"
#include "version.hpp"
#include "plugins/common/stringHelpers.hpp"
#include "traits/PICToSplash.hpp"
#include <splash/splash.h>

namespace xrt {
namespace plugins {
namespace openPMD {

namespace detail {

    template<bool T_isParallel>
    struct SplashProxy
    {
        using Collector = splash::IParallelDataCollector;
        SplashProxy(Collector& hdfFile): hdfFile_(hdfFile){}

        void writeGlobalAttribute(int32_t id,
                        const splash::CollectionType& type,
                        const char *name,
                        const void* buf)
        {
            hdfFile_.writeGlobalAttribute(id, type, name, buf);
        }

        void writeAttribute(int32_t id,
                            const splash::CollectionType& type,
                            const char *dataName,
                            const char *attrName,
                            const void *buf)
        {
            hdfFile_.writeAttribute(id, type, dataName, attrName, buf);
        }

        Collector& hdfFile_;
    };

    template<>
    struct SplashProxy<false>
    {
        using Collector = splash::SerialDataCollector;
        SplashProxy(Collector& hdfFile): hdfFile_(hdfFile){}

        void writeGlobalAttribute(int32_t id,
                        const splash::CollectionType& type,
                        const char *name,
                        const void* buf)
        {
            hdfFile_.writeGlobalAttribute(type, name, buf);
        }

        void writeAttribute(int32_t id,
                            const splash::CollectionType& type,
                            const char *dataName,
                            const char *attrName,
                            const void *buf)
        {
            hdfFile_.writeAttribute(id, type, dataName, attrName, buf);
        }

        Collector& hdfFile_;
    };

}  // namespace detail

template<bool T_isParallel>
struct WriteHeader
{
    using SplashProxy = detail::SplashProxy<T_isParallel>;

    WriteHeader(typename SplashProxy::Collector& hdfFile, int32_t id): hdfFile_(hdfFile), id_(id){}

    void operator()(const std::string& fileNameBase, bool usePIC_ED_Ext = false)
    {
        /* openPMD attributes */
        /*   required */
    	writeGlobalAttribute("openPMD", "1.0.0");
        writeGlobalAttribute("openPMDextension", uint32_t(usePIC_ED_Ext ? 1 : 0)); // ED-PIC ID
        writeGlobalAttribute("basePath", "/data/%T/");
        writeGlobalAttribute("meshesPath", "meshes/");
        writeGlobalAttribute("particlesPath", "particles/");
        writeGlobalAttribute("iterationEncoding", "fileBased");
        writeGlobalAttribute("iterationFormat", fileNameBase + std::string("_%T.h5"));

        /*   recommended */
        std::string author = Environment::get().SimulationDescription().getAuthor();
        if(!author.empty())
        	writeGlobalAttribute("author", author);
        writeGlobalAttribute("software", "XRT");
        std::stringstream softwareVersion;
        softwareVersion << XRT_VERSION_MAJOR << "."
                        << XRT_VERSION_MINOR << "."
                        << XRT_VERSION_PATCH;
        writeGlobalAttribute("softwareVersion", softwareVersion.str());
        writeGlobalAttribute("date", common::getDateString("%F %T %z"));

        /* openPMD: required time attributes */
        writeAttribute(nullptr, "dt", DELTA_T);
        writeAttribute(nullptr, "time", float_X(Environment::get().SimulationDescription().getCurrentStep()) * DELTA_T);
        writeAttribute(nullptr, "timeUnitSI", UNIT_TIME);
    }
private:
    void writeGlobalAttribute(const std::string& name, const std::string& value)
    {
        splash::ColTypeString colType(value.length());
        hdfFile_.writeGlobalAttribute(id_, colType, name.c_str(), value.c_str());
    }

    void writeGlobalAttribute(const std::string& name, const char* value)
    {
        writeGlobalAttribute(name, std::string(value));
    }

    template<typename T>
    void writeGlobalAttribute(const std::string& attrName, const T value)
    {
        typename traits::PICToSplash<T>::type splashType;
        hdfFile_.writeGlobalAttribute(id_, splashType, attrName.c_str(), &value);
    }

    template<typename T>
    void writeAttribute(const char* dataName, const char* attrName, const T value)
    {
        typename traits::PICToSplash<T>::type splashType;
        hdfFile_.writeAttribute(id_, splashType, dataName, attrName, &value);
    }

    const int32_t id_;
    SplashProxy hdfFile_;
};

}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
