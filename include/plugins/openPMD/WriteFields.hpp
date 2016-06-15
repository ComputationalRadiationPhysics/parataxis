#include "xrtTypes.hpp"
#include "plugins/openPMD/WriteField.hpp"

namespace xrt {
namespace plugins {
namespace openPMD {

/**
 * Write field to HDF5 file.
 *
 * @tparam T field class
 */
template<typename T_Field>
class WriteFields
{
private:
    typedef typename T_Field::Type ValueType;

public:

    template<class T_SplashWriter>
    void operator()(T_SplashWriter& writer)
    {
        auto& dc = Environment::get().DataConnector();

        T_Field& field = dc.getData<T_Field>(T_Field::getName());

        /** \todo check if always correct at this point, depends on solver
         *        implementation */
        const float_X timeOffset = 0.0;

        WriteField<T_SplashWriter> writeField(writer);
        writeField(
                  traits::OpenPMDName<T_Field>::get(),
                  field.getGridBuffer().getGridLayout(),
                  T_Field::getUnit(),
                  T_Field::getUnitDimension(),
                  timeOffset,
                  field.getHostDataBox());

        dc.releaseData(T_Field::getName());
    }

};
}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
