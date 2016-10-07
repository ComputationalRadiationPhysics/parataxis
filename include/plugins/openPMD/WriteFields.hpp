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
