#pragma once

#include <type_traits>

namespace xrt {
namespace traits {

    //Shortcuts from C++14 standard

    template< class T >
    using result_of_t = typename std::result_of<T>::type;

    template< class T >
    using remove_reference_t = typename std::remove_reference<T>::type;

    template< class T >
    using remove_const_t = typename std::remove_const<T>::type;

    template< class T >
    using remove_cv_t = typename std::remove_cv<T>::type;

}  // namespace traits
}  // namespace xrt
