#pragma once

namespace xrt {
namespace plugins {

    struct OperatorCreateVectorBox
    {
        template<typename InType>
        struct apply
        {
            typedef
            bmpl::pair< InType, PMacc::VectorDataBox<typename Resolve_t<InType>::type> >
            type;
        };
    };

    struct NoFilter
    {
        template<class T_Space>
        HDINLINE void setSuperCellPosition(T_Space){}

        template<class T_Frame>
        HDINLINE bool operator()(T_Frame& frame, PMacc::lcellId_t id)
        {
            return true;
        }
    };

    template<typename T_Attribute>
    struct GetDevicePtr
    {
        template<typename T_ValueType>
        HINLINE void operator()(T_ValueType& dest, T_ValueType& src)
        {
            typedef typename Resolve_t<T_Attribute>::type type;

            type* ptr = NULL;
            type* srcPtr = src.getIdentifier(T_Attribute()).getPointer();
            if (srcPtr != NULL)
            {
                CUDA_CHECK(cudaHostGetDevicePointer(&ptr, srcPtr, 0));
            }
            dest.getIdentifier(T_Attribute()) = PMacc::VectorDataBox<type>(ptr);
        }
    };

    /** Allocator using CUDA mapped memory */
    struct MappedMemAllocator
    {
        template<typename T>
        static T* alloc(const size_t size, const T& = T())
        {
            T* ptr = NULL;
            if (size != 0)
                CUDA_CHECK(cudaHostAlloc(&ptr, size * sizeof(T), cudaHostAllocMapped));
            return ptr;
        }

        template<typename T>
        static void free(T* ptr)
        {
            if(ptr)
                CUDA_CHECK(cudaFreeHost(ptr));
        }
    };

    /** Allocator using new[]/delete[] */
    struct ArrayAllocator
    {
        template<typename T>
        static T* alloc(const size_t size, const T& = T())
        {
            return (size) ? new T[size] : nullptr;
        }

        template<typename T>
        static void free(T* ptr)
        {
            delete[] ptr;
        }
    };

    template<typename T_Attribute, class T_Allocator>
    struct AllocMemory
    {
        template<typename ValueType >
        HINLINE void operator()(ValueType& v1, const size_t size) const
        {
            typedef typename Resolve_t<T_Attribute>::type type;

            v1.getIdentifier(T_Attribute()) = PMacc::VectorDataBox<type>(T_Allocator::alloc(size, type()));
        }
    };

    template<typename T_Attribute, class T_Allocator>
    struct FreeMemory
    {

        template<typename ValueType >
        HINLINE void operator()(ValueType& value) const
        {
            T_Allocator::free(value.getIdentifier(T_Attribute()).getPointer());
        }
    };

}  // namespace plugins
}  // namespace xrt
