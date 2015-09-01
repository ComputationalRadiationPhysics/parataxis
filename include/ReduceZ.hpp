namespace xrt {

    /**
     * Returns a 2D slice from a 3D or 2D DataBox
     */
    template<uint32_t T_simDim>
    struct ReduceZ;

    template<>
    struct ReduceZ<2>
    {
        template<class T_Box>
        static T_Box
        get(T_Box data, uint32_t zOffset)
        {
            return data;
        }
    };
    template<>
    struct ReduceZ<3>
    {
        template<class T_Box>
        static auto
        get(T_Box data, uint32_t zOffset)
        -> decltype(data[zOffset])
        {
            return data[zOffset];
        }
    };

}  // namespace xrt
