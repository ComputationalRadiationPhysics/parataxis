namespace xrt {

    /**
     * Returns a 2D slice from a 3D or 2D DataBox
     */
    template<unsigned T_simDim>
    struct ReduceZ;

    template<>
    struct ReduceZ<2>
    {
        template<class T_Box>
        static T_Box
        get(T_Box data, unsigned zOffset)
        {
            return data;
        }
    };
    template<>
    struct ReduceZ<3>
    {
        template<class T_Box>
        static auto
        get(T_Box data, unsigned zOffset)
        -> decltype(data[zOffset])
        {
            return data[zOffset];
        }
    };

}  // namespace xrt
