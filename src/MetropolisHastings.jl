module MetropolisHastings

export MHOptions, MHResults, mhsampler
export S2MHOptions, s2mhsampler
export AMOptions, AMResults, amsampler
export GAMOptions, GAMResults, gamsampler
export summarytable

include("f0_header.jl")
include("f1_mhsampler.jl")
include("f2_amsampler.jl")
include("f3_gamsampler.jl")

end # module
