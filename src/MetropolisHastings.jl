module MetropolisHastings

export MHOptions, MHResults, mhsampler, confidenceinterval

include("f0_header.jl")
include("f1_sampler.jl")
include("f2_functions.jl")

end # module
