

"""
    MHOptions(N, Σ, B, γ, burn)

Constructor with keywords (see `Parameters` package). `MHOptions` groups parameters of the `mhsampler` function.

### Fields

- `N::Int64=100`: Size of the Monte-Carlo simulation.

- `Σ::Matrix{Float64}=[1.0]`: Covariance matrix of the Gaussian proposal distribution.

- `B::Int64=1`: Number of parameter blocks per draw (blocks assigned randomly in each iteration).

- `γ::Float64=0.01`: Stepsize for calculation of expected acceptance rate `α`: `α(n+1) = α(n) + γ [AR(n) -  α(n)]`, where `AR(n)` is the acceptance rate of draw `t`.

- `burn::Int64=0`: Number of draws discarded at the end of the sampling procedure.
"""
@with_kw struct MHOptions
    N::Int64 = 100
    Σ::Matrix{Float64} = diagm(0 => [1])
    B::Int64 = 1
    γ::Float64 = 0.01
    burn::Int64 = 0
end

"""
    MHResults(algorithm, sample, density, mode, N, B, Σ, γ, α)

Constructor with results from Metropolis-Hastings algorithm (`mhsampler` function).

### Fields

- `algorithm::String`: name of the algorithm.

- `sample::Matrix{Float64}`: simulated sample, with observation in the first dimension.

- `density::Vector{Float64}`: density evaluated at the corresponding point in `sample`.

- `mode::Vector{Float64}`: mode (draw with largest density).

- `N::Int64`: Size of the sample after burning.

- `B::Int64`: Number of blocks in the sampler.

- `Σ::Matrix{Float64}:` covariance matrix of the proposal Gaussian distribution.

- `γ::Float64=0.01`: Stepsize for calculation of expected acceptance rate `α`: `α(n+1) = α(n) + γ [AR(n) -  α(n)]`, where `AR(n)` is the acceptance rate of draw `t`.

- `α::Vector{Float64}`: expected acceptance rate.
"""
struct MHResults
    algorithm::String
    sample::Matrix{Float64}
    density::Vector{Float64}
    mode::Vector{Float64}
    N::Int64
    B::Int64
    Σ::Matrix{Float64}
    γ::Float64
    α::Vector{Float64}
end


function randomblocks(p::Int64, B::Int64)
    # p : number of elements to be sorted
    # B : number of blocks
    blocks = Vector{Vector{Int64}}(undef, B)
    n = rand(p)
    II = sortperm(n)
    bsize = Int64(floor(p / B))
    for b in 1:B
        i0 = 1 + bsize * (b - 1)
        i1 = (b == B) ? p : b * bsize
        blocks[b] = II[i0:i1]
    end
    return blocks
end

# function randomblocks(p::Int64, B::Int64, N::Int64)
#     blocks = Vector{Vector{Float64}}(undef, N)
#     blocks = [randomblocks(p, B) for _ in 1:N]
#     return blocks
# end

randomblocks(p::Int64, B::Int64, N::Int64) = [randomblocks(p, B) for _ in 1:N]

"""
    R = mhsampler(f, x0, opt)

Simulate from density `f` using the Metropolis-Hastings algorithm.

### Arguments

- `f::Function`: Log-density used in the simulation.

- `x0::Vector{Float64}`: Initial condition of the chain.

- `opt::MHOptions`: Constructor with parameters that govern the sampler.

### Output

- `R::MHResults`: Constructor with results of the sampler.

"""
function mhsampler(f::Function, x0::Vector{Float64}, opt::MHOptions)

    @unpack N, Σ, B, γ, burn = opt

    # check size consistency
    P = length(x0)
    @assert P >= B "More blocks than parameters"

    # prelims
    Gaussian = MvNormal(diagm(0 => ones(P)))
    sqΣ = collect(cholesky(Σ).L)
    blockarray = randomblocks(P, B, N)

    # pre-allocate output
    sample = zeros(N, P)
    density = zeros(N)
    α = zeros(N)

    # simulation
    x = x0
    d = f(x)
    α_n = 0.0
    display(md"**Metropolis-Hastings (MH) Sampler**")
    progress = Progress(N, dt=1.0, barlen=25)
    for n in 1:N
        # draw candidate
        jump = sqΣ * rand(Gaussian)

        # evaluate block by block
        acceptrate = 0.0
        for block in blockarray[n]
            xc = copy(x)
            xc[block] .+= jump[block]
            dc = f(xc)
            acceptrate_b = min(1, exp(dc - d))
            if rand() < acceptrate_b
                x = xc
                d = dc
            end
            acceptrate += acceptrate_b / B
        end
        sample[n, :] = x
        density[n] = d

        # update expected acceptance rate
        if n < N
            α_n = α_n + γ * (acceptrate - α_n)
            α[n+1] = α_n
        end
        next!(progress,
            showvalues=[(:LogDensity, d), (:Expec_Acceptance, α_n)])
    end
    println("")
    println("")

    # build results
    algorithm = "Metropolis Hastings"
    sample = sample[burn+1:end, :]
    density = density[burn+1:end]
    α = α[burn+1:end]
    mode = sample[argmax(density), :]
    R = MHResults(algorithm, sample, density, mode, N - burn, B, Σ, γ, α)

    return R
end



"""
    S2MHOptions(N0, N1, λ, Σ, B, γ, burn0, burn1)

Constructor with keywords (see `Parameters` package). `S2MHOptions` groups parameters of the `s2mhsampler` function.

### Fields

- `N0::Int64=100`: Size of the Monte-Carlo simulation in the first step.

- `N1::Int64=100`: Size of the Monte-Carlo simulation in the second step.

- `λ::Float64=1.0`: Global scaling parameter for the second step. The covariance of the proposal distribution in the second step is `λ (Ψ + ϵ)` where `Ψ` is the covariance matrix of the first-step, and `ϵ` is a small diagonal matrix that ensures it is positive semi-definite. 

- `Σ::Matrix{Float64}=[1.0]`: Covariance matrix of the Gaussian proposal distribution for the first-step simulation.

- `B::Int64=1`: Number of parameter blocks per draw (blocks assigned randomly in each iteration).

- `γ::Float64=0.01`: Stepsize for calculation of expected acceptance rate `α`: `α(n+1) = α(n) + γ [AR(n) -  α(n)]`, where `AR(n)` is the acceptance rate of draw `t`.

- `burn0::Int64=0`: Number of draws discarded at the end of the sampling procedure in the first step.

- `burn1::Int64=0`: Number of draws discarded at the end of the sampling procedure in the second step.
"""
@with_kw struct S2MHOptions
    N0::Int64 = 100
    N1::Int64 = 100
    λ::Float64 = 1.0
    Σ::Matrix{Float64} = diagm(0 => [1])
    B::Int64 = 1
    γ::Float64 = 0.01
    burn0::Int64 = 0
    burn1::Int64 = 0
end


"""
    r, R = s2mhsampler(f, x0, opt)

Simulate from density `f` using the two-step Metropolis-Hastings algorithm.

### Arguments

- `f::Function`: Log-density used in the simulation.

- `x0::Vector{Float64}`: Initial condition of the chain.

- `opt::S2MHOptions`: Constructor with parameters that govern the sampler.

### Output

- `r::MHResults`: Constructor with results of the sampler in the first step.

- `R::MHResults`: Constructor with results of the sampler in the second step.

"""
function s2mhsampler(f::Function, x0::Vector{Float64}, opt::S2MHOptions)

    @unpack N0, N1, λ, Σ, B, γ, burn0, burn1 = opt
    @assert N0 > 0 "N0 must be non-zero."

    # step 1: training sample
    opt_step1 = MHOptions(N=N0, Σ=Σ, B=B, γ=γ, burn=burn0)
    display(md"**Training Sample**")
    r0 = mhsampler(f, x0, opt_step1)

    # step 2: main sample
    P = length(x0)
    x0_step2 = r0.mode
    Σ_step2 = λ * (cov(r0.sample, dims=1) + diagm(0 => 0.00001 * ones(P)))
    opt_step2 = MHOptions(N=N1, Σ=Σ_step2, B=B, γ=γ, burn=burn1)
    display(md"**Main Sample**")
    r1 = mhsampler(f, x0_step2, opt_step2)

    return r0, r1
end

function summarytable(sample::Matrix{Float64},
    density::Vector{Float64},
    jumpcov::Matrix{Float64})

    T, N = size(sample)
    checkaccept(col) = 100 * count(col[2:end] .!= col[1:end-1]) / (T - 1)
    quantile5(col) = quantile(col, 0.05)
    quantile95(col) = quantile(col, 0.95)

    acceptrate = mapslices(checkaccept, sample, dims=1)[:]
    jumpstd = sqrt.(diag(jumpcov))
    stdpar = std(sample, dims=1)[:]
    meanval = mean(sample, dims=1)[:]
    modeval = sample[argmax(density), :]
    quantile5val = mapslices(quantile5, sample, dims=1)[:]
    quantile95val = mapslices(quantile95, sample, dims=1)[:]

    tab = DataFrame(:Parameter => 1:N,
        :AcceptRate => acceptrate,
        :JumpStd => jumpstd,
        :Mode => modeval,
        :Mean => meanval,
        :StdDeviation => stdpar,
        :Quantile5 => quantile5val,
        :Quantile95 => quantile95val)

    pretty_table(tab,
        title="Parameter Analysis",
        tf=tf_simple,
        header=["Parameter",
            "Accept Rate",
            "Jump Std",
            "Mode",
            "Mean",
            "St Dev",
            "q5%",
            "q95%"],
        formatters=ft_printf("%5.2f", 2:8),
        vlines=[3],
        crop=:none)


    return nothing
end

summarytable(R::MHResults) = summarytable(R.sample, R.density, R.Σ)