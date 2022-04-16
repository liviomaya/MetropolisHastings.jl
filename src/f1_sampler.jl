

"""
    MHOptions(calcmode::Bool = false
    calcmodeiter::Int64 = 100
    calcmarginal::Bool = true
    training::Bool = true
    drawstraining::Int64 = 100
    burntraining::Int64 = 100
    draws::Int64 = 200
    burn::Int64 = 100
    blocks::Int64 = 1
    jumpstd::Float64 = 1.0
    jumpscale::Float64 = 1.0
    geweketau::Float64 = 0.025)

Constructor with options for Metropolis-Hastings algorithm. Set `calcmode=true`
to maximize the kernel (`likelihood` times `prior`) prior to simulation. Set
`training=true` to simulate a training sample used to infer a covariance matrix
for the jumping process. Parameter `jumpscale` scales the covariance matrix of
the jumping distribution.
"""
@with_kw struct MHOptions
    calcmode::Bool = false
    calcmodeiter::Int64 = 100
    calcmarginal::Bool = true
    training::Bool = true
    drawstraining::Int64 = 100
    burntraining::Int64 = 100
    draws::Int64 = 200
    burn::Int64 = 100
    blocks::Int64 = 1
    jumpstd::Float64 = 1.0
    jumpscale::Float64 = 1.0
    geweketau::Float64 = 0.025
end



"""
    MHResults(mode::Vector{Float64}
        sample::Matrix{Float64}
        densitykernel::Vector{Float64}
        marginal::Float64
        acceptrate::Float64
        acceptrateblock::Float64)

"""
struct MHResults
    mode::Vector{Float64}
    sample::Matrix{Float64}
    densitykernel::Vector{Float64}
    marginal::Float64
    acceptrate::Float64
    acceptrateblock::Float64
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

function randomblocks(p::Int64, B::Int64, N::Int64)
    blocks = Vector{Vector{Float64}}(undef, N)
    blocks = [randomblocks(p, B) for _ in 1:N]
    return blocks
end

randomblocks(p::Int64, B::Int64, N::Int64) = [randomblocks(p, B) for _ in 1:N]


function mhsampler(N::Int64,
    firstdraw::Vector{Float64},
    prior::Function,
    likelihood::Function,
    jumpcov::Matrix{Float64},
    B::Int64)

    # prelims
    P = length(firstdraw)
    @assert P >= B "More blocks than parameters"
    jumpdist = MvNormal(zeros(P), jumpcov)
    blockarray = randomblocks(P, B, N)

    # initialize output
    drawarray = zeros(N, P)
    densityarray = zeros(N)

    # simulation
    statedraw = firstdraw
    statedensity = prior(firstdraw) + likelihood(firstdraw)
    progress = Progress(N, dt=1, barlen=25)
    for n in 1:N
        jump = rand(jumpdist)
        for block in blockarray[n]
            draw = copy(statedraw)
            draw[block] .+= jump[block]
            cprior = prior(draw)
            (cprior == -Inf) && continue
            density = cprior + likelihood(draw)
            (density == -Inf) && continue
            acceptprob = min(1, exp(density - statedensity))
            if acceptprob < 1 # avoid costly draw if unnecessary
                (rand() > acceptprob) && continue
            end
            statedraw = draw
            statedensity = density
        end
        drawarray[n, :] = statedraw
        densityarray[n] = statedensity
        next!(progress)
    end

    return drawarray, densityarray
end

function getmode(guess::Vector{Float64},
    prior::Function,
    likelihood::Function,
    iterations::Int64)

    # objective function (posterior)
    function objective(x::Vector{Float64})
        cprior = prior(x)
        (cprior == -Inf) && return Inf
        obj = -(cprior + likelihood(x))
        return obj
    end

    println("Numerical approximation of posterior mode...")
    R = optimize(objective, guess, BFGS(),
        Optim.Options(iterations=iterations,
            show_trace=true,
            show_every=100))
    !Optim.converged(R) && println("Search for posterior mode interrupted.")
    mymode::Vector{Float64} = R.minimizer

    return mymode
end

function summarytable(sample::VecOrMat{U},
    density::Vector{Float64},
    jumpcov::VecOrMat{H}) where {U<:Real,H<:Real}

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
            "5%-Quantile",
            "95%-Quantile"],
        formatters=ft_printf("%5.1f", 2:8),
        vlines=[3],
        crop=:none)


    nothing
end


function mhsampler(draw::Vector{Float64},
    prior::Function,
    likelihood::Function;
    options::MHOptions=MHOptions())

    @unpack calcmode, training, draws, burn,
    blocks, jumpstd, jumpscale, calcmarginal,
    calcmodeiter = options

    # calculate mode
    mmode = Vector{Float64}(undef, 0)
    if calcmode
        mmode = getmode(draw, prior, likelihood, calcmodeiter)
        firstdraw = mmode
        println(" ")
        println(" ")
    else
        firstdraw = draw
    end

    # initial covariance matrix of jumping distribution
    N = length(draw)
    jumpcov = jumpstd^2 * collect(I(N))

    # training sample 
    if training
        @unpack drawstraining, burntraining = options
        println("SIMULATE TRAINING SAMPLE")
        trainingsample, trainingdensities = mhsampler(drawstraining,
            firstdraw, prior, likelihood, jumpcov, N)

        summarytable(trainingsample, trainingdensities, jumpcov)
        acceptrate(trainingsample, N)
        println(" ")
        println(" ")

        trainingsample = trainingsample[burntraining+1:end, :]
        jumpcov = cov(trainingsample)
        firstdraw = trainingsample[end, :]
    end

    # sampler
    println("SIMULATE MAIN SAMPLE")
    jumpcovmain = jumpscale^2 * jumpcov
    sample, densities = mhsampler(draws, firstdraw, prior,
        likelihood, jumpcovmain, blocks)

    sample = sample[burn+1:end, :]
    densities = densities[burn+1:end]

    # re-calculate mode
    modeindex = argmax(densities)
    mmode = sample[modeindex, :]

    # summary table and acceptance rate
    summarytable(sample, densities, jumpcovmain)
    acceptratedraw, acceptrateblock = acceptrate(sample, blocks)

    # calculate marginal 
    if calcmarginal
        @unpack geweketau = options
        gewfunc = geweke(sample, geweketau)
        marg = marginal(sample, densities, gewfunc)
    else
        marg = NaN
    end
    println(" ")
    println(" ")

    # build MHResults
    results = MHResults(mmode, sample, densities, marg,
        acceptratedraw, acceptrateblock)

    return results
end

function mhsampler(draw::Float64,
    prior::Function,
    likelihood::Function;
    options::MHOptions=MHOptions())

    prmat = p -> prior(p[1])
    lkmat = p -> likelihood(p[1])

    results = mhsampler([draw], prmat, lkmat, options=options)
    return results
end