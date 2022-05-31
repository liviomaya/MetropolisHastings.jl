function geweke(X::Matrix{Float64}, τ::Float64)
    P = size(X, 2)
    μ = mean(X, dims=1)[:]
    V = cov(X)
    threshold = quantile(Chisq(P), 1 - τ)

    if !isposdef(V)
        println("Sample covariance matrix not invertible.")
        return x -> NaN
    end

    function f(x)
        kernel = ((x .- μ)'*inv(V)*(x.-μ))[1]
        (kernel > threshold) && return -Inf
        term1 = log(1 - τ)
        term2 = (P / 2) * log(2pi)
        term3 = (1 / 2) * log(det(V))
        y = -term1 - term2 - term3 - 0.5 * kernel
        return y
    end

    return f
end
geweke(X::Vector{Float64}, τ::Float64) = geweke(X * ones(1, 1), τ)

function marginal(draw::Matrix{Float64}, density::Vector{Float64}, f::Function)
    far = mapslices(f, draw, dims=2)[:]
    integrand = far .- density
    deleteat!(integrand, far .== -Inf)
    c = mean(integrand) # numerical adjustment to avoid integral = Inf
    marg = -log(mean(exp.(integrand .- c))) - c
    println("Marginal Density: log P(Y) = $(round(marg, digits=2))")
    return marg
end
marginal(draw::Vector{Float64}, density::Vector{Float64}, f::Function) =
    marginal(draw * ones(1, 1), density, f)

function confidenceinterval(X::Matrix{Float64}, p::Float64)
    intfunc(x) = quantile(x, [p, 1 - p])
    intervals = mapslices(intfunc, X, dims=1)
    return intervals
end
confidenceinterval(X::Vector{Float64}, p::Float64) =
    confidenceinterval(X * ones(1, 1), p)[:]

function acceptrate(X::Matrix{Float64}, B::Int64=1)
    N = size(X, 1)
    naccept = 0
    for n in 2:N
        naccept += (X[n, :] != X[n-1, :])
    end
    drawrate = naccept / (N - 1)
    blockrate = 1 - (1 - drawrate)^(1 / B)
    println("Acceptance rate (draw level) : $(round(100*drawrate, digits=1))%")
    println("Acceptance rate (block level) : $(round(100*blockrate, digits=1))%")
    return drawrate, blockrate
end
acceptrate(X::Vector{Float64}, B::Int64=1) = acceptrate(X * ones(1, 1), B)
