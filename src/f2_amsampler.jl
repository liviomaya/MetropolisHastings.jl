struct AMResults
    algorithm::String
    sample::Matrix{Float64}
    density::Vector{Float64}
    mode::Vector{Float64}
    N::Int64
    λ::Float64
    γ::Vector{Float64}
    α::Vector{Float64}
    μ::Matrix{Float64}
    Σ::Array{Float64,3}
end

@with_kw struct AMOptions
    N::Int64
    γ::Vector{Float64}
    λ::Float64
    μ0::Vector{Float64}
    Σ0::Matrix{Float64}
    burn::Int64
end

function updateΣ!(sqΣ::Matrix{Float64}, μ::Vector{Float64}, γ::Float64,
    v::Vector{Float64})
    a = sqrt(1 - γ)
    isqΣ = inv(sqΣ)
    m = norm(isqΣ * (v - μ))^2
    c = (m == 0) ? 0.0 : (sqrt(1 + (γ / (1 - γ)) * m) - 1) / m
    sqΣ .= a * sqΣ + c * a * ((v - μ) * (v - μ)') * isqΣ'
    return
end

function updapprox!(μ::Vector{Float64}, γ::Float64, x::Vector{Float64})
    μ .= μ + γ * (x - μ)
    return
end

function amsampler(f::Function, x0::Vector{Float64}, opt::AMOptions)

    @unpack N, γ, λ, μ0, Σ0, burn = opt

    # assert sizes are consistent
    P = length(x0)
    @assert length(γ) == N "Vector γ must be N × 1."
    @assert size(Σ0) == (P, P) "Sizes of μ0 and Σ0 must be consistent."

    # prelims
    Gaussian = MvNormal(diagm(0 => ones(P)))
    sqλ = sqrt(λ)

    # pre-allocate output
    sample = zeros(N, P)
    density = zeros(N)
    α = zeros(N)
    μ = zeros(N, P)
    Σ = zeros(N, P, P)

    # simulation
    x = x0
    d = f(x)
    α_n = 0.0
    μ_n = μ0
    sqΣ_n = collect(cholesky(Σ0).L)
    α[1] = α_n
    μ[1, :] = μ_n
    Σ[1, :, :] = sqΣ_n * sqΣ_n'

    display(md"**Adaptive Metropolis (AM) Sampler**")
    progress = Progress(N, dt=1.0, barlen=25)
    for n in 1:N
        # draw candidate and accept/reject
        jump = sqλ * sqΣ_n * rand(Gaussian)
        xc = x .+ jump # candidate
        dc = f(xc)
        acceptrate = min(1, exp(dc - d))
        if rand() < acceptrate
            x = xc
            d = dc
        end
        sample[n, :] = x
        density[n] = d

        # update proposal
        if n < N
            γ_n = γ[n+1]
            α_n = α_n + γ_n * (acceptrate - α_n)
            updapprox!(μ_n, γ_n, x)
            updateΣ!(sqΣ_n, μ_n, γ_n, x)
            α[n+1] = α_n
            μ[n+1, :] = μ_n
            Σ[n+1, :, :] = sqΣ_n * sqΣ_n'
        end
        next!(progress,
            showvalues=[(:LogDensity, d), (:Expec_Acceptance, α_n)])

    end
    println("")
    println("")

    # burn and build results
    algorithm = "Adaptive Metropolis"
    sample = sample[burn+1:end, :]
    density = density[burn+1:end]
    α = α[burn+1:end]
    γ = γ[burn+1:end]
    μ = μ[burn+1:end, :]
    Σ = Σ[burn+1:end, :, :]
    mode = sample[argmax(density), :]

    R = AMResults(algorithm, sample, density, mode, N - burn, λ, γ, α, μ, Σ)
    return R
end

summarytable(R::AMResults) = summarytable(R.sample, R.density, R.Σ[end, :, :])