module feastUtil
    using LinearAlgebra
    using Random
    export linquad,trapezoidal,getMaxResInside,Quadrrsolve,beyn,nseig,getInsideIndex
    include("util.jl")
end

module feastCore
    using LinearAlgebra
    using Random
    using ..feastUtil
    export feast_core,feastNS_core
    include("feast_core.jl")
end

module feastLinear
    using LinearAlgebra
    using Random
    using ..feastCore,..feastUtil,IterativeSolvers
    export feast_linear,ifeast_linear,feastNS_linear,ifeastNS_linear
    include("feast_linear.jl")
end

module feastNonlinear
    using LinearAlgebra
    using Random
    using ..feastCore,..feastUtil
    export nlfeast_quad,nlfeast_beyn
    include("feast_nonlinear.jl")
end
