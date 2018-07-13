module feastUtil
    export linquad,trapezoidal,getMaxResInside,Quadrrsolve,beyn,nseig
    include("util.jl")
end

module feastCore
    importall feastUtil
    export feast_core,feastNS_core
    include("feast_core.jl")
end

module feastLinear
    importall feastCore,feastUtil,IterativeSolvers
    export feast_linear,ifeast_linear,feastNS_linear,ifeastNS_linear
    include("feast_linear.jl")
end

module feastNonlinear
    importall feastCore,feastUtil
    export nlfeast_quad,nlfeast_beyn
    include("feast_nonlinear.jl")
end
