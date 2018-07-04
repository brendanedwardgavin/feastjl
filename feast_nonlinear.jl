
#FEAST for general nonlinear eigenvalue problems, using Beyn's algorithm for Rayleigh Ritz.
function nlfeast_beyn(Tf,x0,nc,emid,ra,rb,eps,maxit;log=false)
    #Tf is single value nonlinear residual T(z)

    #pardiso for sparse systems
    #ps=MKLPardisoSolver()
    #set_matrixtype!(ps,Pardiso.COMPLEX_NONSYM)

    #integrand = (I-T^-1(z)T(l))x(zI-l)^-1
    function integrand(z,x,lest,data,resvecs)
            (n,m0)=size(x)
            TinvTxl=zeros(Complex128,n,m0)
            Tfz=Tf(z)
            
            if(typeof(Tfz)==SparseMatrixCSC{Complex{Float64},Int64} || typeof(Tfz)==SparseMatrixCSC{Float64,Int64} ) #use pardiso for sparse matrix
                #TinvTxl=solve(ps,Tfz,R)
                TinvTxl=\(Tfz,resvecs)
            else
                TinvTxl=\(Tfz,resvecs)
            end
            
            gamma=spdiagm(1./(z.-lest))
            Qk=x*gamma-TinvTxl*gamma
            return Qk
    end
    
    
    #solve projected problem Q'*T(z)Q=0 using Beyn's algorithm
    function rrsolve(Q)
        (n,m0)=size(Q)
        
        v0=rand(Complex128,m0,m0)
        Tfr(z)=Q'*Tf(z)*Q

        (lest,xq)=beyn(Tfr,v0,5*nc,emid,ra,rb)
        
        return (lest,Q*xq)
    end
    
    #Block method for applying residual
    function blockTf(l,x)
        (n,m0)=size(x)
        out=zeros(x)
        for i in 1:m0
            out[:,i]=Tf(l[i])*x[:,i]
        end
        return out
    end

    return feast_core(blockTf,integrand,rrsolve,x0,nc,emid,ra,rb,eps,maxit;log=log)
end




#FEAST for quadratic eigenvalue problem
function nlfeast_quad(M,D,K,x0,nc,emid,ra,rb,eps,maxit;log=false)
    (n,m0)=size(x0)
    
    #pardiso for sparse systems
    #ps=MKLPardisoSolver()
    
    Tfm(z)=M*z^2+z*D+K
    
    #Q=integral (I-T^-1(z)T(X,Lambda))X(zI-Lambda)^-1 dz
    function integrand(z,x,lest,data,resvecs)
            (n,m0)=size(x)
            
            Txl=M*x*diagm(lest.^2)+D*x*diagm(lest)+K*x
            #Txl=M*x2+D*x1+K*x
            #Txl=resvecs
            Tz=z^2*M+z*D+K
            
            TinvTxl=\(Tz,Txl)
            
            gamma=diagm(1./(z.-lest))
            Qk=x*gamma-TinvTxl*gamma
            return Qk
    end
    
    
    #solve reduced nonlinear problem with linearization
    #select the m0 solutions that are closest to the center of the contour
    rrsolve(Q)=Quadrrsolve(Q,M,D,K,emid,ra,rb)
        
    Tf(l,x)=M*x*diagm(l.^2)+D*x*diagm(l)+K*x
    
    return feast_core(Tf,integrand,rrsolve,x0,nc,emid,ra,rb,eps,maxit,log=log)
end

