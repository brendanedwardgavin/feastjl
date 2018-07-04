include("iterativeSolvers.jl")

#one-sided FEAST for linear eigenvalue problems
function feast_linear(A,B,x0,nc,emid,ra,rb,eps,maxit;log=false, insideEps=1e-2, set_nin=false)
    (n,m0)=size(x0)
    
    integrand(z,x,lest,data,resvecs)=\(z*B-A,B*x)
    
    function rrsolve(Q)
        Aq=Q'*A*Q
        Bq=Q'*B*Q
       
        (le,xe)=eig(Aq,Bq)
        return (le,Q*xe)
    end
    
    Tf(l,x)=(B*x*diagm(l)-A*x)
    
    return feast_core(Tf,integrand,rrsolve,x0,nc,emid,ra,rb,eps,maxit;log=log,insideEps=insideEps, set_nin=set_nin)
    
end


#two-sided FEAST for linear eigenvalue problems
function feastNS_linear(A,B,x0,y0,nc,emid,ra,rb,eps,maxit; log=false,insideEps=1e-2)
    (n,m0)=size(x0)
    
    integrand(z,x,lest,data,resvecs)=\(z*B-A,B*x)
    hintegrand(z,y,lest,hdata,hresvecs)=\((z*B-A)',B'*y)
    
    function rrsolvens(Q,R)
        Aq=R'*A*Q
        Bq=R'*B*Q
        (ye,le,xe)=nseig(Aq,Bq)
        return (R*ye,le,Q*xe)
    end
    
    Tf(l,x)=(B*x*spdiagm(l)-A*x)
    hTf(l,y)=B'*y*spdiagm(l)'-A'*y
    
    #use SVD to B-biorthogonalize subspaces
    function biortho(Q,R)
        Bq=R'*B*Q
        (u,s,v)=svd(Bq)
        Y=R*u*diagm(1./sqrt.(s))
        X=Q*v*diagm(1./sqrt.(s))
        return (X,Y)
    end
    
    return feastNS_core(Tf,hTf,integrand,hintegrand,rrsolvens,biortho,x0,y0,nc,emid,ra,rb,eps,maxit; log=log, insideEps=insideEps)
    
end


#generalized inexact FEAST algorithm with minres
function ifeast_linear(A,B,x0,alpha,isMaxit,nc,emid,ra,rb,eps,maxit;log=false,insideEps=1e-2, verbose=1)
    (n,m0)=size(x0)
    
    function integrand(z,x,lest,data,resvecs)
        nc=data[:shiftIndex][z]
        M=(z*B-A)
        #b=(B*x*diagm(lest)-A*x)
        #normb=maximum(sqrt.(sum(abs.(b).^2,1)))

        int=zeros(Complex128,n,m0)
        maxits=0
        rhs=convert(Array{Complex128,2},resvecs)
        
        
        
        for i in 1:m0
            #(int[:,i],history)=bicgstabl(M,resvecs[:,i],1,max_mv_products=isMaxit,tol=alpha,initial_zero=true,log=true)
            #(int[:,i],history)=idrs(M,rhs[:,i];maxiter=isMaxit,tol=alpha,log=true)
            #(int[:,i],history)=gmres(M,resvecs[:,i],restart=isMaxit,tol=alpha,initially_zero=true,maxiter=isMaxit,log=true)
            (int[:,i],history)=minres(M,resvecs[:,i],tol=alpha,initially_zero=true,maxiter=isMaxit,log=true)
            nlinits=size(history[:resnorm],1)
            data[:linIts][i,nc,data[:iterations]]=nlinits
            #data[:linResiduals][i,nc,data[:iterations]]=history[:resnorm][nlinits]
            data[:linResiduals][i,nc,data[:iterations]]=norm(resvecs[:,i]-M*int[:,i])/norm(resvecs[:,i])
            
            if(nlinits>maxits)
                maxits=nlinits
            end
        end
        #int=\(M,resvecs)
        #println("      linits=$maxits")
        #int=zbicgstabBlock(M,resvecs,zeros(n,m0),isMaxit,alpha)
        return (x-int)*spdiagm(1./(z.-lest))
    end
    
    function rrsolve(Q)
        Aq=Q'*A*Q
        Bq=Q'*B*Q
        (le,xe)=eig(Aq,Bq)
        return (le,Q*xe)
    end
    
    Tf(l,x)=(B*x*spdiagm(l)-A*x)
    
    return feast_core(Tf,integrand,rrsolve,x0,nc,emid,ra,rb,eps,maxit;log=log,insideEps=insideEps, verbose=verbose)
    
end

#generalized ifeast with bicgstab
function ifeast_linearBicgstab(A,B,x0,alpha,isMaxit,nc,emid,ra,rb,eps,maxit;log=false,insideEps=1e-2,verbose=1)
    (n,m0)=size(x0)
    
    function integrand(z,x,lest,data,resvecs)
        nc=data[:shiftIndex][z]
        M1=(z*B-A)
        P=speye(n)
        M=P*M1*P
        rhs=P*resvecs
        #b=(B*x*diagm(lest)-A*x)
        #normb=maximum(sqrt.(sum(abs.(b).^2,1)))

        int=zeros(Complex128,n,m0)
        
        for i in 1:m0
            (int[:,i],history)=bicgstabl(M,resvecs[:,i],1,max_mv_products=isMaxit,tol=alpha,initial_zero=true,log=true)
            #(int[:,i],history)=idrs(M,rhs[:,i];maxiter=isMaxit,tol=alpha,log=true)
            #(int[:,i],history)=gmres(M,resvecs[:,i],restart=isMaxit,tol=alpha,initially_zero=true,maxiter=isMaxit,log=true)
            #(int[:,i],history)=minres(M,resvecs[:,i],tol=alpha,initially_zero=true,maxiter=isMaxit,log=true)
            nlinits=size(history[:resnorm],1)
            data[:linIts][i,nc,data[:iterations]]=nlinits
            data[:linResiduals][i,nc,data[:iterations]]=history[:resnorm][nlinits]
        end
        
        
        return (x-int)*spdiagm(1./(z.-lest))
    end
    
    function rrsolve(Q)
        Aq=Q'*A*Q
        Bq=Q'*B*Q
        (le,xe)=eig(Aq,Bq)
        return (le,Q*xe)
    end
    
    Tf(l,x)=(B*x*spdiagm(l)-A*x)
    
    return feast_core(Tf,integrand,rrsolve,x0,nc,emid,ra,rb,eps,maxit;log=log,insideEps=insideEps,verbose=verbose)
    
end


#inexact two-sided ifeast with my own bicgstab implementation
function ifeastNS_linear(A,B,x0,y0,alpha,isMaxit,nc,emid,ra,rb,eps,maxit; log=false,insideEps=1e-2)
    (n,m0)=size(x0)
    
    #integrand(z,x,lest,data,resvecs)=\(z*B-A,B*x)
    #hintegrand(z,y,lest,hdata,hresvecs)=\((z*B-A)',B'*y)
    function integrand(z,x,lest,data,resvecs)
        M=z*B-A
        rhs=B*x*spdiagm(lest)-A*x
        int0=zeros(Complex128,n,m0)
        int=zeros(Complex128,n,m0)
        int=zbicgstabBlock(M,resvecs,int0,isMaxit,alpha)
        for i in 1:m0
            #(int[:,i],history)=gmres(M,rhs[:,i],restart=isMaxit,tol=alpha,initially_zero=true,maxiter=isMaxit,log=true)
            #(int[:,i],history)=bicgstabl(M,rhs[:,i],2,max_mv_products=isMaxit,tol=alpha,initial_zero=true,log=true)
        end
        
        #int=\(M,resvecs)
        return (x-int)*spdiagm(1./(z.-(lest)))
    end
    
    function hintegrand(z,y,lest,hdata,hresvecs)
        M=(z*B-A)'
        rhs=B'*y*spdiagm(lest)'-A'*y
        int0=zeros(Complex128,n,m0)
        int=zeros(Complex128,n,m0)
        int=zbicgstabBlock(M,hresvecs,int0,isMaxit,alpha)
        for i in 1:m0
            #(int[:,i],history)=gmres(M,rhs[:,i],restart=isMaxit,tol=alpha,initially_zero=true,maxiter=isMaxit,log=true)
            
            #(int[:,i],history)=bicgstabl(M,rhs[:,i],2,max_mv_products=isMaxit,tol=alpha,initial_zero=true,log=true)
        end
        #int=\(M,hresvecs)
        return (y-int)*spdiagm(1./(z'.-conj.(lest)))
    end
    
    function rrsolvens(Q,R)
        Aq=R'*A*Q
        Bq=R'*B*Q
        (ye,le,xe)=nseig(Aq,Bq)
        return (R*ye,le,Q*xe)
    end
    
    Tf(l,x)=(B*x*spdiagm(l)-A*x)
    hTf(l,y)=B'*y*spdiagm(l)'-A'*y
    
    #use SVD to B-biorthogonalize subspaces
    function biortho(Q,R)
        Bq=R'*B*Q
        (u,s,v)=svd(Bq)
        Y=R*u*diagm(1./sqrt.(s))
        X=Q*v*diagm(1./sqrt.(s))
        return (X,Y)
    end
    
    return feastNS_core(Tf,hTf,integrand,hintegrand,rrsolvens,biortho,x0,y0,nc,emid,ra,rb,eps,maxit; log=log, insideEps=insideEps)
    
end


