#get dictionary for storing log information
function getLog(m0,nc,maxit)
    data=Dict{Symbol, Any}()
    data[:residuals]=zeros(Float64,maxit,m0)
    data[:feastResidual]=zeros(Float64,maxit)
    data[:eigenvalues]=zeros(Complex128,maxit,m0)
    data[:ninside]=zeros(Int64,maxit)
    data[:iterations]=0
    data[:insideIndices]=Dict{Int64, Any}()
    data[:shiftIndex]=Dict{Complex128,Int64}()
    
    data[:linResiduals]=zeros(Float64,m0,nc,maxit)
    data[:linIts]=zeros(Int64,m0,nc,maxit)
    
    data[:hlinResiduals]=zeros(Float64,m0,nc,maxit)
    data[:hlinIts]=zeros(Int64,m0,nc,maxit)
    
    return data
end


########################
# CORE FEAST FUNCTIONS #
########################

#These functions are one- and two-sided implementations of the FEAST algorithm. They are essentially outlines that encompass all possible variations of the FEAST algorithm.

#They are used by passing functions to them to perform the essential operations of the FEAST algorithm, such as evaluating the eigenvector residual and calculating the contour integrand.

#core one-sided FEAST algorithm; this function is used for all variations of FEAST
function feast_core(Tf,integrand,rrsolve,x0,nc,emid,ra,rb,eps::Float64,maxit; log=false, insideEps=1e-2, verbose=1, set_nin=false)
    #Tf: eigenvector residual function. E.g. for linear problems Tf(l,x) = B*x*diagm(l)-A*x
    #integrand: function that returns integral integrand evaluated at point z in complex plane, applied to a vector x. E.g. for linear FEAST integrand(z,x,l)=\(z*B-A,x)
    #rrsolve: function for returning eigenvalue and eigenvector approximations from rayleigh ritz. E.g. for linear FEAST rrsolve(Q)=eig(Q'*A*Q,Q'*B*Q)
    #x0: initial guess for eigenvectors
    #nc: number of quadrature points for numerical integration
    #emid: center of integration ellipse
    # ra,rb: radii of integration ellipse
    # eps: convergence tolerance
    # maxit: maximum number of FEAST iterations
    # insideEps: residual threshold for determining whether or not an eigenvalue inside the contour is spurious
    # log: if true, returns dictionary object with logged convergence information
    # verbose(0-1): how much information to print out
    # set_nin: if equal to an integer, sets the number of eigenvalues inside the contour to 'set_nin'; used for diagnostic purposes

    #get shapes of arrays:
	(n,m0)=size(x0)

    #storing convergence info:
    data=getLog(m0,nc,maxit)
    
    #data[:linsysresidual]=zeros(Float64,1)
    #data[:linsysits]=zeros(Int64,1)  

    #start with initial guess
    x=copy(x0)
    
    #initialize eigenvalue estimates:
	lest=zeros(Complex64,m0,1)
	
	#initialize iteration number:
	it=0
	
    #get quadrature points from trapezoidal rule:
    (gk,wk)=trapezoidal(nc)
    offset=pi/nc #offset angle of first quadrature point; makes sure it isn't on real axis for hermitian problems
    
    #save indices for contour points
    for k in 1:nc
        #curve parametrization angle:
        theta=gk[k]*pi+pi+offset
	   
        #quadrature point:
        z=emid+((ra+rb)/2)*exp(im*theta)+((ra-rb)/2)*exp(-1.0*im*theta)
        data[:shiftIndex][z]=k
    end
    
    #Initialize FEAST subspace:
    Q=copy(x)

    res=1.0 #initial residual
	while res>eps && it<maxit
		it=it+1 #update number of iterations

        data[:iterations]=it

        #rayleigh ritz
        tic()
        (lest,x)=rrsolve(Q)
        dt=toq()
        
        verbose>1 && println("   ritz  $dt s")
        
        
        #sort everything by real part of eigenvalue
        p=sortperm(real(lest))
        lest=lest[p]
        x=x[:,p]
        
        #store convergence data
        #indicate which eigenvalues are inside the contour
        data[:insideIndices][it]=getInsideIndex(lest,emid,ra,rb)
        #store all the eigenvalues
        data[:eigenvalues][it,:]=lest
       
        #calculate eigenvector residuals
        resvecs=Tf(lest,x)
        for i in 1:m0
            data[:residuals][it,i]=norm(resvecs[:,i])/norm(x[:,i])
        end
        
        #find the largest residual inside the contour
        res,ninside=getMaxResInside(lest,x,resvecs,emid,ra,rb;inEps=insideEps, set_nin=set_nin)  

        data[:feastResidual][it]=res  
        data[:ninside][it]=ninside
        
        verbose>0 && println("  $it  res=$res  minres=$(minimum(data[:residuals][it,:]))  nin=$ninside")
        
        if(res<eps)
            if(log)
	            return (lest,x,data)
	        else
	            return (lest,x)
	        end
        end


        #apply contour integral to get FEAST subspace:
	    Q=zeros(x)	
	    for k in 1:nc
	    #Q=@parallel (+) for k in 1:nc
	        #integration curve is an ellipse centered at emid, with radii ra and rb
	        
	        #curve parametrization angle:
	        theta=gk[k]*pi+pi+offset
	        
	        #quadrature point:
	        z=emid+((ra+rb)/2)*exp(im*theta)+((ra-rb)/2)*exp(-1.0*im*theta)

            #integrand evaluated at quadrature point z
            
            #println("     linear system $k")
            tic()
            Qk=integrand(z,x,lest,data,resvecs)
            dt=toq()
            verbose>1 && println("     linear system $k  $dt s")

            #add integrand contribution to quadrature:
	        Q=Q+wk[k]*(((ra+rb)/2)*exp(im*theta)-((ra-rb)/2)*exp(-1.0*im*theta))*Qk
	        #    wk[k]*(((ra+rb)/2)*exp(im*theta)-((ra-rb)/2)*exp(-1.0*im*theta))*Qk	
        end  

        #Orthonormalize FEAST subspace to avoid spurious eigenpairs:
        (Qq,Rq)=qr(Q)
        Q[:]=Qq

	end

    if(log)
        return (lest,x,data)
    else
        return (lest,x)
    end

end


#core two-sided FEAST algorithm 
#Difference from symmetric FEAST: have to solve for left and right eigenvectors simultaneously and biorthogonalize them
function feastNS_core(Tf,hTf,integrand,hintegrand,rrsolvens,biortho,x0,y0,nc,emid,ra,rb,eps,maxit; insideEps=1e-2, log=false)
    #Tf: eigenvector residual function. E.g. for linear problems Tf(l,x) = B*x*diagm(l)-A*x
    #hTf: hermitian conjugate transpose of residual function
    #integrand: function that returns integral integrand evaluated at point z in complex plane, applied to a vector x. E.g. for linear FEAST integrand(z,x,l)=\(z*B-A,x)
    #rrsolvens: function for returning eigenvalue and eigenvector approximations from rayleigh ritz. E.g. for linear FEAST rrsolve(Q)=eig(Q'*A*Q,Q'*B*Q)
    #x0: initial guess for right eigenvectors
    #y0: initial guess for left eigenvectors 
    #nc: number of quadrature points for numerical integration
    #emid: center of integration ellipse
    # ra,rb: radii of integration ellipse
    # eps: convergence tolerance
    # maxit: maximum number of FEAST iterations
    # insideEps: residual threshold for determining whether or not an eigenvalue inside the contour is spurious
    # log: if true, returns dictionary object with logged convergence information

    #get shapes of arrays:
	(n,m0)=size(x0)
	
	#storing convergence info:
    data=getLog(m0,nc,maxit)

    #start with initial guess
    x=copy(x0)
    y=copy(y0)
    
    #initialize eigenvalue estimates:
	lest=zeros(Complex64,m0,1)
	
	#initialize iteration number:
	it=0
	
    #get quadrature points from trapezoidal rule:
    (gk,wk)=trapezoidal(nc)
    offset=pi/nc #offset angle of first quadrature point; makes sure it isn't on real axis for hermitian problems
    
    #save indices for contour points
    for k in 1:nc
        #curve parametrization angle:
        theta=gk[k]*pi+pi+offset

        #quadrature point:
        z=emid+((ra+rb)/2)*exp(im*theta)+((ra-rb)/2)*exp(-1.0*im*theta)
        data[:shiftIndex][z]=k
    end
    
    
    #Initialize FEAST subspaces:
    Q=copy(x)
    R=copy(y)

    res=1.0 #initial residual
	while res>eps && it<maxit
		it=it+1 #update number of iterations
		data[:iterations]=it

        #Biorthogonalize Q and R subspaces
        (Q,R)=biortho(Q,R)

        #rayleigh ritz
        (y,lest,x)=rrsolvens(Q,R)
       
        #calculate eigenvector residuals
        resvecs=Tf(lest,x)
        hresvecs=hTf(lest,y)
        
        #store convergence data
        #indicate which eigenvalues are inside the contour
        data[:insideIndices][it]=getInsideIndex(lest,emid,ra,rb)
        #store all the eigenvalues
        data[:eigenvalues][it,:]=lest

        #calculate eigenvector residuals
        resvecs=Tf(lest,x)
        for i in 1:m0
            data[:residuals][it,i]=norm(resvecs[:,i])/norm(x[:,i])
        end
        
        
        #find the largest residual inside the contour
        res,ninside=getMaxResInside(lest,x,resvecs,emid,ra,rb; inEps=insideEps)
        data[:feastResidual][it]=res
        data[:ninside][it]=ninside
        
        println("  $it  res=$res  minres=$(minimum(data[:residuals][it,:]))  nin=$ninside")
        if(res<eps)
            if(log)
                    return (y,lest,x,data)
                else
                    return (y,lest,x)
            end
        end

        #apply contour integral to get FEAST subspace:
	    Q=zeros(x)
	    R=zeros(y)
	    for k in 1:nc
	        #integration curve is an ellipse centered at emid, with radii ra and rb
	        
	        #curve parametrization angle:
	        theta=gk[k]*pi+pi+offset
	        
	        #quadrature point:
	        z=emid+((ra+rb)/2)*exp(im*theta)+((ra-rb)/2)*exp(-1.0*im*theta)

            #integrand evaluated at quadrature point z
            Qk=integrand(z,x,lest,data,resvecs)
            Rk=hintegrand(z,y,lest,data,hresvecs)
  
            #add integrand contribution to quadrature:
	        Q=Q+wk[k]*(((ra+rb)/2)*exp(im*theta)-((ra-rb)/2)*exp(-1.0*im*theta))*Qk
	        R=R+(wk[k]*(((ra+rb)/2)*exp(im*theta)-((ra-rb)/2)*exp(-1.0*im*theta)))'*Rk
        end  

	end


        if(log)
            return (y,lest,x,data)
        else
            return (y,lest,x)
        end

end
