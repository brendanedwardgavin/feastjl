using LinearAlgebra
using Random

Random.seed!(843)

include("feast.jl")
using ..feastNonlinear,..feastUtil
using IterativeSolvers

using IterativeSolvers

#eigenvalue problem dimension:
n=500

#eigenvector subspaces:
R=rand(n,n)
(X1,R)=qr(R)
R=rand(n,n)
(X2,R)=qr(R)

#quadratic eigenvalues:
L1=zeros(n,n)
L2=zeros(n,n)
#eigenvalues are 1,2,3,...,2n
for i in 1:n
    L1[i,i]=i
    L2[i,i]=i+n
end

#generate quadratic factor matrices:
B1=X1*L1*X1'
B2=X2*L2*X2'

#quadratic eigenvcalue problem matrices: T(z)=z^2*M+z*D+K
M=Matrix(I,n,n)
D=-B1-B2
K=B1*B2

#FEAST parameters
m0=5 #subspace dimension
x0=rand(ComplexF64,n,m0) #eigenvector initial guess
nc=4 #number of integration quadrature points
emid=50 #center point of contour
ra=1.5 #contour radius 1
rb=1.5 #contour radius 2
eps=1e-8 #convergence residual tolerance
maxit=50 #maximum number of FEAST iterations


#use quadratic FEAST
println("Using quadratic feast")
(lest,xest)=nlfeast_quad(M,D,K,x0,nc,emid,ra,rb,eps,maxit)


#Using Beyn
println("Using Beyn")
Tf(z)=(z^2*M+z*D+K)
(lest,xest)=beyn(Tf,x0,nc,emid,ra,rb)
ncmax=100
res=M*xest*diagm(0 => lest.^2)+D*xest*diagm(0 => lest)+K*xest
maxres=getMaxResInside(lest,xest,res,emid,ra,rb)
println("res=$maxres")

#Using general nonlinear feast, i.e. feast with beyn
println("Using general nlfeast")
Tf(z)=(z^2*M+z*D+K)
(lest, xest)=nlfeast_beyn(Tf,x0,nc,emid,ra,rb,eps,maxit)

#using iterative general nonlinear FEAST with GMRES
println("Using general iterative nlfeast")
alpha=0.0 #target accuracy for linear system solutions
maxitls = 50 # maximum linear system iterations
(lest, xest)=inlfeast_beyn(Tf,x0,alpha,maxitls,nc,emid,ra,rb,eps,maxit)
