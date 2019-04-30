using Random
using LinearAlgebra
Random.seed!(843)

include("feast.jl")
using ..feastLinear
using IterativeSolvers

n=100

alpha=0.99 # #determine how nonsymmetric our matrix is; alpha=1.0 makes a symmetric matrix
#generate B-matrix eigenvectors and eigenvalues
#vectors:
R=rand(n,n)
(Xb,R)=qr(R)
Xb=Xb+(1-alpha)*ones(n,n)
Yb=inv(Xb)'
#values:
Lb=rand(n)*0.5 .+ 0.5 #random positive numbers

#generate A-matrix eigenvectors:
R=rand(n,n)
(X,R)=qr(R)
X=alpha*X+(1-alpha)*ones(n,n)
Y=inv(X)'

#generate A-matrix eigenvalues
L1=zeros(n)
for i in 1:n
    L1[i]=i
end

#generate eigenvalue problem matrices
A=Xb*diagm(0 => sqrt.(Lb))*X*diagm(0 => L1)*Y'*diagm(0 => sqrt.(Lb))*Yb'
B=Xb*diagm(0 => Lb)*Yb'

#FEAST parameters
emid=1.0+0.0*im #contour center
ra=0.5 #contour radius 1
rb=0.5 #contour radius 2
nc=8 #number of contour points
m0=2 #subspace dimension
eps=1e-9 #residual convergence tolerance
maxit=100 #maximum FEAST iterations
x0=rand(n,m0) #eigenvector initial guess
y0=copy(x0)

alphals=1e-16 #linear system solve accuracy
isMaxit=30 #maximum number of linear system iterations
#(ye,le,xe)=nseig(A,B)
println("Standard FEAST:")
(ye,le,xe,data)=feastNS_linear(A,B,x0,y0,nc,emid,ra,rb,eps,maxit; log=true)
println("Iterative FEAST:")
(ye,le,xe)=ifeastNS_linear(A,B,x0,y0,alphals,isMaxit,nc,emid,ra,rb,eps,maxit)

p=sortperm(real(le))
println(le[p])
