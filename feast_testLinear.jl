using LinearAlgebra
using Random

Random.seed!(843)

include("feast.jl")
using ..feastLinear
using IterativeSolvers

n=100

#generate B-matrix eigenvectors and eigenvalues
#vectors:
R=rand(n,n)
(Xb,R)=qr(R)
#values:
Lb=rand(n)*0.5+0.5 #random positive numbers

#generate A-matrix eigenvectors:
R=rand(n,n)
(X1,R)=qr(R)

#generate A-matrix eigenvalues
L1=zeros(n)
for i in 1:n
    L1[i]=i
end

#generate eigenvalue problem matrices
A=Xb*diagm(sqrt.(Lb))*X1*diagm(L1)*X1'*diagm(sqrt.(Lb))*Xb'
B=Xb*diagm(Lb)*Xb'

#FEAST parameters
emid=1.0+0.0*im #contour center
ra=0.5 #contour radius 1
rb=0.5 #contour radius 2
nc=6 #number of contour points
m0=2 #subspace dimension
eps=1e-5 #residual convergence tolerance
maxit=100 #maximum FEAST iterations
x0=rand(n,m0) #eigenvector initial guess

println("Standard FEAST")
(lf,xf)=feast_linear(A,B,x0,nc,emid,ra,rb,eps,maxit)
p=sortperm(real(lf))
println(lf[p])

println("Iterative FEAST BiCGSTAB")
alpha=0.01 #relative accuracy of linear system solves
isMaxit=220  #maximum number of linear system iterations per FEAST iteration
(lf,xf,data)=ifeast_linear(A,B,x0,alpha,isMaxit,nc,emid,ra,rb,eps,maxit;log=true)
p=sortperm(real(lf))
println(lf[p])
