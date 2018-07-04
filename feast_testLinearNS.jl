srand(843)

#include("feast.jl")
include("NLEutil.jl")
include("feast_core.jl")
include("feast_linear.jl")

using IterativeSolvers

n=1000

alpha=1.0
#generate B-matrix eigenvectors and eigenvalues
#vectors:
R=rand(n,n)
(Xb,R)=qr(R)
Xb=Xb+(1-alpha)*ones(n,n)
Yb=inv(Xb)'
#values:
Lb=rand(n)*0.5+0.5 #random positive numbers

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
A=Xb*diagm(sqrt.(Lb))*X*diagm(L1)*Y'*diagm(sqrt.(Lb))*Yb'
B=Xb*diagm(Lb)*Yb'

#FEAST parameters
emid=1.0+0.0*im #contour center
ra=0.5 #contour radius 1
rb=0.5 #contour radius 2
nc=4 #number of contour points
m0=2 #subspace dimension
eps=1e-9 #residual convergence tolerance
maxit=100 #maximum FEAST iterations
x0=rand(n,m0) #eigenvector initial guess
y0=copy(x0)

alphals=1e-16
isMaxit=30
#(ye,le,xe)=nseig(A,B)
#(ye,le,xe,data)=feastNS_linear(A,B,x0,y0,nc,emid,ra,rb,eps,maxit; log=true)
(ye,le,xe)=ifeastNS_linear(A,B,x0,y0,alphals,isMaxit,nc,emid,ra,rb,eps,maxit)

p=sortperm(real(le))
println(le[p])
