srand(843)
using IterativeSolvers
include("mybicgstab.jl")

n=1000

#generate eigenvectors:
R=rand(Complex128,n,n)
(X1,R)=qr(R)

#generate eigenvalues
L1=zeros(n,n)
for i in 1:n
    L1[i,i]=i
end

#generate eigenvalue problem matrices
A=X1*L1*X1'

z=2.5+1.0*im
M=(z*eye(n)-A)

x=rand(n,1)
b=M*x

x0=zeros(n,1)
maxit=1000
eps=1e-5


xest=zbicgstabBlock(M,b,x0,maxit,eps)
#xest=bicgstabl(M,b,2,max_mv_products=500,tol=1e-5,initial_zero=true) #zbicgstabBlock(A,b,x0,maxit,eps)

res=norm(b-M*xest)/norm(b)
println("res=$res")
