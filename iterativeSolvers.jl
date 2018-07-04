import Base.diag

function diag(a::Number)
	return a
end

function diagmmult(A,l)
    (n,m)=size(A)
    Anew=zeros(A)
    for i in 1:m
        Anew[:,i]=A[:,i]*l[i]
    end
    
    return Anew
end

function bdot(a,b)
    (n,m)=size(a)
    out=zeros(Complex128,m)
    for i in 1:m
        out[i]=dot(a[:,i],b[:,i])
    end
    
    return out
end

function zbicgstabBlock(A,b,x0,maxit,eps)
	#matrix multiply:

	(n,m)=size(x0)

        r=b-A*x0
	s=copy(r)
	y0=copy(r)
	
	x=copy(x0)

	#inner product:
	#delta=(y0'*r)[1]
	
	#delta=diag(y0'*r)
	delta=bdot(y0,r)
	
	#phi=(y0'*(A*s))[1]/delta

	for i in 1:maxit
        #println("    BCG $i")
		#matrix multiply:
		As=A*s
		#inner product
		#phi=(y0'*(As))[1]/delta
		#phi=diag(y0'*As)./delta
		phi=bdot(y0,As)./delta
		
		#omega=1/phi
		omega=1./phi
		w=r-As*spdiagm(omega)
		#w=r-diagmmult(As,omega)

		#matrix multiply:
		Aw=A*w
		#inner product:
		#chi=((Aw)'*w)[1]/norm(Aw)^2
		#chi=diag(Aw'*w)./diag(Aw'*Aw)
        chi=bdot(Aw,w)./bdot(Aw,Aw)

		#r=w-chi*Aw
		r=w-Aw*spdiagm(chi)
        #r=w-diagmmult(Aw,chi)
			
		rs=sqrt.(bdot(r,r))
		#println("   $i     $(maximum(abs.(rs)))   $(norm(Aw))")
		#if(norm(r)<eps)
		if(maximum(abs.(rs))<eps)
			#println("r finish: $(norm(r))")
			#println("     bicgstab its $i")
			return x
		end

		#x=x+s*omega+w*chi
		x=x+s*spdiagm(omega)+w*spdiagm(chi)
		#x=x+diagmmult(s,omega)+diagmmult(w,chi)
		
		deltaold=delta
		#inner product:
		#delta=(y0'*r)[1]
		delta=bdot(y0,r)#diag(y0'*r)		
		#psi=-1.0*omega*delta/(deltaold*chi)
		psi=-1.0*omega.*delta./(deltaold.*chi)
		s=r-(s-(As)*spdiagm(chi))*spdiagm(psi)
		#s=r-(s-diagmmult(As,chi.*psi))
		#print("     $(norm(s))\n")
		ss=sqrt.(bdot(s,s))
		if(maximum(abs.(ss))<eps)	
		#if(norm(s)<eps)
			#println("s finish: $(norm(s))")
			#println("     bicgstab its $i")
			return x
		end


	end

	#println("          no converge: $(norm(r))")
	return x
end

