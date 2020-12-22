using ForwardDiff
using LinearAlgebra

function my_kron(A, B)
	d = size(A)[1]
	K = zeros((d^2, d^2))
	for i in 1:d
		for j in 1:d
			for k in 1:d
				for l in 1:d
					K[(i-1) .* d .+ j, (k-1) .* d .+ l] = A[i, k] * B[j, l]
				end
			end
		end
	end
	return(K)
end

function my_box(A, B)
	d = size(A)[1]
	K = zeros((d^2, d^2))
	for i in 1:d
		for j in 1:d
			for k in 1:d
				for l in 1:d
					K[(i-1) .* d .+ j, (k-1) .* d .+ l] = A[i, l] * B[j, k]
				end
			end
		end
	end
	return(K)
end

# make a spd matrix X_
p = 2
X_ = rand(p,p)
Y_ = rand(p,p)
X_sym_ = 0.5 * (X_ + X_')
Y_sym_ = 0.5 * (Y_ + Y_')
lambda_min = abs(minimum(eigvals(X_sym_))) + 10e-3
lambda_min_Y = abs(minimum(eigvals(Y_sym_))) + 10e-3
X_ = X_sym_ + lambda_min * Matrix{Int}(I, p, p) 
Y_ = Y_sym_ + lambda_min_Y * Matrix{Int}(I, p, p) 
println("X_: ", X_)
println("Y_: ", Y_)
println("Matrix X_ is symmetric positive definite: ", minimum(eigvals(X_)) > 0)
println("Matrix Y_ is symmetric positive definite: ", minimum(eigvals(Y_)) > 0)

########## check whether kronecker and box product are correctly defined
cond = my_kron(X_, Y_) == kron(X_, Y_)
println("my kron works: ", cond)
cond2 = kron(X_, Y_) * my_box(Matrix{Int}(I, p, p) , Matrix{Int}(I, p, p)) == my_box(X_, Y_) 
println("my box working: ", cond2)


#matric inverse
f(X) = inv(X)
df(X) = - kron(inv(X)', inv(X))
cond = df(X_) ≈ ForwardDiff.jacobian(f, X_)
cond2 = df(X_) - ForwardDiff.jacobian(f, X_)
println("dX inv(X) ~ -kron(inv(X)', inv(X)): ", cond)  # true 

#transpose of matric inverse
f1(X) = inv(X)'
df1(X) = - my_box(inv(X), inv(X))
cond = df1(X_) ≈ ForwardDiff.jacobian(f1, X_)
cond2 = df1(X_) - ForwardDiff.jacobian(f1, X_)
println("dX inv(X)' ~ -my_box(inv(X)', inv(X)): ", cond)  # false

#trace of the square
f(X) = tr(X^2)
df(X) = 2X'
cond = df(X_) ≈ ForwardDiff.gradient(f, X_)  
println("dX tr(X^2) ~ 2X.T: ", cond) # true

# setup some random matrices

n = 3
p = 2

A = rand(p,p)
X_sym = 0.5 * (A + A')
lambda_min = abs(minimum(eigvals(X_sym))) + 10e-3
X_spd = X_sym + lambda_min * Matrix{Int}(I, p, p) 
println("Matrix X is symmetric positive definite: ", minimum(eigvals(X_spd)) > 0)

B = rand(p,p)
V_sym = 0.5 * (B + B')
lambda_V_min = abs(minimum(eigvals(V_sym))) + 10e-3
V_spd = V_sym + lambda_V_min * Matrix{Int}(I, p, p) 
println("Matrix V is symmetric positive definite: ", minimum(eigvals(V_spd)) > 0)


"""
#verify the Laplace approximation of the Wishart in the normal basis

log_f_W(X) = (n-p-1)/2 * logdet(X) - tr(inv(V_spd) * X)/2

####first derivative
d_log_f_W(X) =  (n-p-1)/2 * inv(X)' - inv(V_spd)/2
cond = d_log_f_W(X_spd) ≈ ForwardDiff.gradient(log_f_W, X_spd)
println("first derivative of the log pdf Wishart: ", cond)

####second derivative
d2_log_f_W(X) = -(n-p-1)/2 * my_box(inv(X)', inv(X))
cond = d2_log_f_W(X_spd) ≈ ForwardDiff.hessian(log_f_W, X_spd)
println("Hessian of the log pdf Wishart: ", cond)



#verify the LA of the Wishart in the sqrtm basis

log_f_W_sqrtm(X) = (n-p) * logdet(X) - tr(inv(V_spd) * X^2)/2

function der_XA(A)
	d = size(A)[1]
	K = zeros((d^2, d^2))
	for i in 1:d
		for j in 1:d
			for k in 1:d
				for l in 1:d
					if j == l
						K[(i-1) .* d .+ j, (k-1) .* d .+ l] = A[k, i]
					else
						K[(i-1) .* d .+ j, (k-1) .* d .+ l] = 0
					end
				end
			end
		end
	end
	return(K)
end

function der_AX(A)
	d = size(A)[1]
	K = zeros((d^2, d^2))
	for i in 1:d
		for j in 1:d
			for k in 1:d
				for l in 1:d
					if i == k
						K[(i-1) .* d .+ j, (k-1) .* d .+ l] = A[j, l]
					else
						K[(i-1) .* d .+ j, (k-1) .* d .+ l] = 0
					end
				end
			end
		end
	end
	return(K)
end

function der_XA_T(A)
	d = size(A)[1]
	K = zeros((d^2, d^2))
	for i in 1:d
		for j in 1:d
			for k in 1:d
				for l in 1:d
					if i == l
						K[(i-1) .* d .+ j, (k-1) .* d .+ l] = A[j, k]
					else
						K[(i-1) .* d .+ j, (k-1) .* d .+ l] = 0
					end
				end
			end
		end
	end
	return(K)
end

function der_AX_T(A)
	d = size(A)[1]
	K = zeros((d^2, d^2))
	for i in 1:d
		for j in 1:d
			for k in 1:d
				for l in 1:d
					if j == k
						K[(i-1) .* d .+ j, (k-1) .* d .+ l] = A[i, l]
					else
						K[(i-1) .* d .+ j, (k-1) .* d .+ l] = 0
					end
				end
			end
		end
	end
	return(K)
end

#### verify my derivatives
f(X) = V_spd * X
cond = der_AX(V_spd) ≈ ForwardDiff.jacobian(f, X_spd)
cond2 = my_kron(Matrix{Int}(I, p, p), V_spd) ≈ ForwardDiff.jacobian(f, X_spd)
println("AX derivative: ", cond) #true
println("AX der == kron(I, V_spd): ", cond2) #true

#same for transpose
f(X) = (V_spd * X)'
cond = der_AX_T(V_spd) ≈ ForwardDiff.jacobian(f, X_spd)
cond2 = my_box(V_spd, Matrix{Int}(I, p, p)) ≈ ForwardDiff.jacobian(f, X_spd)
println("(AX).T derivative: ", cond) #true
println("(AX).T der == box(V_spd, I): ", cond2) #true

f(X) = X * V_spd
cond = der_XA(V_spd) ≈ ForwardDiff.jacobian(f, X_spd)
cond2 = my_kron(V_spd, Matrix{Int}(I, p, p)) ≈ ForwardDiff.jacobian(f, X_spd)
println("XA derivative: ", cond) #true
println("XA der == kron(V_spd, I): ", cond2) #true

#same for transpose
f(X) = (X * V_spd)'
cond = der_XA_T(V_spd) ≈ ForwardDiff.jacobian(f, X_spd)
cond2 = my_box(Matrix{Int}(I, p, p), V_spd) ≈ ForwardDiff.jacobian(f, X_spd)
println("(XA).T derivative: ", cond) #false
println("(XA).T der == box(I, V_spd): ", cond2) #true


####first derivative

###### verifiy individual steps

#ONE
log_f_W_sqrtm_one(X) = (n-p) * logdet(X)

d_log_f_W_sqrtm_one(X) = (n-p) * inv(X)'
cond = d_log_f_W_sqrtm_one(X_spd) ≈ ForwardDiff.gradient(log_f_W_sqrtm_one, X_spd)
println("first derivative of the log pdf Wishart in sqrtm basis ONE: ", cond) #True

#TWO
log_f_W_sqrtm_two(X) = tr(inv(V_spd) * X^2)/2
d_log_f_W_sqrtm_two(X) = (X * inv(V_spd) + inv(V_spd) * X)' / 2
cond = d_log_f_W_sqrtm_two(X_spd) ≈ ForwardDiff.gradient(log_f_W_sqrtm_two, X_spd)
println("first derivative of the log pdf Wishart in sqrtm basis TWO: ", cond) #True

#same = d_log_f_W_sqrtm_two(X_spd) ==  (inv(V_spd) * X_spd)'
#println("V^-1 X: ", (inv(V_spd) * X_spd)')
#println("(XV^-1 + V^-1X).T/2: ", d_log_f_W_sqrtm_two(X_spd))
#println("same?: ", same)

d_log_f_W_sqrtm(X) = (n-p) * inv(X) - (X * inv(V_spd) + inv(V_spd) * X)' / 2
cond = d_log_f_W_sqrtm(X_spd) ≈ ForwardDiff.gradient(log_f_W_sqrtm, X_spd)
println("first derivative of the log pdf Wishart in sqrtm basis: ", cond) #false


####second derivative

###### verify individual steps

#ONE
d2_log_f_W_sqrtm_one(X) = -(n-p) * my_box(inv(X), inv(X))
cond = d2_log_f_W_sqrtm_one(X_spd) ≈ ForwardDiff.hessian(log_f_W_sqrtm_one, X_spd)
println("Hessian of the log pdf Wishart in sqrtm basis ONE: ", cond) #True

#TWO
d2_log_f_W_sqrtm_two(X) = 1/2 * (my_box(inv(V_spd), Matrix{Int}(I, p, p)) + my_box(Matrix{Int}(I, p, p), inv(V_spd)))
cond = d2_log_f_W_sqrtm_two(X_spd) ≈ ForwardDiff.hessian(log_f_W_sqrtm_two, X_spd) 
println("Hessian of the log pdf Wishart in sqrtm basis TWO: ", cond) #True

#both
d2_log_f_W_sqrtm(X) = -(n-p) * my_box(inv(X), inv(X)) - 1/2 * (my_box(inv(V_spd), Matrix{Int}(I, p, p)) + my_box(Matrix{Int}(I, p, p), inv(V_spd)))
cond = d2_log_f_W_sqrtm(X_spd) ≈ ForwardDiff.hessian(log_f_W_sqrtm, X_spd) 
println("Hessian of the log pdf Wishart in sqrtm basis: ", cond) #True


#verify the Laplace approximation of the inverse Wishart in the normal basis

log_f_IW(X) = -(n+p+1)/2 * logdet(X) - tr(V_spd * inv(X))/2
log_f_IW_one(X) = -(n+p+1)/2 * logdet(X) 
log_f_IW_two(X) = -tr(V_spd * inv(X))/2


####first derivative

### part one
d_log_f_IW_one(X) =  -(n+p+1)/2 * inv(X)' 
cond = d_log_f_IW_one(X_spd) ≈ ForwardDiff.gradient(log_f_IW_one, X_spd)
println("first derivative of the log pdf inverse Wishart ONE: ", cond) #True

### part two
d_log_f_IW_two(X) = 1/2 * (inv(X) * V_spd * inv(X))'
cond = d_log_f_IW_two(X_spd) ≈ ForwardDiff.gradient(log_f_IW_two, X_spd)
println("first derivative of the log pdf inverse Wishart TWO: ", cond) #True

### the entirety
d_log_f_IW(X) =  -(n+p+1)/2 * inv(X)' + 1/2 * (inv(X) * V_spd * inv(X))'
cond = d_log_f_IW(X_spd) ≈ ForwardDiff.gradient(log_f_IW, X_spd)
println("first derivative of the log pdf inverse Wishart: ", cond) #True


####second derivative

#some tests

function der_XBX1(X, B)
	d = size(B)[1]
	K = zeros((d^2, d^2))
	for i in 1:d
		for j in 1:d
			for k in 1:d
				for l in 1:d
					if l == j
						K[(i-1) .* d .+ j, (k-1) .* d .+ l] = (B*X)[k, i]
					else
						K[(i-1) .* d .+ j, (k-1) .* d .+ l] = 0
					end
				end
			end
		end
	end
	return(K)
end

function der_XBX2(X, B)
	d = size(B)[1]
	K = zeros((d^2, d^2))
	for i in 1:d
		for j in 1:d
			for k in 1:d
				for l in 1:d
					if k == i
						K[(i-1) .* d .+ j, (k-1) .* d .+ l] = (X*B)[j, l]
					else
						K[(i-1) .* d .+ j, (k-1) .* d .+ l] = 0
					end
				end
			end
		end
	end
	return(K)
end

f_test(X) = X*V_spd*X
#d_f(X) = my_kron(V_spd*X, Matrix{Int}(I, p, p)) + my_kron(Matrix{Int}(I, p, p), X*V_spd)
d_f(X) = der_XBX1(X_spd, V_spd) + der_XBX2(X_spd, V_spd)
cond = d_f(X_spd) ≈ ForwardDiff.jacobian(f_test, X_spd)
println("test XBX: ", cond) #True

println("same? XBX1: ", der_XBX1(X_spd, V_spd) == my_kron(X_spd*V_spd, Matrix{Int}(I, p, p))) #true
println("same? XBX2: ", der_XBX2(X_spd, V_spd) == my_kron(Matrix{Int}(I, p, p), X_spd*V_spd)) #true

f_test2(X) = (X*V_spd*X)'
d_f2(X) = (my_box(X*V_spd, Matrix{Int}(I, p, p)) + my_box(Matrix{Int}(I, p, p), X*V_spd))
cond = d_f2(X_spd) ≈ ForwardDiff.jacobian(f_test2, X_spd)
println("test (XBX).T: ", cond) #True

#part one
d2_log_f_IW_one(X) = (n+p+1)/2 * my_box(inv(X), inv(X))
cond = d2_log_f_IW_one(X_spd) ≈ ForwardDiff.jacobian(d_log_f_IW_one, X_spd)
println("Hessian of the log pdf inverse Wishart TWO: ", cond) #True

#part two 
d2_log_f_IW_two(X) = -1/2 * (my_box(inv(X)*V_spd, Matrix{Int}(I, p, p)) + my_box(Matrix{Int}(I, p, p), inv(X)*V_spd)) * my_kron(inv(X), inv(X))
cond = d2_log_f_IW_two(X_spd) ≈ ForwardDiff.jacobian(d_log_f_IW_two, X_spd)
println("Hessian of the log pdf inverse Wishart TWO: ", cond) #True

#the actual Hessian
d2_log_f_IW(X) = (n+p+1)/2 * my_box(inv(X), inv(X)) - 1/2 * (my_box(inv(X)*V_spd, Matrix{Int}(I, p, p)) + my_box(Matrix{Int}(I, p, p), inv(X)*V_spd)) * my_kron(inv(X), inv(X))
cond = d2_log_f_IW(X_spd) ≈ ForwardDiff.hessian(log_f_IW, X_spd)
println("Hessian of the log pdf inverse Wishart: ", cond) #True


#verify the LA of the inverse Wishart in the sqrtm basis

log_f_IW_sqrtm(X) = -(n+p) * logdet(X) - tr(V_spd * inv(X)^2)/2
log_f_IW_sqrtm_one(X) = -(n+p) * logdet(X)
log_f_IW_sqrtm_two(X) = -tr(V_spd * inv(X)^2)/2

####first derivative

## part one
d_f_IW_sqrtm_one(X) = -(n+p) * inv(X)'
cond = d_f_IW_sqrtm_one(X_spd) ≈ ForwardDiff.gradient(log_f_IW_sqrtm_one, X_spd)
println("Jacobian of the Inverse Hessian in the sqrtm basis ONE: ", cond) #True

## tests
#f1(X) = tr(V_spd * X^2)
#d_f1(X) = (X*V_spd + V_spd*X)'
#cond = d_f1(X_spd) ≈ ForwardDiff.gradient(f1, X_spd)
#println("Test1: ", cond) 

#println("same square: ", X_spd * X_spd == X_spd^2)
#println("same traces? 1 ", tr(V_spd * inv(X_spd)^2) == tr(inv(X_spd)^2 * V_spd))
#println("same traces? 2 ", tr(V_spd * inv(X_spd)^2) ≈ tr(inv(X_spd) * V_spd * inv(X_spd)))
#println("same traces? 3 ", tr(V_spd * inv(X_spd)^2) ≈ tr(inv((X_spd * inv(V_spd) * X_spd))))
#println("same traces? 4 ", tr(V_spd * inv(X_spd)^2) ≈ tr(inv(X_spd^2 * inv(V_spd))))


#f2(X) = tr(inv(X' * V_spd *X))
#f2(X) = tr(inv(X) * V_spd * inv(X))
#d_f2(X) = -2 * (V_spd * X * inv(X' * V_spd * X)) * inv(X' * V_spd * X)
#d_f2(X) = -(inv(X) * V_spd * inv(X) * inv(X))' - (inv(X) * inv(X) * V_spd * inv(X))' 
#cond = d_f2(X_spd) ≈ ForwardDiff.gradient(f2, X_spd)
#println("Test2: ", cond) 

## part two
d_f_IW_sqrtm_two(X) = 1/2 * ((inv(X) * V_spd * inv(X) * inv(X))' + (inv(X) * inv(X) * V_spd * inv(X))')
cond = d_f_IW_sqrtm_two(X_spd) ≈ ForwardDiff.gradient(log_f_IW_sqrtm_two, X_spd)
println("Jacobian of the Inverse Hessian in the sqrtm basis TWO: ", cond) #True

d_f_IW_sqrtm(X) = -(n+p) * inv(X)' + 1/2 * ((inv(X) * V_spd * inv(X) * inv(X))' + (inv(X) * inv(X) * V_spd * inv(X))')
cond = d_f_IW_sqrtm(X_spd) ≈ ForwardDiff.gradient(log_f_IW_sqrtm, X_spd)
println("Jacobian of the Inverse Hessian in the sqrtm basis: ", cond)


####second derivative


### tests
f1(X) = X * V_spd * X * X

function d_f1(X)
	d = size(B)[1]
	K = zeros((d^2, d^2))
	for i in 1:d
		for j in 1:d
			for k in 1:d
				for l in 1:d
					one = (V_spd * X^2)[j,l]
					two = (X * V_spd)[k,i] * (X)[j,l]
					three = (X * V_spd * X)[k,i]
					K[(i-1) .* d .+ j, (k-1) .* d .+ l] = one + two + three
				end
			end
		end
	end
	return(K)
end

cond = d_f1(X_spd) ≈ ForwardDiff.jacobian(f1, X_spd)
println("test1: ", cond)
"""

############## verification of the derivatives in the logm basis

log_f_W_logm(X) = (n-p)/2 * logdet(exp(X)) - 1/2 * tr(inv(V_spd) * expm(X))

###

d_log_f_W_logm(X) = (n-p)/2 * Matrix{Int}(I, p, p) - 1/2 * inv(V_spd) * exp(X)'
cond = d_log_f_W_logm(X_spd) ≈ ForwardDiff.gradient(log_f_W_logm, X_spd)
println("Gradient for the Wishart in the logm basis: ", cond) #False





