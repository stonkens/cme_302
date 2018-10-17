
function getrfRook!(A)
    """getrfRook! performs LU factorization with rook pivoting on rank r<n matrix A (size n).
    Pivots through rows and columns until it finds new pivot in Schur complement (highest abs in its row AND column).
    Rows and column are calculated by applying all previous (k-1) factorization operations to the row/column in present
    Pivot values are stored in a seperate array and not written to Schur complement of A. Once pivot found, permute row
    and column to first row and column of Schur complement. Apply all previous (k-1 total) factorization operations to
    this row and column and overwrite first row and first column of Schur complement. 
    Iterate until pivot value is 0. 
    Return permutation arrays P_row and P_column + modifies input A
    """   
    n = size(A,1)
    P_row = collect(1:n); P_col = collect(1:n); #keeps track of swaps between rows and columns whilst pivoting 
    for k=1:n
        row_A = zeros(n-k) #initialize row_A as it is called outside while loop 
        col_A = zeros(1, n-k) #initialize row_A as it is called outside while loop 
        
        # Rook pivoting
        row = 1; row0 = 0; col = 1; col0 = 0
        while row != row0 || col != col0
            row0, col0 = row, col # Save old values
            if k==1
                row_A = A[row+k-1:row+k-1, k:end]  #update row_A to find the actual Schur Complement
            else
                #update row_A to find the actual Schur Complement
                row_A = A[row+k-1:row+k-1, k:end] - A[row+k-1:row+k-1, 1:k-1] * A[1:k-1, k:end] 
            end
            row_A_max = abs.(row_A[1,:]) # Find max of row A
            col = argmax(row_A_max)
            if k==1
                col_A = A[k:end, col+k-1:col+k-1]  #update col_A to find the actual Schur Complement
            else
                #update col_A to find the actual Schur Complement
                col_A = A[k:end, (col+k-1):(col+k-1)] - A[k:end, 1:k-1] * A[1:k-1, (col+k-1):(col+k-1)]
            end
            col_A_max = abs.(col_A[:,1]) 
            row = argmax(col_A_max)
        end

        # If we reach this line, this means that the pivot is the largest
        # in its row and column.
        row += k-1; col += k-1 #becomes row of the whole matrix, not only Schur Complement
        # Swap rows and columns (placing pivot at (1,1) in Schur Complement)
        P_row[k], P_row[row] = P_row[row], P_row[k]
        P_col[k], P_col[col] = P_col[col], P_col[k]
        for j=1:n
            A[k,j],A[row,j] = A[row,j],A[k,j]
        end
        for i=1:n
            A[i,k],A[i,col] = A[i,col],A[i,k]
        end

        # Perform all previous factorization operations to pivot row and column
        A[k:end,k] = A[k:end, k:k] - A[k:end, 1:k-1] * A[1:k-1, k:k]
        A[k,k+1:end] = A[k:k, k+1:end] - A[k:k, 1:k-1] * A[1:k-1, k+1:end]
        
        #All steps that should have been done have now been executed for the current pivot

        #If pivot is 0 (or close to) we have reached rank of matrix and stop operations
        if abs(A[k,k]) < eps()*1000      
            return P_row, P_col
        end
        if A[k,k] != 0
            for i=k+1:n
                A[i,k] /= A[k,k]
            end
        end
    end
    return P_row, P_col
end