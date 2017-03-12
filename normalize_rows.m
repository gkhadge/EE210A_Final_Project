% normalize the rows of a matrix so they sum to 1
% input is an MxN matrix A

function normalized = normalize_rows(A)
    M = size(A,1);
    for i = 1:M
		s = sum(A(i,:));
		if s == 0
			A(i,i) = 1;
		else
			A(i,:) = A(i,:)/s;
		end
    end
    normalized = A;
end