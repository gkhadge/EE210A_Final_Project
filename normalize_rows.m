% normalize the rows of a matrix so they sum to 1
% input is an MxN matrix A

function normalized = normalize_rows(A)
    M = size(A,1);
    for i = 1:M
        A(i,:) = A(i,:)/sum(A(i,:));
    end
    normalized = A;
end