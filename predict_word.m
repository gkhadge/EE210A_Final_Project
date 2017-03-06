%return the index of the most probable word from a word array

function index = predict_word(observations, word_arr)
    L = length(word_arr);
    likelihood_arr = zeros(L,1);
    for i = 1:L
        likelihood_arr(i) = word_arr(i).log_likelihood(observations);
    end
    [~,index] = max(likelihood_arr);
end