%return the index of the most probable word from a word array

function [max_ind, confidence] = predict_word(observations, word_arr)
    L = length(word_arr);
    likelihood_arr = zeros(L,1);
    for i = 1:L
        likelihood_arr(i) = word_arr(i).log_likelihood(observations);
    end
    [out,indices] = sort(likelihood_arr,'descend');
%     disp(out)
    max_ind = indices(1);
    confidence = out(1)-out(2);
%     [~,index] = max(likelihood_arr);
end