% Given an array of trained word HMMs and a set of audio signals and the
% corresponding labels, return the error rate and a vector which contains
% the predicted labels

function [error_rate, predicted_labels] = cross_validate(test_set, labels, word_arr)
    % labels are going to be in the form of a cell array
    % test_set is also going to be in the form of a cell array
    
    features_extracted = true; % paramter that determines if test_set already has features extracted
    
    L = length(test_set);
    
    num_errors = 0;
    
    for i = 1:L
        %extract features (possibly)
        if (features_extracted)
            features = test_set{i};
        else
            features = extract_features(test_set{i});
        end
        
        %predict word
        predicted_idx = predict_word(features, word_arr);
        
        %update predicted_labels
        predicted_labels{i} = word_arr(predicted_idx).name;
        
        %update errors

      
        if (~strcmp(predicted_labels{i}, labels{i}))
            num_errors = num_errors+1;
        end

    end

    error_rate = num_errors/L;
end