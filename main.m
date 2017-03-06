%Jason Zheng & Gourav Khadge
%EE210A Speech Recognition

[audio_signals, word_labels] = load_audio('audio');
apple = Word('apple');
banana = Word('banana');

apple_signals = {};

%plot all the feature vectors for apple signals (feature vector length 2)
for i = 1:15
    apple_signals(i) = {extract_features(audio_signals{i})};
    banana_signals(i) = {extract_features(audio_signals{i+15})};
%     scatter(apple_signals{i}(1,:), apple_signals{i}(2,:));
%     hold on;
end

% axis([0 4000 0 4000]);

apple.initialize(apple_signals{1}); % use one observation set to iniialize our HMM
apple.trainAll(apple_signals, 15); % train our HMM using the baum welch with 15 iterations

banana.initialize(banana_signals{1});
banana.trainAll(banana_signals, 15);

word_arr = [apple banana];
predict_word(apple_signals{7}, word_arr)