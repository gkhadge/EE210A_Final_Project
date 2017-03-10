%Jason Zheng & Gourav Khadge
%EE210A Speech Recognition

bw_iters = 30; %adjustable parameter

[audio_signals, word_labels] = load_audio('audio');
apple = Word('apple');
banana = Word('banana');
kiwi = Word('kiwi');
lime = Word('lime');
orange = Word('orange');
peach = Word('peach');
pineapple = Word('pineapple');


apple_signals = {};

%plot all the feature vectors for apple signals (feature vector length 2)
for i = 1:15
    apple_signals(i) = {extract_features(audio_signals{i})};
    banana_signals(i) = {extract_features(audio_signals{i+15})};
    kiwi_signals(i) = {extract_features(audio_signals{i+30})};
    lime_signals(i) = {extract_features(audio_signals{i+45})};
    orange_signals(i) = {extract_features(audio_signals{i+60})};
    peach_signals(i) = {extract_features(audio_signals{i+75})};
    pineapple_signals(i) = {extract_features(audio_signals{i+90})};
%     scatter(apple_signals{i}(1,:), apple_signals{i}(2,:));
%     hold on;
end

% axis([0 4000 0 4000]);



apple.initialize(apple_signals{1}); % use one observation set to iniialize our HMM
apple.trainAll(apple_signals, bw_iters); % train our HMM using the baum welch with 15 iterations

banana.initialize(banana_signals{1});
banana.trainAll(banana_signals, bw_iters);

kiwi.initialize(kiwi_signals{1});
kiwi.trainAll(kiwi_signals, bw_iters);

lime.initialize(lime_signals{1});
lime.trainAll(lime_signals, bw_iters);

orange.initialize(orange_signals{1});
orange.trainAll(orange_signals, bw_iters);

peach.initialize(peach_signals{1});
peach.trainAll(peach_signals, bw_iters);

pineapple.initialize(pineapple_signals{1});
pineapple.trainAll(pineapple_signals, bw_iters);

test_signals = [apple_signals banana_signals kiwi_signals lime_signals orange_signals peach_signals pineapple_signals];

word_arr = [apple banana kiwi lime orange peach pineapple];

%predict_word(apple_signals{7}, word_arr)

[error_rate, predicted_labels] = cross_validate(test_signals, word_labels, word_arr);




