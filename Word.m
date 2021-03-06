%% Word.m
% Class stores HMM model for each word
% Includes functions to initialize, train, validate test data

classdef Word < handle %inherit from handle so all copies reference this one class
    properties
       %add more properties as needed
       A     =  []; % NxN transition probability matrix (fill out with baum welch)
       prior =  []; % Nx1 initial state distribution vector
       mu    =  []; % DxN mean vector (D = number of features) 
       Sigma =  []; % DxDxN covariance matrix
       name  =  ''; % Word label
       
       % Configuration settings
       N     =   15; % number of states (adjustable variable)
       startAtFirstState = true; % Always start at first state 
       useLinearTopology = true;  % Use linear topology, otherwise ergodic
       topology = 0;
    end
    
    methods
        function self = Word(name)
            self.name = char(name);
        end
        
        % Configuration Settings
        function setStartAtFirstState(self,bool)
            self.startAtFirstState = bool; 
        end
        
        function setMarkovLinearTopology(self,bool)
            self.useLinearTopology = bool;
        end
                
        function setLinearModel(self,bool)
            self.useLinearTopology = bool;
            self.startAtFirstState = bool;
        end
        
        % for testing purposes, will set prior with baum welch training 
        function setPrior(self,prior)
            self.prior = prior; 
        end        
        
        % for testing purposes, will set TPM with baum welch training
        function setA(self,A)
            self.A = A;
        end
        
        function setN(self,N)
            self.N = N;
        end
        
        function log_likelihood = log_likelihood(self, observations)
            D = self.state_likelihood(observations);
            [~, c] = self.forward(observations, D);
            if any(c==0)
                log_likelihood = -Inf;
            else
                log_likelihood = sum(log10(c));
            end
        end
        
        % observations refers to just 1 set of observations
        % each word will have multiple sets of observations, 1 per audio
        % file
        
        % need f(o|state) at each observation step for each state,
        % this will be represented in matrix D (size NxL, L length of
        % observations)
        function [alpha, c] = forward(self, observations, D, normalize)
            if nargin < 4
                normalize = 1;
            end
			
            L = size(observations,2); % num of observations
            alpha = zeros(self.N,L);
            c = ones(L,1);
            
            for i = 1:L
                if i == 1                               %initialize 
                    alpha(:,i) = D(:,i).*self.prior;
                else                                    %recursion
                    alpha(:,i) = D(:,i).*(self.A'*alpha(:,i-1));
                end
                
                if (normalize)
					c(i) = sum(alpha(:,i));
                    alpha(:,i) = alpha(:,i)/(c(i)+(c(i)==0)); % handle c(i)==0
                    % if c(i) ==0, alpha(:,i) ==0
                end
            end
        end
        
        %calculate backwards recursion (no normalization)
        function beta = backward(self, observations, D)
            L = size(observations,2); 
            beta = zeros(self.N,L);
            
            for i = L:-1:1
                if i == L                               %initialization
                    beta(:,i) = ones(size(beta(:,i)));
                else                                    %recursion
                    beta(:,i) = self.A*(D(:,i+1).*beta(:,i+1));
                end
            end 
        end
        
        %do both forward and backwards recursion with normalization
        function [alpha, beta, c] = forward_backward(self, observations, D)
            [alpha, c] = self.forward(observations, D);
            
            %now do the backwards part with normalization
            L = size(observations,2); 
            beta = zeros(self.N,L);
            
            for i = L:-1:1
                if i == L                               %initialization
                    beta(:,i) = ones(size(beta(:,i)));
                else                                    %recursion
                    beta(:,i) = (self.A*(D(:,i+1).*beta(:,i+1)))/(c(i+1)+(c(i+1)==0));
                    if c(i+1)==0
                        % Handle c(i+1) == 0
                        beta(:,i) = 0;
                    end
                end
            end 
        end
		
		% create the D matrix based on observations and current state of 
		% mu and Sigma (assuming gaussian mixture model)
		function D = state_likelihood(self, observations)
            D = zeros(self.N, size(observations,2));
            
            for i = 1:self.N
                D(i, :) = mvnpdf(observations', self.mu(:,i)', self.Sigma(:,:,i));
            end
		end

        % perform baum welch training here. Will set prior, and estimate A,
        % mu, and Sigma
		% implement assuming 1 set of observations first, extend to multiple sets later
        % r NxL (num states x length of observation)
        % S NxNxL-1 (L-1 NxN matrices)
        % Nr Nx1
        function [r,S,Nr] = e_step(self, observations)
            D = self.state_likelihood(observations);
            [alpha, beta, c] = self.forward_backward(observations, D);
            L = size(observations,2);
            
            r = alpha.*beta;            %size NxL (L length of observation vector, N number of states)
            S = zeros(self.N, self.N, L-1);
            for i = 2:L
                S(:,:,i-1) = ((self.A).*(alpha(:,i-1)*(beta(:,i).*D(:,i))'))/c(i);
            end
            Nr = sum(r,2);
            
        end
		
		% observation set will probably have to be a cell matrix since each set of 
		% observations can be a different length
		function [mu_convergence,Sigma_convergence,A_convergence] = trainAll(self, observation_set, num_iter)
            L = size(observation_set,2);
            
            mu_convergence = zeros(num_iter,1);
            Sigma_convergence = zeros(num_iter,1);
            A_convergence = zeros(num_iter,1);

            for i = 1:num_iter
                expected_mu = zeros(size(self.mu));     %size DxN (D size of feature vector)
                expected_Sigma = zeros(size(self.Sigma));
                expected_prior_num = zeros(size(self.prior));
                expected_prior_den = 0;
                expected_A_num = zeros(size(self.A));
                %expected_A_den = zeros(size(self.A));
                % since we know A is a stochastic matrix, we know the rows
                % must sum up to 1. Since we know the numerator, we can
                % just normalize the rows instead
                expected_N = zeros(self.N,1);
                
                for l = 1:L
                    Y = observation_set{l};
                    Nl = size(Y,2);
                    num_features = size(Y,1);
                  	[r,S,Nr] = self.e_step(Y);
                    for j = 1:self.N
                        mu_k = repmat(self.mu(:,j),1,Nl); %[mu_k mu_k ... mu_k]
                        r_k = repmat(r(j,:),num_features,1); 
                        expected_mu(:,j) = expected_mu(:,j) + sum(r_k.*Y,2);
                        expected_Sigma(:,:,j) = expected_Sigma(:,:,j) + (r_k.*(Y-mu_k))*(Y-mu_k)';
                    end
                    expected_prior_num = expected_prior_num + r(:,1);
                    expected_prior_den = expected_prior_den + sum(r(:,1));
                    expected_A_num = expected_A_num + sum(S,3);
                    %expected_A_den = expected_A_den + repmat(sum(sum(S,3)),self.N,1);
                    expected_N = expected_N + Nr;    
                end
                
                % Set any zeros to one before dividing to avoid NaN
                expected_N = expected_N + (expected_N == 0);
                
                for j = 1:self.N
                    expected_mu(:,j) = expected_mu(:,j)/expected_N(j);
                    
                    mu_convergence(i) = mu_convergence(i) + norm(self.mu(:,j)-expected_mu(:,j),'fro')^2;
                    self.mu(:,j) = expected_mu(:,j);
                    
                    expected_Sigma(:,:,j) = expected_Sigma(:,:,j)/expected_N(j);
                    % Ninja trick to ensure positive semidefiniteness
                    expected_Sigma(:,:,j) = expected_Sigma(:,:,j) + 0.01*eye(num_features);
                    
                    Sigma_convergence(i) = Sigma_convergence(i) + norm(self.Sigma(:,:,j)-expected_Sigma(:,:,j),'fro')^2;                    
                    self.Sigma(:,:,j) = expected_Sigma(:,:,j);
                    
                    % Check if Sigma is Positive Definite
                    [~,p] = chol(self.Sigma(:,:,j));
                    if p~=0
                        j
                        self.Sigma(:,:,j)
                        % Has caused errors in the past, not sure why
                    end
                end
%               
                % Force Baum-Welch to always start at first state
                if self.startAtFirstState
                    self.prior = zeros(size(self.prior));
                    self.prior(1) = 1;
                else
                    self.prior = expected_prior_num/expected_prior_den;
                end
                expected_A_num = normalize_rows(expected_A_num);
                
                A_convergence(i) = norm(self.A - expected_A_num,'fro')^2;
                self.A = expected_A_num;
            end
        end
        
		% observation set will probably have to be a cell matrix since each set of 
		% observations can be a different length
		function num_iters = trainAll2convergence(self, observation_set)
            L = size(observation_set,2);
            
            mu_convergence = 100;
            Sigma_convergence = 100;
            A_convergence = 100;
            
            num_iters = 0;

            while (mu_convergence+Sigma_convergence+A_convergence > 1e-10 || num_iters < 50) && num_iters < 5e3
                num_iters = num_iters + 1;
                
                mu_convergence = 0;
                Sigma_convergence = 0;
                
                expected_mu = zeros(size(self.mu));     %size DxN (D size of feature vector)
                expected_Sigma = zeros(size(self.Sigma));
                expected_prior_num = zeros(size(self.prior));
                expected_prior_den = 0;
                expected_A_num = zeros(size(self.A));
                %expected_A_den = zeros(size(self.A));
                % since we know A is a stochastic matrix, we know the rows
                % must sum up to 1. Since we know the numerator, we can
                % just normalize the rows instead
                expected_N = zeros(self.N,1);
                
                for l = 1:L
                    Y = observation_set{l};
                    Nl = size(Y,2);
                    num_features = size(Y,1);
                  	[r,S,Nr] = self.e_step(Y);
                    for j = 1:self.N
                        mu_k = repmat(self.mu(:,j),1,Nl); %[mu_k mu_k ... mu_k]
                        r_k = repmat(r(j,:),num_features,1); 
                        expected_mu(:,j) = expected_mu(:,j) + sum(r_k.*Y,2);
                        expected_Sigma(:,:,j) = expected_Sigma(:,:,j) + (r_k.*(Y-mu_k))*(Y-mu_k)';
                    end
                    expected_prior_num = expected_prior_num + r(:,1);
                    expected_prior_den = expected_prior_den + sum(r(:,1));
                    expected_A_num = expected_A_num + sum(S,3);
                    %expected_A_den = expected_A_den + repmat(sum(sum(S,3)),self.N,1);
                    expected_N = expected_N + Nr;    
                end
                
                % Set any zeros to one before dividing to avoid NaN
                expected_N = expected_N + (expected_N == 0);
                
                for j = 1:self.N
                    expected_mu(:,j) = expected_mu(:,j)/expected_N(j);
                    
                    mu_convergence = mu_convergence + norm(self.mu(:,j)-expected_mu(:,j),'fro')^2;
                    self.mu(:,j) = expected_mu(:,j);
                    
                    expected_Sigma(:,:,j) = expected_Sigma(:,:,j)/expected_N(j);
                    % Ninja trick to ensure positive semidefiniteness
                    expected_Sigma(:,:,j) = expected_Sigma(:,:,j) + 0.01*eye(num_features);
                    
                    Sigma_convergence = Sigma_convergence + norm(self.Sigma(:,:,j)-expected_Sigma(:,:,j),'fro')^2;                    
                    self.Sigma(:,:,j) = expected_Sigma(:,:,j);
                    
                    % Check if Sigma is Positive Definite
                    [~,p] = chol(self.Sigma(:,:,j));
                    if p~=0
                        j
                        self.Sigma(:,:,j)
                        % Has caused errors in the past, not sure why
                    end
                end
%               
                % Force Baum-Welch to always start at first state
                if self.startAtFirstState
                    self.prior = zeros(size(self.prior));
                    self.prior(1) = 1;
                else
                    self.prior = expected_prior_num/expected_prior_den;
                end
                expected_A_num = normalize_rows(expected_A_num);
                % Makes sure no state is ever accidentally removed
                expected_A_num = expected_A_num + self.topology*1e-10;
                expected_A_num = normalize_rows(expected_A_num);
                
                A_convergence = norm(self.A - expected_A_num,'fro')^2;
                self.A = expected_A_num;
            end
        end
        
        function initialize(self, observations)
            % Initialize prior
            if self.startAtFirstState
                self.prior = zeros(self.N,1);
                self.prior(1) = 1;
            else
                self.prior = rand(self.N, 1);
                self.prior = self.prior/(sum(self.prior));
            end
            % Can also try uniform priors
            % self.prior = ones(self.N, 1)/self.N;
        
            if self.useLinearTopology
                % Initialize Linear Markov Chain Topology
                self.A = zeros(self.N);
                for q = 1:(self.N-1)
                    self.A(q,q) = 0.5;%rand(1);
                    self.A(q,q+1) = 0.5; %rand(1);
                end
                self.A(self.N,self.N) = 1;
                self.topology = self.A > 0;
            else
                % Initialize Ergodic Markov Chain Topology
                self.A = normalize_rows(rand(self.N));   
                self.topology = ones(size(self.A));
            end
            
            % Use one set of observations to form a diagonal covariance
            self.Sigma = repmat(diag(diag(cov(observations'))), [1 1 self.N]);
            
            % Pick a random data points to be the mean
            indices = randperm(size(observations, 2));
            self.mu = observations(:, indices(1:self.N));
        end
        %add functions as needed
      
        % Viterbi Sequency Decoder
        % Works on one observation at a time
        function k_o = viterbi_decoder(self, observation)
            num_obs = size(observation,2);
            log_p = zeros(self.N,num_obs);
            psi = zeros(self.N,num_obs);
            
            D = self.state_likelihood(observation);
            
            log_p(:,1) = log(self.prior.*D(:,1));
            
            % Forward Pass
            for n = 2:num_obs
               for k = 1:self.N 
                   arg = log(self.A(:,k)) + log_p(:,n-1);
                   psi(k,n) = max(arg);
                   log_p(:,n) = log(self.A(psi(:,n),k)) + log_p(psi(:,n),n-1) + log(D(:,n));
               end
            end
            
            % Backward Pass
            k_o = zeros(num_obs,1);
            k_o(num_obs) = max(log_p(:,num_obs));
            
            for n = num_obs-1:-1:1
                k_o(n) = psi(k_o(n+1),n+1);
            end
        end
    end
end