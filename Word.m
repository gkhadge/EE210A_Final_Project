classdef Word < handle %inherit from handle so all copies reference this one class
    properties
       %add more properties as needed
       N     =   2; % number of states
       A     =  []; % NxN transition probability matrix (fill out with baum welch)
       prior =  []; % Nx1 initial state distribution vector
       mu    =  []; % DxN mean vector (D = number of features) 
       Sigma =  []; % DxDxN covariance matrix
       name  =  ''; % Word label
    end
    
    methods
        function self = Word(name)
            self.name = char(name);
        end
        
        % for testing purposes, will set prior with baum welch training 
        function setPrior(self,prior)
            self.prior = prior; 
        end
        
        % for testing purposes, will set TPM with baum welch training
        function setA(self,A)
            self.A = A;
        end
        
        % need f(o|state) at each observation step for each state,
        % this will be represented in matrix D (size NxL, L length of
        % observations)
        function alpha = forward(self, observations, D) 
            L = size(observations,2); % num of observations
            alpha = zeros(self.N,L);
            
            for i = 1:L
                if i == 1                               %initialize 
                    alpha(:,i) = D(:,i).*self.prior;
                else                                    %recursion
                    alpha(:,i) = D(:,i).*(self.A'*alpha(:,i-1));
                end
            end
        end
        
        function beta = backward(self, observations, D)
            L = size(observations,2); 
            beta = zeros(self.N,L);
            
            for i = L:-1:1
                if i == L                               %initialization
                    beta(:,i) = ones(size(beta(:,i)));
                else
                    beta(:,i) = self.A*(D(:,i+1).*beta(:,i+1));
                end
            end 
        end
        
        
        
        
        % perform baum welch training here. Will set prior, and estimate A,
        % mu, and Sigma
        function train(self, observations)
        end
        
        %add functions as needed
      
        
    end
end