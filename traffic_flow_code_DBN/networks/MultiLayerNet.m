classdef MultiLayerNet < handle & AbstractNet
    % MULTILAYERNET Implements stacking of interconnected neural networks
    % as an AbstractNet
    
    % author  : Nicolas Granger <nicolas.granger@telecom-sudparis.eu>
    % licence : MIT
    
    properties
        nets = {};
        frozenBelow = 0;
    end
    
    methods
        
        % AbstractNet implementation ---------------------------------------- %
        
        function S = insize(self)
            S = self.nets{1}.insize();
        end
        
        function S = outsize(self)
            S = self.nets{end}.outsize();
        end
        
        function [Y, A] = compute(self, X)
            nbNets = length(self.nets);
            
            if nargout > 1
                computeA = true;
                A = cell(nbNets, 1);
            else
                computeA = false;
            end
            
            for o = 1:nbNets
                if computeA
                    [X, A{o}] = self.nets{o}.compute(X);
                else
                    X = self.nets{o}.compute(X);
                end
            end
            Y = X;
        end
        
        function [] = pretrain(self, X)
            for o = 1:length(self.nets)
                if o > self.frozenBelow
                    self.nets{o}.pretrain(X);    
                end
                X = self.nets{o}.compute(X);
            end
        end
        
        function [G, inErr] = backprop(self, A, outErr)
            G     = cell(length(self.nets), 1);
            inErr = [];
            % Backprop and compute gradient
            for l = length(self.nets):-1:self.frozenBelow + 2
                [G{l}, outErr] = self.nets{l}.backprop(A{l}, outErr);
            end
            bottom = self.frozenBelow + 1;
            G{bottom} = self.nets{bottom}.backprop(A{bottom}, outErr);
            
            % Silently ignore gradient backpropagation eventhough G is 
            % requested
            
%             if nargout == 2 % Backprop through frozen layers anyways
%                 for l = self.frozenBelow + 1:-1:1
%                     [~, outErr] = self.nets{l}.backprop(A{l}, outErr);
%                 end
%                 inErr = outErr;
%             end
        end
        
        function [] = gradientupdate(self, G)
            for l = length(self.nets):-1:self.frozenBelow + 1
                self.nets{l}.gradientupdate(G{l});
            end
        end
        
        % Methods ----------------------------------------------------------- %
        
        function [] = add(self, net)
            % ADD Stack an additional network on top
            %   [] = ADD(self, net) add net (an implementation of
            %   AbstractNet) on top of the networks currently in self.
            %   Input size of net must match the current output size of the
            %   multilayer network.
            assert(isa(net, 'AbstractNet'), 'net must implement AbstractNet');
%             TODO: check size compatibility (even for groups of neurons)
%             assert(isempty(self.nets) || ...
%                 all(self.outsize() == net.outsize), ...
%                 'self and net should have equal sizes');
            
            nbNets                    = length(self.nets) + 1;
            self.nets{nbNets}         = net.copy();
        end % add(self, net)
        
        function [] = freezeBelow(self, varargin)
            % FREEZEBELOW(l) Freeze bottom layers weights
            %   FREEZEBELOW(l) disables pretraining and gradient update on
            %   the l bottom layers of the network
            %   
            %   FREEZEBELOW() unfreezes all layers.
            if ~isempty(varargin)
                self.frozenBelow = varargin{1};
            else
                self.frozenBelow = 0;
            end
        end
        
    end % methods
    
    methods(Access = protected)
        
        % Copyable implementation ------------------------------------------- %
        
        % Override copyElement method
        function copy = copyElement(self)
            copy = MultiLayerNet();
            copy.frozenBelow = self.frozenBelow;
            % Make a deep copy of self.nets
            copy.nets = cell(size(self.nets));
            for i = 1:numel(self.nets)
                copy.nets{i} = self.nets{i}.copy();
            end
        end
        
    end
    
end % MultiLayerNet