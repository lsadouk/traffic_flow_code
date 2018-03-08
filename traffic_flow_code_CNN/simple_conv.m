% SIMPLE CONVOLUTION FUNCTION conv(x,h) 
% Lucas Emanuel Batista dos Santos 
% - 
% Receive two vectors and show on screen a new vector resultant of 
% convolution operation 
function simple_conv(f, g)

% Transform the vectors f and g in new vectors with the same length 
F = [f,zeros(1,length(g))]; 
G = [g,zeros(1,length(f))];

% FOR Loop to put the result of convolution between F and G vectors 
% in a new vector C. According to the convolution operation characteristics, 
% the length of a resultant vector of convolution operation between two vector 
% is the sum of vectors length minus 1 
for i=1:length(g)+length(f)-1 
% Create a new vector C 
C(i) = 0; 
% FOR Loop to walk through the vector F ang G 
for j=1:length(f) 
if(i-j+1>0) 
C(i) = C(i) + F(j) * G(i-j+1); 
else 
end 
end 
end

% Show C vector on screen 
C

end