
mu = 0.8024; %for US-101-North pr=3
sigma= 0.1873;%for US-101-North pr=3
lambda = 1;  
c= 0.8; % was c=0.3
syms x;

%% display the probabilistic loss function and its derivative at c
y1 = 1 / 2 * sum( (x - c).^ 2 + lambda .* abs(x - c) .*(1 - pdf('Normal',c,mu,sigma) .* (sigma*sqrt(2*pi))) ); % .* 2-pdf
ydot = diff(y1,x,1);
%xddot = diff(x,t,2)

x_ = -1:0.01:1; %x_ = -1:0.01:1;
y1_ = subs(y1,x,x_);
ydot_ = subs(ydot,x,x_);
ydot_L2_loss = subs(x-c,x,x_);
%xddot_ = subs(xddot,t,t_);
plot(x_,y1_,'b',x_,ydot_,'r',x_,ydot_L2_loss,'g')


%% display the derivative of the probabilistic loss function and at c
%x = t*exp(-3*t)+0.25*exp(-3*t);
%y1= abs(x);
y1 = 1 / 2 * sum( (x - c).^ 2 + lambda .* abs(x - c) .*(1 - pdf('Normal',c,mu,sigma) .* (sigma*sqrt(2*pi))) ); % .* 2-pdf
ydot = diff(y1,x,1);

%xddot = diff(x,t,2)

x_ = 0:0.01:1; %x_ = -1:0.01:1;
%y1_ = subs(y1,x,x_);
ydot_ = subs(ydot,x,x_);
y2dot_ = subs(y2dot,x,x_);
ydot_L2_loss = subs(x-c,x,x_);
%xddot_ = subs(xddot,t,t_);
plot(x_,ydot_,'r',x_,ydot_L2_loss,'--b', );

%% plot the abs(x-c) vs the (2*sigmoid-0.5) 
%ps: sigmoid y is in range [0,1] -> need to convert into [-0.5,0.5], then
%to [-1,1]
figure,
hold on
y= abs(x-c);
ydot = diff(y,x,1);

yy= 2*(1/(1+exp(-20*(x-c))) -0.5); %2/(1+exp(-(x-c)))-1

x_ = -1:0.01:1; %x_ = -1:0.01:1;
ydot_ = subs(ydot,x,x_);
yy_ = subs(yy,x,x_);
plot(x_,yy_,'r',x_,ydot_,'b');

%% display the derivative of the Loss_p and at c + derivative of loss_L2 + sigmoid Loss_p 
%x = t*exp(-3*t)+0.25*exp(-3*t);
%y1= abs(x);
figure,
y1 = 1 / 2 * sum( (x - c).^ 2 + lambda .* abs(x - c) .*(1 - pdf('Normal',c,mu,sigma) .* (sigma*sqrt(2*pi))) ); % .* 2-pdf
ydot = diff(y1,x,1);

%inte_reg_sigmoid = x+2*log(exp(-x+c)+1)-c; % for w=1
inte_reg_sigmoid = x+2/20*log(exp(20*(-x+c))+1)-c; % for w=1
y2 = 1 / 2 * sum( (x - c).^ 2 + lambda .* inte_reg_sigmoid .*(1 - pdf('Normal',c,mu,sigma) .* (sigma*sqrt(2*pi))) ); % .* 2-pdf
y2dot = diff(y2,x,1);

%xddot = diff(x,t,2)

x_ = 0:0.01:1; %x_ = -1:0.01:1;
%y1_ = subs(y1,x,x_);
ydot_ = subs(ydot,x,x_);
y2dot_ = subs(y2dot,x,x_);
ydot_L2_loss = subs(x-c,x,x_);
%xddot_ = subs(xddot,t,t_);
plot(x_,ydot_,'r',x_,ydot_L2_loss,'--b', x_,y2dot_,'g');
