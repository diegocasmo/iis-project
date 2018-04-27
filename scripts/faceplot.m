%% init
filename = 'bs002_E_SADNESS_0.bnt';
x = read_bntfile(filename);
y=[];

%% take all the points which have some real value.
for i=1:size(x,1)
    if x(i,1) ~= -1e+09
        y(end+1,:) = x(i,:); %#ok
    end
end

%% plot
dotsize = 6;
scatter3(y(:,1), y(:,2), y(:,3), dotsize, y(:,3), 'filled');