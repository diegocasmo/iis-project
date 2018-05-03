%% init, set the filename, and plot pts if desirable.
filename = 'bs002_E_SADNESS_0';
plot_pts = true;
bin = read_bntfile(sprintf('%s.bnt',filename));
pts = read_lm3file(sprintf('%s.lm3', filename))';

y=[];
%% take all the points which have some real value.
for i=1:size(bin,1)
    if bin(i,1) ~= -1e+09
        y(end+1,:) = bin(i,:); %#ok
    end
end

%% plot
dotsize = 2;
scatter3(y(:,1), y(:,2), y(:,3), dotsize, y(:,3), 'filled');

%% plot each 3d point
if plot_pts
    hold on;
    dotsize = 6;
    for i=1:size(pts,1)
        scatter3(pts(i,1),pts(i,2), pts(i,3), 'filled', ...
            'MarkerEdgeColor', 'k', ...
            'MarkerFaceColor', 'black');
    end
    hold off;
end