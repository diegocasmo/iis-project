%% init
filename = 'bs002_E_SADNESS_0';
img = imread(sprintf('%s.png', filename));
pts = read_lm2file(sprintf('%s.lm2', filename))';

%% plot each point on the image
imshow(img);
hold on;
for i=1:size(pts,1)
    plot(pts(i,1),pts(i,2), 'o', 'MarkerEdgeColor', 'white');
end
hold off;