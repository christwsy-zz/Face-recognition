function [imgs, w, h] = load_images(filelist, downscale_factor)
	%%% imgs = [img1; img2; ... imgn]
	
	% load mask
% 	mask = imread('mask.pgm');

	num_of_file = length(filelist);
	firstFile = sprintf('%s.pgm', filelist{1});
	[w, h] = size(imresize(imread(firstFile), downscale_factor));
	imgs = [];
	for i = 1 : num_of_file
		if (~ischar(filelist{i}))
			break;
		end
		[~, img_class] = regexp(filelist{i}, '.*/s(.*)/.*', 'match', 'tokens');
		
		pgmFile = sprintf('%s.pgm', filelist{i});
		img = imread(pgmFile); % load image		
		
		% applying mask to the images
% 		temp = find(~mask);
% 		img(temp) = 0;
		
		img = imresize(img, downscale_factor); % scale down to save memory
		
		temp = str2num(cell2mat(img_class{1})); % fix a very, very, odd bug
		temp = [temp, reshape(img, 1, w*h)];
		imgs = [imgs; temp]; % store into one matrix
	end
end
