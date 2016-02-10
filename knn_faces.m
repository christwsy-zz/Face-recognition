function accu = knn_faces(train_imgs, test_imgs, num_subjects, w, h, k)
	% training prepare useful data
	train_class = train_imgs(:,1);
	test_class = test_imgs(:,1);
	train_imgs_no_class = train_imgs(:, 2:end);
	test_imgs_no_feat = test_imgs(:, 2:end);
	train_imgs_no_class = double(train_imgs_no_class);
	test_imgs_no_feat = double(test_imgs_no_feat);
	
	% number of image for each class in training and testing sets
	num_per_class_train = size(train_imgs_no_class, 1) / num_subjects;
	
	% pca
	[eigFaces, ~] = eigenFaces(train_imgs_no_class, 0.2);
	train_proj = train_imgs_no_class * eigFaces;
	test_proj = test_imgs_no_feat * eigFaces;
	
	% k-nn
	idx = knnsearch(train_proj, test_proj, 'k', k, 'Distance', 'cityblock');
	
	% testing, using a voting schema
	wrong = 0;
	correct = 0;
	for i = 1:length(idx)
		% find the most frequent one in the array
		
		guessed_class = mode(ceil(idx(i,:) / num_per_class_train));
		if (guessed_class == test_class(i, 1))
			correct = correct + 1;
		else
			wrong = wrong + 1;
		end
	end
	
	accu = correct / (correct + wrong);
end
