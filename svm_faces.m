function accu = svm_faces(train_imgs, test_imgs, num_subjects, w, h)
	% training, prepare useful data
	train_imgs_no_class = train_imgs(:, 2:end);
	test_imgs_no_feat = test_imgs(:, 2:end);
	train_imgs_no_class = double(train_imgs_no_class);
	test_imgs_no_feat = double(test_imgs_no_feat);
	
	% number of image for each class in training and testing sets
	num_per_class_train = size(train_imgs_no_class, 1) / num_subjects;
	num_per_class_test = size(test_imgs, 1) / num_subjects;
	
	% pca
	[eigFaces, topN] = eigenFaces(train_imgs_no_class, 0.1);
	train_proj = train_imgs_no_class * eigFaces;
	test_proj = test_imgs_no_feat * eigFaces;
	models = {};
	for i = 1:num_subjects
		c = zeros(size(train_imgs, 1), 1) - 1;
		
		start_idx = (i - 1) * num_per_class_train + 1;
		end_idx = start_idx + num_per_class_train - 1;
		for j = start_idx : end_idx
			c(j, 1) = 1;
		end
		
		% generate svm models
		models{i} = svmtrain(train_proj, c, 'boxconstraint', 1, 'kernel_function', 'rbf'); 
	end
	% testing
	wrong = 0;
	correct = 0;
	for i = 1:size(test_proj, 1)
		guessed = zeros(num_subjects, 1);
		for j = 1:num_subjects
			guessed(j,1) = svmclassify(models{j}, test_proj(i,:));
		end
% 		disp(guessed);
		idx = find(guessed == 1);
		if (idx == ceil(i / num_per_class_test))
% 			subplot(1,2,1);
% 			imshow(reshape(test_imgs_no_feat(i,:), w, h));
			correct = correct + 1;
		else
			wrong = wrong + 1;
% 			fprintf('-%i,+%i\n', idx, ceil(i / num_per_class_test));
		end
	end
	accu = correct / (correct + wrong);
end

function sim = gaussianK(x1, x2, sigma)
    tmp = x1 - x2; % difference
    n = tmp' * tmp; % norm
    sim = exp(-n/(2 * sigma ^ 2));
end

function sim = hyperbolicK(x1, x2, sigmoid)
	sim = tanh(sigmoid * x1' * x2);
end

function sim = linearK(x1, x2)
	sim = x1' * x2;
end