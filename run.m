function run
	% initialize parameters
	trials = 10;
	train_percent = 0.8; % percent of training in all images
	num_subjects = 29; % also the number of classes
	downscale_factor = 0.4;
	k = 1; % for K-nn
	
	% initialize variables
	num_train = 10 * train_percent;
	num_test = 10 - num_train;
	train_file_list = cell(num_subjects * num_train, 1);
	test_file_list = cell(num_subjects * num_test, 1);
	
	
	for k = 1:2:3
		accu_knn = zeros(1, trials);
		accu_svm = zeros(1, trials);
		
		for n = 1:trials
			% use same data for both classfiers
			% load training & testing samples
			arr = 1:10;
			arr = randperm(10, 10); % randomize images order
			
			for i = 1:num_subjects
				% 		subplot(3,3,i);
				% 		if (i == 5)
				% 			imshow('att_faces/s1/5.pgm');
				% 		else
				% 			imshow(sprintf('att_faces/s%i/1.pgm',i));
				% 		end
				for j = 1:num_train
					train_file_list{(i - 1) * num_train + j} = sprintf('att_faces/s%i/%i', i, arr(j));
				end
				for j = 1:num_test
					test_file_list{(i - 1) * num_test + j} = sprintf('att_faces/s%i/%i', i, arr(10 - j + 1));
				end
			end
			[train_imgs, ~, ~] = load_images(train_file_list, downscale_factor);
			[test_imgs, w, h] = load_images(test_file_list, downscale_factor);
			
			
			accu_knn(n) = knn_faces(train_imgs, test_imgs, num_subjects, w, h, k);
			% 		accu_svm(n) = svm_faces(train_imgs, test_imgs, num_subjects, w, h);
		end
		
% 		fprintf('trials = %i, k = %i\n', trials, k);
% 		fprintf('PCA + KNN %f\n', mean(accu_knn));
		fprintf('%f\n', mean(accu_knn));
	end
	% 	fprintf('PCA + SVM Accu: %f\n', mean(accu_svm));
end
