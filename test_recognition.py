import models.vj_script as vj
import models.keras_recognition_script as keras_recognition
import models.mmod_script as mmod
import models.onet_script as onet
import tool_scripts.crop_detected_face as crop_detected_face
from keras.preprocessing import image as preprocess_image
from keras.models import Model, load_model
import sys
import numpy as np
import os
import functools
import matplotlib.pyplot as plt
from tabulate import tabulate
import argparse
import readline, glob
from collections import defaultdict
import shutil
from PIL import Image
import time

# ["data", "ID", "item", "conditions"+]
#       item        conditions
#     "base" : [light_intense, side_face],
#     "blocking" : [full_direct, percent],
#     "distance-angle" : [distance, L_R, angle],
#     "left-right-sideface" : [L_R],
#     "lighting" : [light_intense],
#     "up-down-sideface" : [up_down],
#     "with-glasses" : [with_without]

item_list = ["base", "distance-angle", "up-down-sideface", "left-right-sideface", "blocking", "with-glasses",
			  "lighting"]

def complete(text, state):
	return (glob.glob(text + '*') + [None])[state]



'''
The assumed structure of the folder is item/IDs/images
'''
def generate_face_vectors_in_folder(item_folder, model, mirror):
	person_num = len(os.listdir(item_folder))
	a_person_folder = os.path.join(item_folder, os.listdir(item_folder)[0])
	image_num = len(os.listdir(a_person_folder))

	face_vectors = np.zeros((person_num, image_num, model.bottleneck))
	person_folders = []
	personIDs = []
	for person_index, personID in enumerate(os.listdir(item_folder)):
		person_folder = os.path.join(item_folder, personID)
		person_folders.append(person_folder)
		for file_index, file_ in enumerate(os.listdir(person_folder)):
			file_path = os.path.join(person_folder, file_)
			img = preprocess_image.load_img(file_path, target_size=model.input_size)
			x = preprocess_image.img_to_array(img)
			if model.grayscale:
				x = model.rgb2gray(x)
			x = x / 256.0 - 0.5
			x = x.reshape((1,) + x.shape)
			if mirror:
				prediction = np.concatenate((model.run(x), model.run(x[:, :, ::-1, :])), axis=1)
			else:
				prediction = model.run(x)
			prediction = prediction / np.linalg.norm(prediction)
			if file_index >= image_num:
				break
			face_vectors[person_index, file_index, :] = prediction
		personIDs.append(personID)

	return face_vectors, personIDs


def find_euclideian_distance(face_vector_a, face_vector_b):
	return np.linalg.norm(face_vector_a - face_vector_b)


'''
the return distance_over_base is a list of (minimum distance, the index of the face with the minimum distance)
'''
def find_distance_distribution_over_base(base_face_vectors, test_face_vector):
	distance_over_base = []
	for base_person_index, base_person_vectors in enumerate(base_face_vectors):
		min_dist_over_all_face = sys.float_info.max
		min_face_index = 0
		for face_index, a_face_vector in enumerate(base_person_vectors):
			dist = find_euclideian_distance(test_face_vector, a_face_vector)
			if dist < min_dist_over_all_face:
				min_dist_over_all_face = dist
				min_face_index = face_index
		distance_over_base.append((min_dist_over_all_face, min_face_index))

	return distance_over_base


# def visualize_threshold(shreshold):
#     plt.imshow(np.full((10, 10), shreshold), cmap='gray')
#     plt.show()


def visualize_distance_distribution(li, title, x_axis, img_name):
	y_axis = [row[0] for row in li]
	M = [row[1:] for row in li]
	fig, ax = plt.subplots()
	fig.canvas.set_window_title(title)
	ax.matshow(M, cmap='gray')
	plt.xticks(np.arange(len(x_axis)), x_axis, rotation='vertical')
	plt.yticks(np.arange(len(y_axis)), y_axis)
	plt.savefig(img_name)
	plt.show()



def find_closest_base(distance_over_base):
	min_person_index = 0
	min_face_index = 0
	min_dist = sys.float_info.max
	for person_index, (distance, face_index) in enumerate(distance_over_base):
		if distance < min_dist:
			min_person_index = person_index
			min_face_index = face_index
			min_dist = distance
	return min_person_index, min_face_index, min_dist


def sort_base_according_to_test(base_face_vectors, basePersonIDs, testPersonIDs):
	swap_order = []
	basePersonIDs_list_head_index = 0
	for testID in testPersonIDs:
		for base_index, baseID in enumerate(basePersonIDs):
			if testID == baseID:
				swap_order.append(base_index)
				basePersonIDs[base_index], basePersonIDs[basePersonIDs_list_head_index] = basePersonIDs[basePersonIDs_list_head_index], basePersonIDs[base_index]
				basePersonIDs_list_head_index += 1
				break

	for i in range(len(basePersonIDs)):
		if i not in swap_order:
			swap_order.append(i)

	base_face_vectors = base_face_vectors[swap_order, :]

	return base_face_vectors, basePersonIDs

def overwrite_dir(dir):
	if os.path.exists(dir):
		shutil.rmtree(dir)
	os.makedirs(dir)


def align_base_and_test(base_folder, test_folder, alignmnet_model):

	faces_to_be_aligned = []
	for person_ID in os.listdir(base_folder):
		bbox_file = os.path.join(base_folder, person_ID, 'bbox_file.txt')
		faces_to_be_aligned.append(bbox_file)
	for test_item in os.listdir(test_folder):
		for person_ID in os.listdir(os.path.join(test_folder, test_item)):
			bbox_file = os.path.join(test_folder, test_item, person_ID, 'bbox_file.txt')
			faces_to_be_aligned.append(bbox_file)

	for bbox_file_path in faces_to_be_aligned:
		bbox_file = open(bbox_file_path, 'r')
		for line in bbox_file:
			face_path, bbox = line.split(';')
			bbox = eval(bbox)
			aligned = alignmnet_model.align_image(bbox=bbox, image_path=face_path)
			im = Image.fromarray(aligned)
			im.save(face_path)
		os.remove(bbox_file_path)

def fillup_features_dict(people_features_dict, item, item_faces_vectors, personIDs):
	for i, faces_for_a_person in enumerate(item_faces_vectors):
		personID = personIDs[i]
		for face in faces_for_a_person:
			people_features_dict[personID][item].append(face)




def get_feature_list(people_features_dict):
	same_list = []
	diff_list = []
	people_with_all_their_faces = []
	for person in people_features_dict.keys():
		person_with_all_their_faces = []
		person_faces = people_features_dict[person]
		for item in person_faces.keys():
			item_faces = person_faces[item]
			for face in item_faces:
				person_with_all_their_faces.append(face)
		people_with_all_their_faces.append(person_with_all_their_faces)

	for person_with_all_their_faces in people_with_all_their_faces:
		for i, face in enumerate(person_with_all_their_faces):
			for another_face in person_with_all_their_faces[i+1:]:
				same_list.append([face, another_face])

	for i, person_with_all_their_faces in enumerate(people_with_all_their_faces):
		for another_person_with_all_their_faces in people_with_all_their_faces[i+1:]:
			for face in person_with_all_their_faces:
				for another_face in another_person_with_all_their_faces:
					diff_list.append([face, another_face])

	return same_list, diff_list

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def command_reader():
	provide_dataset_input = \
		input('provide dateset or generate dataset:(provide/generate)\n').strip()
	if provide_dataset_input == 'provide':
		provide_dataset = True
	elif provide_dataset_input == 'generate':
		provide_dataset = False
	else:
		raise Exception("invalid input, please input provide or generate")
	description = 'Test face recognition algorithm, you can provide your dataset or use detection algorithm to generate dataset'
	parser = argparse.ArgumentParser(description=description,
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-t', '--threshold', metavar='UNKNOWN THRESHOLD', required=True,
						nargs=1, type=str, help='the threshold to determine unknown')
	parser.add_argument('-rmw', '--recognition_model_weight', metavar='RECOGNITION MODEL WEIGHT', required=True,
						nargs=1, type=str, help='the weight file of the recognition model')
	parser.add_argument('-rmt', '--recognition_model_type', metavar='RECOGNITION MODEL TYPE', default=['keras'],
						nargs=1, type=str, help='the type of the recognition model')
	parser.add_argument('-bf', '--base_face', metavar='BASE FACE FOLDER', default=['./dataset/provided_dataset/base'],
						nargs=1, type=str, help='the folder for your base face')
	parser.add_argument('-tf', '--test_face', metavar='TEST FACE FOLDER', default=['./dataset/provided_dataset/test'],
						nargs=1, type=str, help='the folder for your test face')
	parser.add_argument('-ts', '--test_model_speed', metavar='TEST MODEL SPEED', default=[False],
						nargs=1, type=str2bool, help='test the speed of the model')

	if not provide_dataset:
		parser.add_argument('-dmt', '--detection_model_type', metavar='FACE DETECTION MODEL TYPE', required=True,
							nargs=1, type=str, help='the type of the model, example: mmod, vj')
		parser.add_argument('-dmw', '--detection_model_weight', metavar='FACE DETECTION MODEL WEIGHT', required=True,
							nargs=1, type=str, help='the weight file of the model')
		parser.add_argument('-dt', '--detector_threshold', metavar='FACE DETECTION THRESHOLD', default=[0.0],
							nargs=1, type=str, help='the threshold of the model')
		parser.add_argument('-isf', '--include_side_face', metavar='INCLUDE SIDE FACE', default=[False],
							nargs=1, type=str2bool, help='include side face in your base')
		parser.add_argument('-d', '--dataset', metavar='DATASET', default=['./dataset/kneron_dataset'],
							nargs=1, type=str, help='from which datatset you want to generate face data')
		parser.add_argument('-i', '--items', metavar='TEST_ITEMS', default=['lighting', 'with-glasses'],
							nargs='+', type=str, help="the items to be tested, example: all, base, blocking,\
									distance-angle, left-right-sideface, lighting, up-down-sideface, with-glasses\
													can be combination of them separated by whitespace ")
		parser.add_argument('-a', '--alignment', metavar='ALIGNMENT', nargs=1, type=str2bool, default=[True], help='do you want to do alignment')
		parser.add_argument('-amt', '--alignment_model_type', metavar='ALIGNMENT MODEL TYPE', nargs=1, type=str, default=['onet'],
							help='the type of your alignment model, example: onet')
		parser.add_argument('-amw', '--alignment_model_weight', metavar='ALIGNMENT MODEL WEIGHT', nargs=1, type=str, default=['./models/onet/Onet_L.hdf5'],
							help='the weight for your alignment model')
	parser.add_argument('-dig', '--display_interactive_graph', metavar='DISPLAY INTERACTIVE GRAP', default=[True],
						nargs=1, type=str2bool, help='display the effect of different thresholds on accept/reject rate')
	parser.add_argument('-dcm', '--display_confusion_matrix', metavar='DISPLAY CONFUSION MATRIX', default=[True],
						nargs=1, type=str2bool, help='display confusion matrix')
	parser.add_argument('-m', '--mirror', metavar='MIRROR INPUT IMAGE', default=[True],
						nargs=1, type=str2bool, help='mirror input image')

	parser.print_help()

	readline.set_completer_delims(' \t\n;')
	readline.parse_and_bind("tab: complete")
	readline.set_completer(complete)
	input_args = input().split(' ')

	command = parser.parse_args(input_args)
	return provide_dataset, command


def get_alignment_model(command):
	alignment_model_type = command.alignment_model_type[0]
	alignment_model_weight = command.alignment_model_weight[0]
	if alignment_model_type == 'onet':
		alignmnet_model = onet.ONET(alignment_model_weight, alignment_model_type)
	else:
		raise Exception("unsupported alignment model type")
	return alignmnet_model

def get_detection_model(command):
	detection_model_type = command.detection_model_type[0]
	detection_model_weight = command.detection_model_weight[0]
	detector_threshold = float(command.detector_threshold[0])
	if detection_model_type == 'mmod':
		detection_model = mmod.MMOD(detection_model_weight, detector_threshold, detection_model_type)
	elif detection_model_type == 'vj':
		# 'haarcascade_frontalface_default.xml'
		detection_model = vj.VJ(detection_model_weight, detector_threshold, detection_model_type)
	else:
		raise Exception('unsupported face detection model: ' + detection_model_type)
	return detection_model

def get_recognition_model(command):
	recognition_model_type = command.recognition_model_type[0]
	recognition_model_weight = command.recognition_model_weight[0]
	if recognition_model_type == 'keras':
		model_type = recognition_model_weight.split('/')[-1]
		recognition_model = keras_recognition.KerasModel(recognition_model_weight, model_type)
	else:
		raise Exception('unsupported face recognition model: ' + recognition_model_type)
	return recognition_model

class DisplayResult:
	def __init__(self, unknown_threshold, total_count, unknown_count,
				 mismatch_count, match_count, mismatched):
		self.unknown_threshold = unknown_threshold
		self.total_count = total_count
		self.unknown_count = unknown_count
		self.mismatch_count = mismatch_count
		self.match_count = match_count
		self.mismatched = mismatched



def write_and_print_result(result):
	lines = []
	lines.append("S test result summary\n")
	lines.append("===============================================================================================\n")
	lines.append("threshold: %f\n" % result.unknown_threshold)
	lines.append("count_all: %d\n" % result.total_count)
	lines.append("count_unkonwn: %d\n" % result.unknown_count)
	lines.append("count_mismatch: %d\n" % result.mismatch_count)
	lines.append("count_match: %d\n" % result.match_count)
	lines.append("unkonwn rate: %f\n" % (result.unknown_count * 1.0 / result.total_count))
	lines.append("mismatch rate: %f\n" % (result.mismatch_count * 1.0 / result.total_count))
	lines.append("match rate: %f\n" % (result.match_count * 1.0 / result.total_count))
	lines.append("===============================================================================================\n")
	lines.append("M mismatched\n")
	lines.append("===============================================================================================\n")

	for line in lines:
		print(line, sep='')

	if result.mismatch_count == 0:
		lines.append("None\n")
	else:
		for test_person_index, test_face_index, base_person_index, base_face_index, distance in result.mismatched:
			lines.append('test ID:%s, image:%d, mismatched with base ID:%s, image:%d, distance:%f\n' % (
			testPersonIDs[test_person_index], test_face_index, basePersonIDs[base_person_index], base_face_index,
			distance))
	lines.append("===============================================================================================\n")
	for line in lines:
		res_file.write(line)


	res_file.write("A all test image distances distribution over base\n")
	res_file.write("===============================================================================================\n")
	if display_every_test_image_distribution:
	    res_file.write(tabulate(all_face_distance_over_base,
	                            headers=['all test image'] + basePersonIDs))
	res_file.write("\n")
	res_file.write("===============================================================================================\n")

	res_file.write("V average test image distances distribution over base\n")
	res_file.write("===============================================================================================\n")
	if display_average_test_image_distribution:
	    res_file.write(tabulate(average_face_distance_over_base, headers=['average test image'] + basePersonIDs))
	    visualize_distance_distribution(average_face_distance_over_base, 'average_test_image_distribution',
	                                    basePersonIDs, result_folder + item_folder + '+' + test_result_filename)
	res_file.write("\n")
	res_file.write("===============================================================================================\n")





if __name__ == "__main__":
	result_folder = './recognition_result'

	provide_dataset, command = command_reader()

	unknown_threshold = float(command.threshold[0])
	base_folder = command.base_face[0]
	test_folder = command.test_face[0]

	if not provide_dataset:
		detection_model = get_detection_model(command)

		alignment = command.alignment[0]
		if alignment:
			alignmnet_model = get_alignment_model(command)




		base_include_sideface = command.include_side_face[0]
		data_folder = command.dataset[0]
		test_items = command.items
		if 'all' in test_items:
			test_items = item_list

		overwrite_dir(base_folder)
		overwrite_dir(test_folder)

		print('using detection algorithm to generate base and test...')
		faceCropper = crop_detected_face.faceCropper(detection_model.threshold, detection_model)
		faceCropper.generate_base(base_include_sideface, data_folder, base_folder)
		faceCropper.generate_test(data_folder, test_items, test_folder)
		if alignment:
			print('using alignment algorithm to align base and test faces...')
			align_base_and_test(base_folder, test_folder, alignmnet_model)
		detection_model_name = detection_model.model_type
	else:
		detection_model_name = 'provided_dataset'


	#################################################################
	## Do you want to get the distance distribution
	#################################################################
	display_every_test_image_distribution = command.display_confusion_matrix[0]
	display_average_test_image_distribution = command.display_confusion_matrix[0]


	if not os.path.exists(base_folder):
		os.makedirs(base_folder)
	if not os.path.exists(test_folder):
		os.makedirs(test_folder)
	if not os.path.exists(result_folder):
		os.makedirs(result_folder)

	recognition_model = get_recognition_model(command)
	mirror = command.mirror[0]

	test_result_filename = detection_model_name + '+' + ''.join(recognition_model.model_type.split('.')[:-1])
	if not result_folder[-1] == '/':
		result_folder += '/'
	res_file = open(result_folder + test_result_filename + '.txt', 'w')


	##############################
	### test speed of the model
	##############################
	test_speed = command.test_model_speed[0]
	if test_speed:
		background = Image.new("RGB", (640, 480))
		image_num = 500
		used_time = recognition_model.test_speed(background, image_num)
		print('used', used_time, 's to run', image_num, 'images')
		res_file.write('used' + str(used_time) + 's to run' + str(image_num) + 'images\n')

	##############################
	### test different items on the model
	##############################
	display_interactive_graph = command.display_interactive_graph[0]
	
	if mirror:
		recognition_model.bottleneck *= 2
	# shape  = (persons, images, vector_dims) // (20, 15, 512) // (15 is base include sideface)
	base_face_vectors, basePersonIDs = generate_face_vectors_in_folder(base_folder, recognition_model, mirror)

	if display_interactive_graph:
		people_features_dict = defaultdict(lambda: defaultdict(list))
		fillup_features_dict(people_features_dict, 'base', base_face_vectors, basePersonIDs)

	for item_folder in os.listdir(test_folder):
		print(item_folder)
		res_file.write("I item %s\n" % item_folder)
		res_file.write("===============================================================================================\n")

		item_folder_path = os.path.join(test_folder, item_folder)
		# shape  = (persons, images, vector_dims) // (20, 20, 512)
		test_item_vectors, testPersonIDs = generate_face_vectors_in_folder(item_folder_path, recognition_model, mirror)
		if display_interactive_graph:
			fillup_features_dict(people_features_dict, item_folder, test_item_vectors, testPersonIDs)

		unknown_count = 0
		match_count = 0
		mismatch_count = 0
		total_count = 0
		all_face_distance_over_base = []
		average_face_distance_over_base = []
		person_num = test_item_vectors.shape[0]
		face_num = test_item_vectors.shape[1]
		mismatched = []
		base_face_vectors, basePersonIDs = sort_base_according_to_test(base_face_vectors, basePersonIDs, testPersonIDs)
		for test_person_index, test_person_vector in enumerate(test_item_vectors):
			testPersonID = testPersonIDs[test_person_index]
			for test_face_index, test_face_vector in enumerate(test_person_vector):
				distance_over_base = find_distance_distribution_over_base(base_face_vectors, test_face_vector)
				all_face_distance_over_base.append([testPersonID + '_img_' + str(test_face_index)] + [row[0] for row in distance_over_base])
				base_person_index, base_face_index, distance = find_closest_base(distance_over_base)
				if distance > unknown_threshold:
					unknown_count += 1
				else:
					if testPersonID == basePersonIDs[base_person_index]:
						match_count += 1
					else:
						mismatched.append((test_person_index, test_face_index, base_person_index, base_face_index, distance))
						mismatch_count += 1
				total_count += 1
			face_distance_over_base_per_person = np.array(all_face_distance_over_base)[test_person_index*face_num:(test_person_index+1)*face_num, 1:].astype(float)
			average_face_distance_over_base_per_person = np.average(face_distance_over_base_per_person, axis=0)
			average_face_distance_over_base.append([testPersonID] + list(average_face_distance_over_base_per_person))
		result = DisplayResult(unknown_threshold, total_count, unknown_count,
		 mismatch_count, match_count, mismatched)
		write_and_print_result(result)
	print(display_interactive_graph)
	if display_interactive_graph:
		same_list, diff_list = get_feature_list(people_features_dict)
		print('number of pairs in same person list', len(same_list))
		print('number of pairs in different person list', len(diff_list))
		# save data
		print('Saving feature data...')
		output = result_folder + test_result_filename + '.npz'
		data_file = open(output, 'wb')
		np.savez(data_file, diff=diff_list, same=same_list, model=recognition_model.model_type, shape='N/A',
				 same_file='N/A', diff_file='N/A', image_dir=base_folder + ', ' + test_folder, lib='N/A')
		print('Data saved to {}'.format(output))

		import tool_scripts.kfrep as kfrep

		# input_args = [output, '-t', unknown_threshold, '-go', result_folder, '-to', result_folder, '-di', '']
		#
		# command = parser.parse_args(input_args)

		kfrep.main(output, unknown_threshold, result_folder + test_result_filename)



	res_file.close()











