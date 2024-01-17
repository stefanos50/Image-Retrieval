import os
import re
import cv2
from scipy.spatial.distance import cdist
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet50,vgg16,resnet101,vgg19
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score
import faiss


def load_and_preprocess_images(image_paths):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preprocessed_images = [transform(Image.open(image_path).convert('RGB')) for image_path in image_paths]
    return torch.stack(preprocessed_images)

def get_features(image_tensor,model,model_name):
    if 'vgg' in model_name:
        model = model.features
    else:
        model = torch.nn.Sequential(*(list(model.children())[:-1]))

    with torch.no_grad():
        image_tensor = image_tensor.to(device) #send data to GPU
        features = model(image_tensor)

    return features.squeeze()

def normalize_symetric_matrix(symetric_matrix):
    L = symetric_matrix.shape[0]

    for i in range(symetric_matrix.shape[0]):
        for j in range(i,symetric_matrix.shape[0]):
            symetric_matrix[i][j] = 2*L - (symetric_matrix[i][j] + symetric_matrix[j][i])
            symetric_matrix[j][i] = 2 * L - (symetric_matrix[i][j] + symetric_matrix[j][i])

    return symetric_matrix

def create_hypergraph(similarity_matrix, k):
    num_features = similarity_matrix.shape[0]
    hypergraph = []

    for i in range(num_features):
        # Get the indices of the top k most similar features (excluding self)
        top_k_indices = np.argsort(similarity_matrix[i, :])[:-k-1:-1]

        # Create a sublist with the top k indices
        hypergraph.append(top_k_indices.tolist())

    return hypergraph

def build_association_matrix(hypergraph, k):
    num_features = len(hypergraph)

    # Initialize the matrix R with zeros
    R = np.zeros((num_features, num_features))

    # Loop through each edge (hyperedge)
    for edge_index, edge in enumerate(hypergraph):
        # Loop through each vertex in the hyperedge
        for vertex_index in edge:
            # Check if the vertex is in the hyperedge
            if vertex_index in edge:
                # Calculate the value based on the formula R[i][v] = 1 - np.math.log(pos, k+1)
                pos = edge.index(vertex_index) + 1
                R[edge_index, vertex_index] = 1 - np.math.log(pos, k + 1)
            else:
                # If the vertex is not in the hyperedge, set the value to zero
                R[edge_index, vertex_index] = 0

    return R

def calculate_weights(hypergraph, association_matrix):
    weights = []

    for edge_index, edge in enumerate(hypergraph):
        weight = 0

        for vertex_index in edge:
            weight += association_matrix[edge_index, vertex_index]

        weights.append(weight)

    return weights

def calculate_hyperedge_similarities(association_matrix):
    H = association_matrix
    Sh = np.dot(H, H.T)
    Sv = np.dot(H.T, H)
    S = np.multiply(Sh, Sv)

    return S

def calculate_pairwise_values_list(hypergraph, association_matrix, weights_list):
    pairwise_values_list = []

    # Iterate through each hyperedge
    for edge_index, edge in enumerate(hypergraph):
        weight = weights_list[edge_index]

        # Calculate Cartesian product of nodes in the hyperedge
        node_pairs = list(product(edge, repeat=2))

        # Create a dictionary with pair as key and the specified value as the value
        pairwise_values = {}
        for pair in node_pairs:
            v1, v2 = pair
            value = weight * association_matrix[edge_index, v1] * association_matrix[edge_index, v2]
            pairwise_values[pair] = value

        # Append the dictionary to the list
        pairwise_values_list.append(pairwise_values)

    return pairwise_values_list

def calculate_c_matrix(pairwise_values_list, hyperedges):
    c = np.zeros((len(hyperedges), len(hyperedges)))

    for edge_index, pairwise_values in enumerate(hyperedges):
        for (v1, v2) in pairwise_values_list[edge_index]:  # for each pair in the dict
            c[v1][v2] += pairwise_values_list[edge_index][(v1, v2)]

    return c

def extract_class_number(file_path):
    pattern = re.compile(r'class_(\d+)')
    match = re.search(pattern, file_path)

    if match:
        class_number = int(match.group(1))
        return class_number
    else:
        return None


"-------------- PARAMETERS ---------------------"
load_finetuned = 'best_finetuned.pth' #best_finetuned.pth
k_neighbors = 10
num_iters = 3
top_k_results = 5
images_directory = "A:\\animals" #https://www.kaggle.com/datasets/alessiocorrado99/animals10
feature_extraction_model = 'resnet50' #resnet50,resnet101,vgg16,vgg19
similarity_metric = 'euclidean' #euclidean,inv_euclidean,cosine,manhatan,minkowski
p = 3
algorithm = 'paper' #paper, faiss

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4
text_size_percentage = 5
font_thickness = 2
font_color = (235, 52, 52)
text_position = (0, 15)
"------------------------------------------------"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if feature_extraction_model == 'resnet50':
    model = resnet50(pretrained=True).to(device) #run on GPU
elif feature_extraction_model == 'resnet101':
    model = resnet101(pretrained=True).to(device)
elif feature_extraction_model == 'vgg16':
    model = vgg16(pretrained=True).to(device)
elif feature_extraction_model == 'vgg19':
    model = vgg19(pretrained=True).to(device)

model.eval()


class_labels = []
image_paths = []
class_names = ['dog','cat','elephant','butterfly','squirrel','sheep']
for root, dirs, files in os.walk(images_directory):
    for file in files:
        file_path = os.path.join(root, file)
        image_paths.append(file_path)
        class_labels.append(extract_class_number(file_path))
class_labels = np.array(class_labels)

if load_finetuned is not None:
    model.fc = torch.nn.Linear(model.fc.in_features, max(class_labels)+1)
    model.load_state_dict(torch.load(load_finetuned))

preprocessed_images = load_and_preprocess_images(image_paths)
features = get_features(preprocessed_images,model,feature_extraction_model).detach().cpu().numpy()

print("Shape of the features:", features.shape)

index = None
if algorithm == 'faiss':
    features = features / np.sqrt(np.sum(np.square(features), axis=1, keepdims=True))
    index = faiss.IndexFlatL2(features.shape[1])  # L2 (Euclidean) distance index
    index.add(features)
else:
    #create a cosine similarity symetric matrix
    distance_matrix = np.zeros((features.shape[0], features.shape[0]))
    for i in range(features.shape[0]):
        for j in range(i + 1, features.shape[0]):
            if similarity_metric == 'euclidean':
                distance = np.linalg.norm(features[i] - features[j])
            elif similarity_metric == 'inv_euclidean':
                distance = 1/(np.linalg.norm(features[i] - features[j])+0.0000001) #we add a small value to avoid getting nan values if distance is 0
            elif similarity_metric == 'manhatan':
                distance = np.sum(np.abs(features[i] - features[j]))
            elif similarity_metric == 'minkowski':
                distance = np.power(np.sum(np.abs(features[i] - features[j]) ** p), 1 / p)
            else:
                break

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    if similarity_metric == 'cosine':
        sim_matrix = cosine_similarity(features, features)
    else:
        sim_matrix = distance_matrix
    norm_sim_matrix = normalize_symetric_matrix(sim_matrix)

    for i in range(num_iters):
        hypergraph = create_hypergraph(norm_sim_matrix,k_neighbors)
        association_matrix = build_association_matrix(hypergraph,k_neighbors)
        weights = calculate_weights(hypergraph,association_matrix)
        hyperedge_similarities = calculate_hyperedge_similarities(association_matrix)
        pairwise_values = calculate_pairwise_values_list(hypergraph,association_matrix,weights)
        c_matrix = calculate_c_matrix(pairwise_values,hypergraph)
        affine = np.multiply(c_matrix,hyperedge_similarities)
        norm_sim_matrix = affine

query_results = []
ground_truth_results = []

#for all the images of the dataset calculate the precision and recall based on the top k retrieved images
#note that we already know the ground truth labels of each image
for image in range(features.shape[0]):
    if algorithm == 'faiss':
        query_vector = features[image].reshape(1, -1)  # Example query vector
        _, indices = index.search(query_vector, top_k_results)
        indices_of_k_highest = indices[0]
    else:
        indices_of_k_highest = np.argsort(norm_sim_matrix[image])[-top_k_results:][::-1]

    original_label = class_labels[image]
    predicted_labels = class_labels[indices_of_k_highest]

    ground_truth_results.append([original_label])
    query_results.append(list(predicted_labels))


num_classes = max(class_labels) + 1
true_labels = [np.isin(np.arange(num_classes), classes) for classes in ground_truth_results]
predicted_labels_top5 = [np.isin(np.arange(num_classes), classes) for classes in query_results]

precision = precision_score(true_labels, predicted_labels_top5, average='micro')
recall = recall_score(true_labels, predicted_labels_top5, average='micro')

print(f"Precision: {precision}")
print(f"Recall: {recall}")

#show in a cv2 window for all the images of the dataset the top k retrieved images
for image_id in range(features.shape[0]):
    if algorithm == 'faiss':
        query_vector = features[image_id].reshape(1, -1)  # Example query vector
        _, indices = index.search(query_vector, top_k_results)
        indices_of_k_highest = indices[0]
    else:
        indices_of_k_highest = np.argsort(norm_sim_matrix[image_id])[-top_k_results:][::-1]

    images_results = []
    images_results.append(cv2.imread(image_paths[image_id]))

    if not algorithm == 'faiss':
        highest_values = norm_sim_matrix[image_id,indices_of_k_highest]

    for i in range(len(indices_of_k_highest)):
        image = cv2.imread(image_paths[indices_of_k_highest[i]])
        images_results.append(image)

    cnt = 0
    images_results = [cv2.resize(img, (180, 180)) for img in images_results]
    for img in images_results:
        if cnt == 0:
            cv2.putText(img, "Query Image - "+str(class_names[class_labels[image_id]]), text_position, font, font_scale, font_color, font_thickness)
        else:
            if algorithm == 'faiss':
                cv2.putText(img,str(class_names[class_labels[indices_of_k_highest[cnt-1]]]), text_position, font, font_scale, font_color, font_thickness)
            else:
                cv2.putText(img, str(round(highest_values[cnt - 1], 2)) + " - " + str(class_names[class_labels[indices_of_k_highest[cnt - 1]]]), text_position, font, font_scale,font_color, font_thickness)
        cnt += 1

    combined_image = np.hstack(images_results)
    cv2.imshow('Retrieved Images for image: '+image_paths[image_id], combined_image)
    key = cv2.waitKey(0)
    if key == 100:
        cv2.destroyAllWindows()