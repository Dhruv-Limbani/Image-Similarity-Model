import cv2
import numpy as np
import os


class IdentityMetadata():
    def __init__(self, base, name, file):
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)
    
    
def distance(emb1, emb2, threshold=0.8):

    return np.sum(np.square(emb1 - emb2))


def recognize(img_path,metadata,embeddings,embedder):
    img = load_image(img_path)
    img = (img / 255.).astype(np.float32)
    img = cv2.resize(img, dsize = (224,224))
    embedding_vector = embedder.predict(np.expand_dims(img, axis=0),verbose=0)[0]
    distances = []
    names = []
    for i in range(len(embeddings)):
        dist = distance(embedding_vector,embeddings[i])
        if dist > 0:
            distances.append(dist)
            names.append(metadata[i].name)
    if distances:
        min_dist = min(distances)
        return names[distances.index(min_dist)]
    return "No Match Found"

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

def show_pair(idx1, idx2, metadata, embeddings):
    plt.figure(figsize=(8,3))
    plt.suptitle(f'Distance between {idx1} & {idx2}= {distance(embeddings[idx1], embeddings[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()))
    
    
def recognize_face(img_path, embedder, pipeline, le):
    img = load_image(img_path)
    img = (img / 255.).astype(np.float32)
    img = cv2.resize(img, dsize = (224,224))
    embedding_vector = embedder.predict(np.expand_dims(img, axis=0),verbose=0)[0]
    return le.inverse_transform(pipeline.predict([embedding_vector]))[0][5:]
  
