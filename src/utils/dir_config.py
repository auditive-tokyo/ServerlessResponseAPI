import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VECTOR_DIR = os.path.join(ROOT_DIR, "vector")

VECTORS_PATH = os.path.join(VECTOR_DIR, "vectors.npy")
REFERENCE_PATH = os.path.join(VECTOR_DIR, "reference.json")