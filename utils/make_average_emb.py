import numpy as np

def avg_embedding(embeddings):
  """
  This function takes a list of embeddings and returns the average embedding.

  Args:
      embeddings: A list of numpy arrays, where each array represents an embedding.

  Returns:
      A numpy array representing the average embedding.
  """


  # Get the average embedding by dividing the sum by the number of embeddings
  avg_embedding = np.mean(embeddings, axis=0)


  return avg_embedding

