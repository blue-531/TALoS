import matplotlib.pyplot as plt
import itertools
import io
import numpy as np


def label_2_cmap(target,pred):
  pred_counts=np.zeros((20,20))
  for i in range(20):
    pred_i=np.where(target==i,pred,255)
    pred_id,counts=np.unique(pred_i,return_counts=True)
    
    for idx,pred_category in enumerate(pred_id):
      if pred_category==255:
        continue
      pred_counts[i,pred_category]=counts[idx]
  return pred_counts

def plot_confusion_matrix(cm, class_names =["unlabeled","car","bicycle","motorcycle","truck","other-vehicle","person","bicyclist","motorcyclist","road","parking","sidewalk","other-ground","building","fence","vegetation","trunk","terrain","pole","traffic-sign"]):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """ 
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=4)

  figure = plt.figure(figsize=(15, 15))
  plt.imshow(cm, cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=4)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  # plt.savefig(f'{fig_name}.png', dpi=300)
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  plt.close(figure)
  buf.seek(0)
  
  return buf

