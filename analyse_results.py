import numpy as np

action_classes = [x.strip() for x in open('models/i3d/label_map.txt')]
res = np.load('out/i3d/i3d-ViolentHumanActions_v0-joint.npz')
preds = res['preds']
truth_labels_txt = res['truth_labels']
truth_labels = np.zeros([truth_labels_txt.shape[0],1])

for i in range(truth_labels.shape[0]):
    truth_labels[i] = int( action_classes.index(truth_labels_txt[i]) )
    


def topN_acc(preds, truth_labels, n=1):
    top_preds = np.argsort(preds, 1)
    topN_preds = top_preds[:,-n:,:]
    is_topN = np.zeros([preds.shape[0], preds.shape[2]])
    for stream in range(preds.shape[2]):
        for i in range(preds.shape[0]):
            is_topN[i,stream] = np.isin(truth_labels[i], topN_preds[i,:,stream])
        
    return np.sum(is_topN,0) / is_topN.shape[0]

def confusion_matrix(preds, truth_labels, stream='rgb'):
    stream_idx = {'rgb': 0, 'flow': 1, 'joint': 2}
    conf_matrix = np.zeros([400,400])
    top_preds = np.argmax(preds, 1)
    for i in range(truth_labels.shape[0]):
        conf_matrix[int(truth_labels[i,0]),top_preds[i,stream_idx[stream]]] += 1
    
    return conf_matrix

def summary_confmat(conf_mat, truth_labels, k=10):
    _summary_confmat = []
    for x in np.unique(truth_labels).astype(int):
        sorted_args = np.flip( np.argsort(conf_matrix_rgb[x,:])[-10:],0 )
        sorted_values = np.flip( np.sort(conf_matrix_rgb[x,:])[-10:],0 )
        _summary_confmat.append([action_classes[x], [np.take(action_classes, sorted_args), sorted_values]])
    return _summary_confmat


